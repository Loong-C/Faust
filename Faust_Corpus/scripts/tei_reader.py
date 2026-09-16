"""Source-faithful TEI extraction. No downloads, model calls or source edits."""
from __future__ import annotations
import collections, hashlib, re
import xml.etree.ElementTree as ET
TEI = 'http://www.tei-c.org/ns/1.0'
XML = 'http://www.w3.org/XML/1998/namespace'
NS = {'t': TEI}
SKIP = {'note','castList','titlePage','docTitle','docAuthor','docImprint','fw','pb','cb'}
REVIEW_SKIPS = {'note'}
DIVS = {'div','div1','div2','div3','div4','div5','div6','div7'}
LEAF_TEXT = {'l','p','ab','speaker','stage','head','trailer'}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha_text(text: str) -> str:
    return sha_bytes(text.encode('utf-8'))

def tag(el: ET.Element) -> str:
    return el.tag.rsplit('}', 1)[-1] if isinstance(el.tag, str) else ''

def norm(text: str) -> str:
    """Remove XML layout whitespace, not spelling, punctuation, or Unicode."""
    return re.sub(r'[\t\r\n ]+', ' ', text).strip()

def parse_xml(data: bytes) -> ET.Element:
    if len(data) > 25_000_000:
        raise ValueError('Unexpectedly large TEI file')
    if re.search(br'<!\s*(DOCTYPE|ENTITY)\b', data, re.I):
        raise ValueError('DTD/entity declarations are not accepted by this parser')
    root = ET.fromstring(data)
    if root.tag != f'{{{TEI}}}TEI':
        raise ValueError('Expected a TEI P5 document, not HTML or an error response')
    if root.find('t:text', NS) is None:
        raise ValueError('TEI contains no text element')
    return root

class TEIParser:
    """Extract immutable-source-linked atoms; never invent original line numbers."""
    def __init__(self, data: bytes, work_id: str):
        self.data = data
        self.root = parse_xml(data)
        self.work_id = work_id
        self.source_hash = sha_bytes(data)
        self.paths: dict[int, str] = {}
        self._index_paths(self.root, '/t:TEI')
        self.scenes: list[dict] = []
        self.atoms: list[dict] = []
        self.exclusions: list[dict] = []
        self.editorial: list[dict] = []
        self._excluded_paths: set[str] = set()
        self._editorial_paths: set[str] = set()
        self.verse_counter = 0
        self.eligible_line_paths: list[str] = []
        self.person_ids = {x.attrib.get(f'{{{XML}}}id') for x in self.root.iter()
                           if tag(x) in {'person', 'personGrp'}} - {None}

    def _index_paths(self, el: ET.Element, path: str) -> None:
        self.paths[id(el)] = path
        counts: collections.Counter = collections.Counter()
        for child in el:
            counts[tag(child)] += 1
            self._index_paths(child, f'{path}/t:{tag(child)}[{counts[tag(child)]}]')

    def _exclude(self, el: ET.Element) -> None:
        path = self.paths[id(el)]
        if path not in self._excluded_paths:
            self._excluded_paths.add(path)
            self.exclusions.append({'work_id': self.work_id, 'xpath': path,
                                   'tag': tag(el), 'reason': 'non_dialogue_or_editorial',
                                   'requires_review': tag(el) in REVIEW_SKIPS,
                                   'text': norm(''.join(el.itertext()))})

    def _selected(self, el: ET.Element) -> list[ET.Element]:
        if tag(el) not in {'choice', 'app'}:
            return list(el)
        preferences = (['orig', 'sic', 'abbr', 'corr', 'reg', 'expan']
                       if tag(el) == 'choice' else ['lem', 'rdg'])
        selected = next((child for name in preferences for child in el if tag(child) == name), None)
        if selected is None:
            raise ValueError(f'Unsupported editorial alternatives at {self.paths[id(el)]}')
        path = self.paths[id(el)]
        if path not in self._editorial_paths:
            self._editorial_paths.add(path)
            self.editorial.append({'work_id': self.work_id, 'xpath': path,
                                   'type': tag(el), 'selected_tag': tag(selected),
                                   'selected_xpath': self.paths[id(selected)],
                                   'requires_review': True})
        return [selected]

    def inline(self, el: ET.Element) -> str:
        if tag(el) in SKIP:
            self._exclude(el)
            return ''
        if tag(el) == 'lb':
            return '\n'
        if tag(el) in {'choice', 'app'}:
            return ''.join(self.inline(c) for c in self._selected(el))
        result = el.text or ''
        for c in el:
            result += self.inline(c) + (c.tail or '')
        return result

    def _effective(self, el: ET.Element):
        if tag(el) in SKIP:
            return
        yield el
        for c in self._selected(el):
            yield from self._effective(c)

    def events(self, el: ET.Element, stanza_ids: list[str] | None = None) -> list[dict]:
        stanza_ids = list(stanza_ids or [])
        name = tag(el)
        if name in SKIP:
            self._exclude(el)
            return []
        if name in {'choice', 'app'}:
            return [event for c in self._selected(el) for event in self.events(c, stanza_ids)]
        if name == 'lg':
            stanza_ids.append(self.paths[id(el)])
        if name in LEAF_TEXT:
            text = norm(self.inline(el))
            if not text:
                return []
            event = {'kind': {'l': 'verse', 'p': 'prose', 'ab': 'prose'}.get(name, name),
                     'text': text, 'source_xpath': self.paths[id(el)],
                     'source_xml_id': el.attrib.get(f'{{{XML}}}id'),
                     'source_n': el.attrib.get('n'), 'source_part': el.attrib.get('part'),
                     'stanza_ids': stanza_ids}
            if name == 'l':
                self.verse_counter += 1
                event['verse_index_in_work'] = self.verse_counter
                self.eligible_line_paths.append(self.paths[id(el)])
            nested_stages = [norm(self.inline(c)) for c in el.iter()
                             if c is not el and tag(c) == 'stage']
            if nested_stages:
                event['inline_stage_directions'] = nested_stages
            return [event]
        out = []
        if norm(el.text or ''):
            out.append({'kind': 'unclassified_text', 'text': norm(el.text or ''),
                        'source_xpath': self.paths[id(el)] + '/text()',
                        'stanza_ids': stanza_ids, 'requires_review': True})
        for child in self._selected(el):
            out.extend(self.events(child, stanza_ids))
            if norm(child.tail or ''):
                out.append({'kind': 'unclassified_text', 'text': norm(child.tail or ''),
                            'source_xpath': self.paths[id(child)] + '/following-sibling::text()[1]',
                            'stanza_ids': stanza_ids, 'requires_review': True})
        return out

    @staticmethod
    def render_events(events: list[dict]) -> str:
        """Only add line/paragraph separators, never words, brackets, or labels."""
        parts = []
        previous = None
        for e in events:
            if parts:
                continuous_verse = (previous['kind'] == e['kind'] == 'verse' and
                                    previous.get('stanza_ids') == e.get('stanza_ids'))
                parts.append('\n' if continuous_verse else '\n\n')
            parts.append(e['text'])
            previous = e
        return ''.join(parts)

    def _scene(self, nodes: list[ET.Element], hierarchy: list[dict], region: str,
               scene_path: str, group_path: str | None) -> None:
        meaningful = [n for n in nodes if tag(n) != 'head' and tag(n) not in SKIP]
        for n in nodes:
            if tag(n) in SKIP:
                self._exclude(n)
        if not meaningful:
            return
        scene_index = len(self.scenes) + 1
        scene_id = f'{self.work_id}_S{scene_index:03d}'
        own_heads = [norm(self.inline(n)) for n in nodes if tag(n) == 'head']
        title = ' / '.join(x for x in own_heads if x) or next(
            (h['title'] for h in reversed(hierarchy) if h['title']), region)
        group = self.work_id + '_G_' + sha_text(group_path or scene_path)[:12]
        atoms = []
        for n in meaningful:
            events = self.events(n)
            if not events:
                continue
            # A normal <sp> is never broken into individual poem lines.
            who = n.attrib.get('who', '').split()
            speakers = [s.lstrip('#') for s in who]
            labels = [e['text'] for e in events if e['kind'] == 'speaker']
            line_events = [e for e in events if e['kind'] == 'verse']
            atom_id = f'{scene_id}_A{len(atoms) + 1:04d}'
            atom = {'atom_id': atom_id, 'work_id': self.work_id, 'scene_id': scene_id,
                    'source_xpath': self.paths[id(n)],
                    'source_xml_id': n.attrib.get(f'{{{XML}}}id'),
                    'kind': 'speech' if tag(n) == 'sp' else tag(n),
                    'speakers': speakers, 'speaker_labels': labels, 'events': events,
                    'text': self.render_events(events),
                    'verse_indices': [e['verse_index_in_work'] for e in line_events],
                    'source_line_numbers': [e.get('source_n') for e in line_events]}
            atom['text_sha256'] = sha_text(atom['text'])
            atoms.append(atom)
        if not atoms:
            return
        scene = {'scene_id': scene_id, 'work_id': self.work_id, 'scene_group_id': group,
                 'scene_index_in_work': scene_index, 'title': title,
                 'hierarchy': hierarchy, 'region': region, 'source_xpath': scene_path,
                 'source_sha256': self.source_hash,
                 'atom_ids': [a['atom_id'] for a in atoms],
                 'source_text_sha256': sha_text('\n\n'.join(a['text'] for a in atoms)),
                 'word_count': len(re.findall(r'\S+', '\n\n'.join(a['text'] for a in atoms)))}
        self.scenes.append(scene)
        self.atoms.extend(atoms)

    def _walk_sections(self, el: ET.Element, hierarchy: list[dict], region: str,
                       group_path: str | None = None) -> None:
        name = tag(el)
        if name in SKIP:
            self._exclude(el)
            return
        this_path = self.paths[id(el)]
        if name in DIVS:
            title = ' / '.join(norm(self.inline(x)) for x in el if tag(x) == 'head')
            h = {'title': title, 'type': el.attrib.get('type'),
                 'n': el.attrib.get('n'), 'source_xpath': this_path,
                 'source_xml_id': el.attrib.get(f'{{{XML}}}id')}
            hierarchy = hierarchy + [h]
            if group_path is None and (el.attrib.get('type') or '').casefold() in {'scene', 'scena', 'szene'}:
                group_path = this_path
        has_divs = any(tag(c) in DIVS for c in el)
        if not has_divs:
            self._scene(list(el), hierarchy, region, this_path, group_path)
            return
        pending = []
        block = 0
        for c in el:
            if tag(c) in DIVS:
                if pending:
                    block += 1
                    self._scene(pending, hierarchy, region, this_path + f'/block[{block}]', group_path)
                    pending = []
                self._walk_sections(c, hierarchy, region, group_path)
            else:
                pending.append(c)
        if pending:
            block += 1
            self._scene(pending, hierarchy, region, this_path + f'/block[{block}]', group_path)

    def run(self) -> tuple[list[dict], list[dict], dict]:
        text = self.root.find('t:text', NS)
        assert text is not None
        source_line_paths = []
        for region in ['front', 'body']:
            section = text.find('t:' + region, NS)
            if section is not None:
                source_line_paths.extend(self.paths[id(e)] for e in self._effective(section)
                                         if tag(e) == 'l' and norm(self.inline(e)))
                self._walk_sections(section, [], region)
        if not self.scenes or not self.atoms:
            raise ValueError('No usable scenes were extracted')
        if source_line_paths != self.eligible_line_paths:
            missing = sorted(set(source_line_paths) - set(self.eligible_line_paths))
            extra = sorted(set(self.eligible_line_paths) - set(source_line_paths))
            raise ValueError(f'Poem-line coverage mismatch; missing={missing[:5]}, extra={extra[:5]}')
        titles = [norm(''.join(x.itertext())) for x in self.root.findall('.//t:titleStmt/t:title', NS)]
        return self.scenes, self.atoms, {
            'work_id': self.work_id, 'title': ' / '.join(titles),
            'source_sha256': self.source_hash, 'source_root_xml_id': self.root.attrib.get(f'{{{XML}}}id'),
            'verse_element_count': len(source_line_paths), 'scene_count': len(self.scenes),
            'atom_count': len(self.atoms), 'exclusions': self.exclusions,
            'editorial_choices': self.editorial,
            'note': 'verse_index_in_work is generated order, NOT scholarly Faust verse numbering',
            'unknown_speaker_ids': sorted({p for a in self.atoms for p in a['speakers']}
                                          - self.person_ids) if self.person_ids else [],
        }
