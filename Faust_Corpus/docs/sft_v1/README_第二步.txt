第二步交付说明：faust-sft-0.1.0

交付范围

本版本包含195条逐段制作的说明—原文对，继承第一步的全部片段ID、文本、顺序和数据划分。158条用于训练，18条用于验证，19条用于测试。没有另行生成新的“仿歌德原文”，没有微调模型，也没有调用外部大模型API。内容说明由本次对话中的助手读取德文后逐段撰写并自查。

文件入口

data/sft_v1/faust_sft_pairs.jsonl保存全部195条审计主记录，含instruction_de、description_zh、target_text、结构化节拍和来源定位。它是全量对照档，不应直接整体送入训练。

splits/sft_v1/train.jsonl、validation.jsonl、test.jsonl是简化的prompt/completion格式；每条只有这两个字符串。同行顺序与同目录的train_index.jsonl、validation_index.jsonl、test_index.jsonl一一对应，索引保存segment_id与校验值。

review/sft_v1/annotations.jsonl是本次逐段写出的原始说明，data/sft_v1/scene_specs.jsonl是展开后的结构化版本。修改说明时应建立下一版，而不是覆盖这一版。docs/sft_v1/剧本说明_中文.txt方便不读德语时核查内容。

说明如何写成

form_de记录局部形式，situation_de记录开端状态，beats_de按出现顺序记录行动与言语行为，ending_state_de复用最后一个节拍。意图、情绪、转述与幻象均写在相关节拍中；没有凭空添加独立的人物心理档案。表述使用普通现代德语，不要求模型分析作者风格，也不复述歌德的名句、隐喻或独特韵式。献辞保留为抒情文本，歌曲、合唱和散文对白不改成普通对话。

instruction_de由上述德语内容加固定的输出要求组成。它不包含原文目标、后续原文、中文说明、片段ID或校验值。描述为本片段服务，必要时用场景元数据与前文辨认说话人，但不把后续结果冒写为当前片段发生的事情。

目标原文不动

target_text逐字匹配第一步的target_text，原有标点、人物标签和分行不变。source保留原文定位与source_issue_ids，原标注中的已知问题仍通过第一步的注释解释，不在训练目标中暗自修订。诗行索引仍是TEI元素顺序，不冒称通行版本的原作行号。

中文对照

description_zh是与德语说明表达相同核心内容的中文摘要，不是逐字翻译，也不是训练输入。它不包含必须重新审批的任务清单；这批说明已经完成助手自查。文学解释仍可能有不同理解，不冒称德语文学专家校勘。

下一阶段使用

先固定底模，再核算实际prompt与completion的合计token长度、格式模板和EOS处理。训练时只在completion上计算损失；不将中文说明、质量报告或来源字段拼入prompt。不对过长原文目标静默截断。首次实验沿用本版158条训练数据，避免靠多次复制同一原文制造虚假独立样本。

验证集可用于选择配置；测试集不用于训练、检索示例或根据得分反复改提示。经典原著可能已存在于底模预训练中，因此保留原作场景的测试不能单独证明对新剧情的创作能力，之后仍须用真正原创场景评价。本版完成的是数据准备，不承诺微调必然改善文学质量。

复现

在Faust_Corpus目录执行python scripts/build_sft_v1.py，再执行python scripts/validate_sft_v1.py。均只读本地文件，无需密钥、网络或显卡。构建脚本不自行生成语义说明；它读取已交付的annotations.jsonl编译数据。输出可重现，但文学注释本身是这次制作所得，不宣称可由一个确定性程序重新推导。

质量与同步

本版校验报告在verification/sft_v1，词组重合明细在review/sft_v1/lexical_overlap.jsonl。相邻的plan目录负责版本与更新清单。第一步的SHA256SUMS.txt仍仅校验第一步的历史文件；本版新增文件以plan/releases/faust-sft-0.1.0/manifest.json为准。
