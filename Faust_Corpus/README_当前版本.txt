当前交付：faust-sft-0.1.0

这是以 corpus-reviewed-v2 为固定底稿完成的第二步。原著数据和原有说明均保持原样；旧的 README_请先读.txt 与 verification 中第一步的报告属于历史记录，不表示第二步尚未完成。

195个片段已分别配上现代德语戏剧说明、中文内容对照与不改动的原文目标。详细说明请看 docs/sft_v1/README_第二步.txt；只想阅读时打开 docs/sft_v1/剧本说明_中文.txt。

主数据在 data/sft_v1/faust_sft_pairs.jsonl。训练请使用 splits/sft_v1/train.jsonl；validation.jsonl与test.jsonl分别留作验证和测试。不要将含195条的主数据整体当训练集。

仓库层面的当前版本以同级 plan/STATE.json 为准。同步方法见 plan/仓库管理与同步.txt。
