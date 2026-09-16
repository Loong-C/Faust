Faust：原著数据与LoRA基线

当前新增准备版本：faust-lora-prep-0.1.0。原著切分和SFT数据保持冻结，195条管理记录按158训练／18验证／19测试划分。尚未进行真实8B GPU训练。

[在Colab打开训练入口](https://colab.research.google.com/github/Loong-C/Faust/blob/training/lora-v0.1.0/training/Faust_LoRA_Colab.ipynb)

先选择GPU运行时（建议24GB或以上显存），然后运行全部。默认请求Google Drive授权保存检查点。两步试跑成功后才从原始底模开始正式训练。付费运行时由你自行选择，本仓库不创建或购买算力。16GB只作尝试，不承诺足够。

本地Linux/WSL2可执行 `bash training/run.sh`。详见 [中文训练说明](training/README_训练说明.txt)。

底模为Qwen3-8B-Base，4位NF4 QLoRA，仅对原文completion与EOS计算loss。主数据和分词长度均有检查，超长样本报错而不截断。未训练中文模型。模型权重、密钥和运行输出不应提交到普通Git。

[训练准备验证记录](training/verification.json) 区分已跑的离线检查和未跑的GPU/Colab检查。训练后请提供运行目录中的run_manifest.json、metrics.json及比较输出进行下一轮分析。

Faust_Corpus/保留原始语料与说明；training/保存训练配置和脚本；eval/保存不进入训练的原创评测情景；plan/保存项目状态；models/只放模型登记说明，实际adapter在运行目录或另行托管。
