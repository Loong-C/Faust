《浮士德》LoRA准备版 faust-lora-prep-0.1.0

从哪里开始

最简单的入口是 training/Faust_LoRA_Colab.ipynb。通过仓库的Colab链接打开，在“运行时 → 更改运行时类型”选择NVIDIA GPU，然后运行全部单元。建议显存24GB或以上；16GB卡可以试跑，但没有显存适配保证。不要选择CPU或TPU。本地Mac不在这份CUDA脚本的支持范围内。

笔记本默认请求挂载你的Google Drive，只为持久化检查点。是否授权由你在Google界面决定。不授权时可将 SAVE_TO_DRIVE=False，但运行时回收会丢掉本地输出。公开GitHub仓库与公开Qwen底模通常不需要API密钥。本代码不建立付费实例、不购买GPU、不自动推送任何权重或文件。

Linux或WSL2用户可在仓库根目录执行：bash training/run.sh。需要已可用的NVIDIA驱动、Python 3.10–3.13（建议3.11/3.12）、git、至少35GiB可用磁盘和稳定网络。Windows用户不必安装CUDA工具链，使用Colab更简单。运行脚本会建立独立 .venv-faust，安装PyTorch 2.9.1的CUDA12.8版以及固定Transformers/PEFT等依赖，不修改系统Python或笔记本内核。安装后还会运行一个随机初始化的微型Qwen3 CPU集成测试，检查真实Trainer/PEFT的训练与存取。它不下载8B权重，也不能替代GPU试跑。

训练具体做什么

底模是 Qwen/Qwen3-8B-Base。用bitsandbytes NF4双重量化加载冻结底模，仅训练LoRA参数。不是全参数微调，也不是GRPO。本轮只使用现有158条训练对；18条验证对用于损失监控与早停，19条原著测试对不用于训练或选模型。预检可以检查测试文件的哈希和集合隔离，这不是将测试原文作为学习目标。

默认rank=16、alpha=32、dropout=0.05，目标为注意力q/k/v/o投影与MLP gate/up/down投影。学习率5e-5，最多3轮，单卡batch=1，梯度累积8，每10次优化器更新评估/保存，保留2个检查点，验证损失连续2次不改善则早停。约60次优化器更新是按158条、3轮计算的上限；实际以Trainer日志为准。保存的adapter来自最低验证损失的已保存检查点。最低验证损失不是文学质量最高的证明。

每条数据使用统一的纯文本 Aufgabe/Textauszug 分隔，不套用聊天模板，不加入think块。prompt与completion分别分词后拼接，推理采用相同prompt编码方式；只有原文和末尾EOS参与loss。padding和prompt标签为-100。原文不现代化、不改写，不把中文对照送进训练。EOS即使与PAD使用同一个ID也不会被误屏蔽。

4096是prompt+原文+EOS的完整预算，不是只给原文的预算。第一次运行会使用真实Qwen分词器统计长度并输出preflight.json；发现超限即报出segment_id，绝不会偷偷截断、遗漏或改写文本。本次准备环境无法下载并运行真实分词器，故没有伪造token长度报告。

模型固定到官方历史提交49e3418；新实验第一次运行将其解析为40位Hugging Face提交号，续训复用该提交。run_manifest同时记录数据哈希、分词器标识、代码哈希、配置、库版本和Git版本。改动影响训练的配置/代码/数据后，必须使用新的输出目录，不能覆盖原实验。

检查点与结果

一键脚本先使用最长训练/验证样本做两步smoke训练，输出至正式目录名后加-smoke的独立目录。smoke不继承到正式模型。通过后，正式训练重新加载原始底模并设置种子。训练输出默认为 runs/faust-lora-v0.1.0；Colab默认输出至 MyDrive/FaustTraining/faust-lora-v0.1.0。用FAUST_OUTPUT环境变量可更换目录。

中断后重新运行相同命令。只恢复同目录中拥有模型、优化器、调度器、随机状态和Trainer状态的完整检查点；不存在完整检查点时不会伪装续训。不完整检查点不会被选中。同目录已经完成时不重复训练。resume依赖原输出目录、相同配置、代码与软件版本；跨GPU精度变化时请新建实验。

训练后adapter/保存小型适配器和分词器；run_manifest.json记录来源；metrics.json保存训练/验证损失与实测峰值显存；base_validation.json保存未更新LoRA之前的验证损失；preflight.json包含每条训练/验证样本真实token长度；checkpoint-*用于中断恢复。不要把这些大文件提交到普通Git仓库，也不要把adapter目录当成完整8B底模。

原创评测

运行 .venv-faust/bin/python training/compare.py --run <运行目录>，默认比较前3个原创dev情景的同量化底模与LoRA，二者使用相同prompt和采样设置。输出有中文情景和打乱来源的德文候选，映射保存在单独的blind_key.json。生成达到max_new_tokens的样本会标记hit_max_new_tokens，不冒称自然结束。

正式比较全部12个dev情景可加 --limit 0 --seeds 20260916 20260917。可选 --reference 会加入底模+原文参考、LoRA+原文参考两个条件，参考只能来自训练集，使用简单的形式匹配，不声称语义检索。这不是训练时增加任务。4个原创holdout情景用 --set holdout --limit 0 单独执行，建议只在固定最终方案后查看。原著19条test仍另行保留。

8词组匹配只是检测直接复用的一项诊断，不是“AI味”或文学质量的评分。你不必懂德语：把generations.jsonl、blind_review.txt和run_manifest.json交回对话，可以逐例解释、比较。当前目标是德文风格试验；这次未训练中文译写器，也不保证195段样本足以达到歌德的文学质量。

出错处理

没有GPU：切换运行时，确认nvidia-smi正常。显存不足：更换显存更大的GPU/高内存运行时，不要削掉原文。内存RAM不足：使用高内存运行时；QLoRA只降低显存负担，并不保证低RAM。磁盘不足：清理无用模型缓存，保留检查点。网络下载失败或429：稍后在同一环境重试或使用你自己的受支持下载配置；不要将密钥贴入仓库。数据哈希不符：上传run_manifest/报错与plan/STATE.json，先核对版本。不要手改锁文件绕过检查。

验证边界

本次已执行真实数据一致性测试、分隔/掩码/截断防护测试、恢复选择测试、CPU张量与反向传播测试、Notebook格式与Python语法检查。当前准备环境没有GPU且无法下载模型依赖，因此没有实际执行8B加载、bitsandbytes CUDA内核、Trainer训练或Colab整本运行。上述GPU路径由你启动时的smoke检查验证；准备完成不等于模型已训练成功。

依据（核查于2026-09-16）

https://huggingface.co/Qwen/Qwen3-8B-Base
https://huggingface.co/docs/transformers/v4.57.1/en/quantization/bitsandbytes
https://huggingface.co/docs/peft/v0.18.0/en/developer_guides/quantization
https://pypi.org/project/transformers/4.57.6/
https://pypi.org/project/peft/0.18.1/
https://pypi.org/project/accelerate/1.12.0/
https://pypi.org/project/bitsandbytes/0.49.2/
https://pytorch.org/get-started/previous-versions/
https://research.google.com/colaboratory/faq.html

这些来源支持模型/API和运行时事实；具体超参数是本项目的保守起点，不是论文证明的最佳参数。
