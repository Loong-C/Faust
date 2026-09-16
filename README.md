Faust：原著数据与本机QLoRA实验

当前本机准备版本：faust-local8gb-0.2.0。数据仍为faust-sft-0.1.0，195条管理记录按158训练／18验证／19测试划分。语料没有改动，尚未进行实际8B GPU训练。

RTX 4060 Ti 8GB请使用新增的本机入口，不要运行旧的training/run.sh。Windows安装WSL2/Ubuntu后，双击根目录Run_Faust_8GB.cmd；Linux执行bash run_faust_8gb.sh。完整步骤见 [本机启动说明](START_LOCAL_8GB.txt)。

入口自动建立独立环境，核对数据，检查真实token长度，运行最长样本的梯度更新、验证与保存试跑，通过后才从原始底模开始正式训练。8GB是否足够以实测为准；显存不足时停止而不截断原文。不会购买服务器或调用收费API。

本机使用Unsloth与Qwen3-8B-Base的现成NF4量化版，只训练LoRA参数，只对原文和EOS计算loss。默认rank16，显式选择lean可改用rank8。正式adapter和诊断日志保存在runs/，不提交Git。

[本机准备验证记录](training/local8gb/verification.json) 明确区分36项已完成的离线测试与未执行的Windows、依赖安装、Qwen分词和GPU集成测试。没有GPU实测记录时不声称本机已经训练成功。

GitHub分支还保留先前的较大显存训练方案和原创评测框架。本机独立下载包只携带冻结数据与8GB入口；它不需要旧方案才能运行。所有训练数据、项目状态与运行环境记录均可追溯。
