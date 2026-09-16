本目录不保存模型权重。2026-09-16已完成faust-local8gb-0.2.0的lean/rank-8本机QLoRA训练；可审计记录见faust-local8gb-r8-20260916.json。

记录包含run_manifest中的底模ID与HF提交号、数据指纹、训练所用Git提交、最佳检查点、指标、adapter哈希和张量完整性结论。独立生成比较尚未执行，记录中明确标为pending，不以训练loss代替生成质量结论。

实际权重保存在runs/local8gb-r8/train/adapter/，由.gitignore排除。需要远端备份时由用户另行决定托管位置与可见性，不自动公开上传。
