# Development Plan

> 用户和 agent 共同维护的当前计划。最新的在最上面。

<!-- 新条目追加到本行下方，保持最新在最上 -->
## 2026-07-21 · FlowDAgger 复现实验准备

- 目标：先安装 Python 3.11 依赖，验证 GPU/MetaWorld/openpi/checkpoint 链路，再按 smoke → short → default 顺序运行实验。
- 已确认：当前仓库只注册 `metaworld_assembly`；环境为 `metaworld`；基础策略为 `pi05`；没有现成 YAML sweep 或多任务入口。
- 运行环境：复用 `/root/miniconda3/envs/dsrl_pi0`；重要输出放到 `/mnt/data/atticux/FlowDAgger/`，每个实验使用独立目录和独立结果 Markdown。
- 当前门槛：先初始化 `flowdagger_pi05/openpi` 子模块并按 README 顺序安装依赖；安装后做 `pip check`、`--help`、JAX GPU、MetaWorld assembly 和 openpi 导入验证。
- 实验策略：smoke 使用 `max_steps=0`、1 个 seed-expert episode、1 个 BC step、1 个 eval episode；仅在 smoke 成功后运行约 100-step short；最后才考虑默认 4000-step 全量。
- 新发现：`bc_batch_size=64` 的 short 在第一次 BC update 触发 JAX `Too small divisible part of the contracting dimension`；改为 `bc_batch_size=16` 后，100 BC steps、step 0/100 eval 和 checkpoint100 均通过，耗时约 5 分 12 秒。
- 启动前结论：默认 `bc_batch_size=256` 未验证；结合 64 的失败和短程实测速度，决定先使用已验证的 batch size 16 稳定化覆盖配置启动 full。

## 2026-07-21 · 切换 W&B 在线记录并准备全量

- W&B 凭证从本机 `.netrc` 可用，当前 entity 为 `1831768457`；`flowdagger` 项目已通过在线同步自动创建。
- `exp-01-smoke`、失败的 `exp-02-short` 和通过的 `exp-02-short-b16` 均已同步到 W&B，后续命令统一使用 `WANDB_MODE=online`、`--wandb_entity 1831768457` 和 `--wandb_project flowdagger`。
- 全量实验将使用独立的 online run 名称和 OSS 本地日志；默认配置仍需决定是否采用已验证的 `bc_batch_size=16` 覆盖。
- `exp-03-default-b16` 已完成：W&B online run 为 finished，4000 BC steps、step 0–4000 的 9 次 25-episode eval、checkpoint500–4000 均通过；最终 success rate 为 0.8，完整结果见 OSS 结果文件。
- 当前实验结论：稳定化 batch size 16 可以完成全量；默认 batch size 256 仍未验证，不应把本次结果标作严格官方默认复现。
