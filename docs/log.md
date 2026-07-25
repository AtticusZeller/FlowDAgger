# Development Log

> 已验证完成的任务记录。最新的在最上面。

<!-- 每个任务通过全部必要验证后，在本行下方追加一条 -->

## 2026-07-25 · 归档 MetaWorld Assembly 实验结论与资源账本

- 新增 [`docs/experiment-report.md`](experiment-report.md)，以 W&B 与本机控制台日志
  双重核对 smoke、short 和 4,000-step full；明确 batch 64 的 short 虽在 W&B 标为
  ``finished``，但首次 BC update 已在本机以 JAX contracting-dimension 错误失败。
- batch 16 的 4,000-step full 已完成，最终 5-episode eval 为 4/5；W&B runtime
  为 53 分 33 秒，监测两张 GPU 的峰值显存分别为 72.5 / 72.2 GiB，近似 1.79
  GPU·小时。
- 换机复现应按 2 张每卡至少 80 GiB 显存 GPU 预留，并使用 batch 16 的 smoke →
  100-step short → full 顺序；没有将单 seed、5-episode 结果外推为 MetaWorld 总体性能。

## 2026-07-21 · 依赖安装与最小复现实验

- 状态：已验证当前阶段。
- 依赖：初始化 `openpi` 子模块并在 Python 3.11 `dsrl_pi0` 环境完成安装；JAX 识别两张 NVIDIA H20，MetaWorld `assembly-v3`、MuJoCo、openpi 和 pi0.5 checkpoint 均可导入/加载。
- 实验：`exp-01-smoke` 通过；`exp-02-short` 使用 `bc_batch_size=64` 在第一次 BC update 失败；改为 `bc_batch_size=16` 的 `exp-02-short-b16` 通过 100 BC steps、step 0/100 eval 和 checkpoint100。
- 文档：根目录 `cmd.md` 已包含依赖安装、tmux 后台命令、输出路径、验收标准和全量实验的待确认风险；每次实验结果单独记录在 `/mnt/data/atticux/FlowDAgger/results/`。
- 待处理：`pip check` 的元数据冲突、CUDA `ptxas` 警告以及默认 `bc_batch_size=256` 尚未消除/验证；默认 4000-step 全量实验暂不启动。

## 2026-07-21 · W&B 在线记录

- W&B entity/project：`1831768457/flowdagger`。
- 前面三个本地 offline run 已同步到 W&B；远端项目当前包含 smoke、失败的 batch64 short 和通过的 batch16 short 三个 run。
- `cmd.md` 已统一改为 `WANDB_MODE=online`，并要求显式传入 entity、project 和非空 prefix。

## 2026-07-21 · W&B 全量实验完成

- `exp-03-default-b16` 在 NVIDIA H20 上运行约 53 分 48 秒，4000 BC steps，返回码 `0`。
- W&B run 状态为 `finished`，远端记录了 step 0/500/1000/1500/2000/2500/3000/3500/4000 的 25-episode eval；最终 success rate 为 `0.8`。
- 本地生成 checkpoint500–4000，未出现 JAX contracting-dimension 错误或未捕获异常。
- 结果文件：`/mnt/data/atticux/FlowDAgger/results/exp-03-default-b16.md`。
