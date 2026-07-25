# 实验报告：FlowDAgger π0.5 · MetaWorld Assembly

> 本报告归档本机在 MetaWorld ``assembly-v3`` 上完成的 FlowDAgger 验证和全量运行。
> 结论以 W&B 记录和 `/mnt/data/atticux/FlowDAgger/runs/` 的控制台日志共同判定；
> W&B 的 `finished` 状态不覆盖本机 traceback。

## 实验范围

- **任务：** MetaWorld ``assembly-v3``，提示词为“Pick up a nut and place it onto a peg”。
- **基础策略：** ``mmurray-ms/pi05-metaworld``；seed 42；使用 scripted expert intervention。
- **关键变量：** ``bc_batch_size``。本机验证表明 64 在首次 BC update 失败，16 可完成
  short 与 4,000-step full。

## 结果

| 运行 | 配置与状态 | 结果 | 可用结论 |
| --- | --- | --- | --- |
| ``exp-01-smoke`` | batch 16；0 BC step；完成 | 1 回合评测 success 0/1 | 仅证明导入、GPU、环境和 W&B 链路可运行。 |
| ``exp-02-short`` | batch 64；目标 100 steps；失败 | W&B 在首次 update 前记录的 5 回合 eval 为 4/5；本机日志在首次 BC update 抛出 ``JaxRuntimeError: Too small divisible part of the contracting dimension`` | **失败实验**；不得把预训练评测或 W&B `finished` 当作 100-step 结果。 |
| ``exp-02-short-b16`` | batch 16；100 steps；完成 | 5 回合 eval success 4/5；BC loss 0.15013 | batch 16 通过 100-step 训练、评测和 checkpoint 链路。 |
| ``exp-03-default-b16`` | batch 16；4,000 steps；完成 | 5 回合 eval success 4/5；BC loss 0.01125；4,461 env steps | 已验证的稳定化 full 配置。 |

full run 的线上指标显示 intervention step rate 为 90.6%、episode rate 为 80.0%。
它描述该 single-seed 训练过程中的专家接管比例，不是跨任务的泛化指标。

## 资源与时长

资源数据来自 W&B ``system`` 流。显存以 ``memoryAllocatedBytes`` 换算为 GiB；
GPU·小时按 ``实际 runtime × W&B 中显存分配超过 1 GiB 的 GPU 数`` 计算；
峰值显存和利用率为采样值。

| 运行 | W&B runtime | W&B 监测 GPU | 峰值显存 / 卡 | 平均 GPU 利用率 | GPU·小时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoke | 10 分 36 秒 | 2 | 72.5 / 72.2 GiB | 4.5 / 3.4% | 0.35 |
| short batch 64（失败） | 1 分 50 秒 | 2 | 72.5 / 72.2 GiB | 64.6 / 41.7% | 0.06 |
| short batch 16 | 4 分 56 秒 | 2 | 72.5 / 72.2 GiB | 41.8 / 41.5% | 0.16 |
| full batch 16 | 53 分 33 秒 | 2 | 72.5 / 72.2 GiB | 42.8 / 42.8% | 1.79 |

full run 的进程峰值 RSS 约 13.2 GiB、线程数峰值 367。W&B 监测到两张 GPU 同时有
大额显存分配，即使启动参数写为 ``cuda:0``；迁移时应按两卡而不是单卡预留。

## 换机复现

- 使用 2 张每卡至少 80 GiB 显存的 GPU。此次峰值接近 72.5 GiB，切勿按单卡或
  16 GiB 级别的显存规格迁移。
- 先运行 batch 16 的 smoke 与 100-step short，再运行 4,000-step full；full 的
  实测预算约 54 分钟、1.8 GPU·小时，建议预留 1.5 小时以覆盖首次下载、JIT 编译
  与环境抖动。
- 不要直接复用 ``bc_batch_size=64`` 或代码默认的 256；它们在这台机器上都没有
  获得完整训练验证。使用 [cmd.md](../cmd.md) 的 batch 16 命令，并为每次复现创建
  新的 W&B run 名称。

## 证据与边界

- W&B 项目：[1831768457/flowdagger](https://wandb.ai/1831768457/flowdagger)。
- 控制台日志：`/mnt/data/atticux/FlowDAgger/runs/exp-*/console.log`。当前 OSS
  挂载为只读，日志可以复核但不能在原位置补写。
- 所有完成结果仅覆盖 assembly-v3、seed 42 和 1 或 5 个评测 episode，不能作为
  MetaWorld 总体或跨 seed 的比较结论。
