# flowdagger_pi05 模块

该目录是 pi0.5 后端实验入口，负责加载 openpi 的 pi0.5 MetaWorld policy、运行 JAX steering actor、收集 scripted-expert intervention、反演噪声目标、执行 BC 更新和评估。

- 入口：`flowdagger_pi05/train_flowdagger.py`
- 训练循环：`flowdagger_pi05/dagger_loop.py`
- 训练/评估辅助：`flowdagger_pi05/train_utils.py`
- 依赖：`flowdagger_pi05/openpi` git submodule 和 `flowdagger_pi05/requirements.txt`
- 当前可复现实验：论文 MetaWorld-12 任务，每个任务独立训练 steering policy，基础策略
  均为 `pi05`
- 命令与后台启动方式：见根目录 [`cmd.md`](../cmd.md)

`train_flowdagger.py` 仍是单任务原生入口；父工作区根据 suite manifest 生成 12 tasks ×
3 seeds 的 YAML 并负责汇总。成功运行写出 `flowdagger_result.json`。策略观察固定为
`corner3`，人类评估视频由独立 `corner` renderer 输出 640×480、30 FPS，避免录像复用
128×128 steering 输入。

CLI 的 ablation 参数以 `train_flowdagger.py --help` 为准。
