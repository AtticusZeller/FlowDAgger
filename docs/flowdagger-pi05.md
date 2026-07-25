# flowdagger_pi05 模块

该目录是 pi0.5 后端实验入口，负责加载 openpi 的 pi0.5 MetaWorld policy、运行 JAX steering actor、收集 scripted-expert intervention、反演噪声目标、执行 BC 更新和评估。

- 入口：`flowdagger_pi05/train_flowdagger.py`
- 训练循环：`flowdagger_pi05/dagger_loop.py`
- 训练/评估辅助：`flowdagger_pi05/train_utils.py`
- 依赖：`flowdagger_pi05/openpi` git submodule 和 `flowdagger_pi05/requirements.txt`
- 当前可复现实验：MetaWorld `assembly-v3`，任务键 `metaworld_assembly`，基础策略 `pi05`
- 命令与后台启动方式：见根目录 [`cmd.md`](../cmd.md)

当前代码没有独立 YAML sweep 或多任务训练入口；CLI 的 ablation 参数以 `train_flowdagger.py --help` 为准。
