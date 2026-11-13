# 当前项目目录与保留原因

本文按当前代码依赖说明目录职责。主导航为 [readme.md](../readme.md)，当前任务说明为 [多智能体系统框架.md](../多智能体系统框架.md)。核心二维/三维训练代码、权重、生产数据接口和 SLD 算法保持原实现；目录整理不改变它们的路径或行为。

## 根目录入口

| 文件 | 保留用途 |
|---|---|
| `main.py` | 二维模型、三维 residual、后掠理论、原生数据库读取、检查点与 CLI |
| `foilpara.py` | `foilgeo/foiloc/foilsld` 单算例读取、外部 SLD 推理及正式结果导出 |
| `low_fidelity_sld.py` | 官方 PyTornado 原生 VLM 适配、真实几何弯度面和 128 站位映射 |
| `superwing_reader.py`、`superwing_io.py` | SuperWing 单 case 读取、来源追踪和输入/CFD/可选网络结果交付 |
| `train_2d_train_test.py` | 当前 60 轮二维训练、固定拆分、每轮 test 和最后权重 |
| `train_2d_once.py` | 当前二维训练器依赖其 `JointSuperFoil`；不能仅因文件名像旧单轮任务而删除 |
| `superfoil_train_test_split.py`、`superfoil_test_metrics.py` | 固定 CST 家族隔离 9∶1 拆分和真实 test 指标 |
| `prepare_superwing_3d_training.py` | 当前三维几何隔离拆分、真实 Re、VLM/冻结二维先验缓存 |
| `train_3d_train_test.py` | 三维独立训练、完整 test、曲线、可恢复 latest 和十轮检查点 |
| `bridge.py` | 受保护的独立兼容接口；当前 VLM 权重不满足其已训练 FNN-SLD 前序条件 |
| `readforagent` | 原始项目说明，原文不修改 |
| 论文译文与 `LICENSE` | 方法来源、作者代码许可 |

根目录各 Markdown 按职责分别说明二维结构/使用、三维实现、单算例契约、参考库交付、PyTornado 与 bridge。当前文档不将不同目录协议混为同一个接口，也不把未使用的算法参数称为受过训练的权重。

## 保留目录

| 目录 | 用途与保留范围 |
|---|---|
| `flowvae/` | 原作者模型和算法库；保留当前入口及作者三种 SLD 方案的导入依赖 |
| `examples/wing/` | 作者机翼方法来源，包含三种 SLD 选择及相关模型/训练示例 |
| `tests/` | 当前模型、训练器、原始数据适配、生产接口、SLD 和任务监督相关回归 |
| `weights/2d/` | 当前二维权重、固定 split、训练与 test 历史、验证结果及图像 |
| `weights/3d/` | 当前三维监督状态、逐轮日志/图像、下载检查点及已有权重归档 |
| `checkpoints/` | 已有检查点文件；不由文档整理重新训练或替换 |
| `build/` | 当前训练控制、不可移动的元数据、SSH 主机校验及 SLD 构建资料；详见下节 |
| `.deps/` | 当前连接等依赖的项目内副本；不把正在使用的依赖当成临时输出清理 |
| `docs/` | 本目录结构说明；原始外部通用说明与当前本机契约需区分 |
| `maintenance/cleanup_20261112/` | 本次清理清单、保留文件校验与回归测试记录；不参与训练或推理 |

`D:/SuperFoil`、`D:/SuperWing` 和 `D:/data` 位于仓库外，本次项目整理不修改源数据库或正式输入/输出的 `format.md` 契约。`.git`、IDE 配置及凭据不是可通过普通代码冗余分析删除的业务文件。

## 为什么 build 不能整个删除

`build` 兼有运行控制与准备材料，并不等同于可以随时重新生成的编译缓存。当前至少有以下直接依赖：

| 路径或材料 | 原因 |
|---|---|
| `build/foil_meta_for_condition_audit.npz` | 二维固定拆分默认读取它，将原始 Running_ID 关联到 20 系数 CST 家族；缺失会影响重训 |
| `build/superwing_200_bs8_config.json` | 当前三维 batch size 8／200 轮任务的实际配置 |
| `build/launch_superwing_bs8.py` | 从 PyCharm subprocess 启动监督器，并使用任务标记防止重复派发 |
| `build/run_superwing_200_server.py` | 正在使用的 SSH 任务监督、断线重连、日志同步和检查点校验下载 |
| `build/check_superwing_server.py`、`build/server_known_hosts` | 项目内 SSH 连接助手和固定主机密钥校验；文档不记录密码 |
| `build/check_superwing_bs8_progress.py` | 当前任务的只读进度检查 |
| `build/run_superfoil_60_background.py` | 当前二维 60 轮训练的本机后台启动入口 |
| `build/stage_superwing_200.py` | 原生 SuperWing 向项目内服务器目录传输；压缩缓存可从原始库重新生成 |
| `build/setup_superwing_flogen_runtime.py` | 服务器 Conda `flogen` 的项目内运行组件准备，保留官方 VLM 内核构建方式 |
| 任务 launch marker、config 与运行控制记录 | 绑定同一任务身份，避免重复启动；不能只看文件较旧就删除 |
| PyTornado wheel、构建脚本、必要源材料及兼容清单 | 保留官方原生求解器构建与现代 Python/NumPy 兼容来源，不能以另一套算法替换 |

完整 VLM 构建材料列表见 [PYTORNADO.md](../PYTORNADO.md)，当前训练控制和文件路径见 [IMPLEMENTATION.md](../IMPLEMENTATION.md)。已完成的一次性探测、失败试算和可再生编译缓存应按依赖逐项清理，而不是整目录删除。

服务器当前任务会复用原命名缓存 `.agent_runs/superwing_200_vlm_20261110_151059/cache`。虽然路径含较早的任务标识，当前配置和 SHA 绑定仍直接依赖它；不能为统一命名而移动或删除。新任务的运行时、训练输出则在 `.agent_runs/superwing_200_bs8_20261111_190048/`。这些远端目录全部位于 `/home/cvlab/Projects/floGen`。

## 作者三种 SLD 与较大的模型依赖

[examples/wing/run_kfold.py](../examples/wing/run_kfold.py) 保留作者三种来源：`ref_type=1` 的 VLM，`ref_type=2` 的 FNN 直接预测，以及 `ref_type=3` 的 FNN 预测 `CFD_CL−VLM_CL` 后加回 VLM。FNN 构造为 [flowvae/app/wing/models.py](../flowvae/app/wing/models.py) 中的 `triinput_simplemodel1`。当前正式任务使用 `low_fidelity_sld.py` 的纯 VLM，两条 FNN 方案的保留不意味着当前已有对应训练权重。

作者的 `flowvae.app.wing.models` 在模块顶层同时导入卷积、ResNet、MLP、RNN、Transformer 和 PDE Transformer 等构造。因此即使当前正式 residual 不采用 Transformer，完整保留作者方案的导入链仍可能需要 `base_model/rnn.py`、`base_model/trans.py`、`base_model/pdet/`、`vae.py` 以及相关通用算子。`run_kfold.py` 还导入 `flowvae.app.multipoint.models`、`ml_operator`、数据集和机翼工具。

这类文件是否被当前单次训练 forward 调用，与它是否是受保护作者算法的导入依赖，是两个不同判断。整理时保留其实际导入闭包，不能仅按类名中有 Transformer 或当前权重未使用某个分支就删除。作者示例采用较早的数据/网格与外部依赖，保留方法来源也不等同于已经把示例自动改造成当前 128×256 训练任务。

## 接口与运行状态

正式生产输入始终是 `D:/data/foilpara/{foilgeo,foiloc,foilsld}`，网络输出为 `D:/data/foilperf/network`；`bridge` 的 geometry manifest 是另一套受保护的兼容协议。字段、数组形状、物理单位、倍率和目录冲突行为分别以 [FOILPARA.md](../FOILPARA.md)、[SUPERWING_IO.md](../SUPERWING_IO.md)、[BRIDGE.md](../BRIDGE.md) 为准。

训练目标、已完成轮数和当前 epoch 必须区分。配置是 200 轮，不代表训练已完成；监督器在服务器完成、最终读回验证通过且十轮权重全部下载校验后才标记 complete。保留当前记录和运行依赖，才能在清理后继续检查与恢复同一个任务。
