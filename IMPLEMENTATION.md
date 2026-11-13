# 物理嵌入三维预测：结构、训练与运行

当前实现将已训练并冻结的二维翼型模型、官方 PyTornado 几何驱动低保真 SLD 和三维残差网络组合。正式输入输出采用 [FOILPARA.md](FOILPARA.md) 的单算例协议；训练使用原生 SuperWing 数据库，不改变该生产接口。

## 当前训练任务与已有权重

二维已完成 60 轮训练，当前权重为 `weights/2d/latest.pt`，SHA-256 为 `e4ae5a3daa5a1d5b2c6cc025882c6a535e09eee156172f42d3e7ede1a9e295ee`。三维任务使用它的同哈希服务器副本。二维结构和训练记录见 [MODEL_2D.md](MODEL_2D.md)。

当前三维任务由 [build/superwing_200_bs8_config.json](build/superwing_200_bs8_config.json) 指定，run ID 为 `superwing_200_bs8_20261111_190048`。本任务重新初始化三维 residual 和 Adam，从第 0 轮开始训练；二维权重继续冻结，SLD 采用 VLM。

| 项目 | 当前任务设置 |
|---|---|
| 数据 | SuperWing 原生 4,239 个几何、28,856 条工况记录 |
| 固定 train / test | 25,965 / 2,885；精确 9∶1，几何身份无交集 |
| 未纳入 | 固定的 6 条训练侧余数记录；源数组保持不变 |
| 训练总轮数 | 200 |
| train / test batch size | 8 / 4 |
| 优化更新 | 每轮 3,246 次，最后训练批为 5 个样本 |
| 优化器与精度 | Adam，仅优化 residual；FP32，AMP=none |
| 学习率 | 基准最大值 `1e-4`；前 60 轮线性预热 `1e-5→1e-4`，之后每轮乘 `0.95^(1/3)` |
| test | 每个完整 epoch 后评估全部 2,885 条，不更新参数、归一化范围或学习率 |
| 检查点选择 | 最后完成一轮的权重，不根据 test 回滚最佳轮 |
| 独立保留与下载 | 第 10、20、…、200 轮各一个检查点，下载到本机后校验 SHA-256 和文件大小 |
| 曲线与日志 | 每轮更新 train/test loss 和 test 百分比误差曲线 |

固定 9∶1 拆分按完整几何分组，而非随机打散同一机翼不同工况；seed=42，精确比例所需余数通过固定 seed=43 选择。当前不是原官方训练列表加内部验证的旧划分。所有归一化范围只从 train 拟合；test 不参与拟合、优化或检查点选择。

任务进度会变化，应查询 `status.json` 或监督状态，不能把此文档中的目标 200 轮当作已完成记录。

## 核心代码职责

| 文件 | 职责 |
|---|---|
| [main.py](main.py) | 模型定义、物理迁移、原生数据库适配、检查点加载和通用 CLI |
| [foilpara.py](foilpara.py) | 三目录单算例输入、外部 SLD 前向、网络结果导出 |
| [low_fidelity_sld.py](low_fidelity_sld.py) | 官方 PyTornado 原生 VLM 及几何/展向映射 |
| [prepare_superwing_3d_training.py](prepare_superwing_3d_training.py) | 几何分组 9∶1 拆分、真实 Re、VLM 与冻结二维先验缓存 |
| [train_3d_train_test.py](train_3d_train_test.py) | 当前三维训练、逐轮 test、曲线、可恢复权重与十轮快照 |
| [build/run_superwing_200_server.py](build/run_superwing_200_server.py) | 项目内远端派发、断线重连及校验后下载检查点 |
| [build/launch_superwing_bs8.py](build/launch_superwing_bs8.py) | 从 PyCharm subprocess 启动当前任务监督器，防止重复派发 |
| [build/check_superwing_bs8_progress.py](build/check_superwing_bs8_progress.py) | 只读查询当前 batch size 8 任务的远端进度 |
| `flowvae/base_model/conv.py`、`flowvae/vae.py` | 二维条件卷积 U-Net 底层实现 |
| `flowvae/base_model/resnet.py` | 三维表面残差网络的 ResNet U-Net 编解码器 |

原有 SLD FNN、`main.py train-sld/train-3d` 和桥接功能继续保留。当前训练不调用 SLD FNN，不使用 CFD 场生成 SLD；保留类和参数并不表示这些参数在当前任务中受过训练。

## 几何、后掠理论与二维先验

几何网格为单元中心 `3×128×256` 和顶点 `3×129×257`；坐标为 `[x,y高度,z展向]`。`geometry_features` 从真实顶点构造各截面局部弦坐标，沿上下表面分别按物理 x/c 插值到 SuperFoil 401 点位置。二维输出再按 x/c 插值回三维单元中心，不按数组下标直接缩放网格。

采用论文式 (1)～(4) 的全局后掠变换。设 `cΛ=cos(后掠角)`，每个截面的转换为：

| 量 | 转换 |
|---|---|
| 二维相对形状 | `(y/c)_2D = (y/c)_section / cΛ` |
| 二维 Mach | `Ma_2D = Ma_wing × cΛ` |
| 二维截面 CL | `CL_2D = CL_SLD / cΛ²` |
| 二维实际 Re | `Re_2D = Re_wing × local_chord × cΛ`；网络再以百万为单位输入 |
| 二维 Cp/Cf 到三维先验 | 将二维原单位预测乘 `cΛ²` |

针对 SuperWing 折点机翼，后掠角取暴露翼面首末截面的四分之一弦点连线，保持单一全局余弦修正；复杂三维效应由 residual 学习。SuperWing 长度采用参考弦长 1 的尺度。真实三维 Re 从原日志或明确提供的数据读取，不用二维默认 Re 覆盖。

周向上下表面的摩擦方向不同，二维 `1000×Cf` 转为三维 `150×Cf_tau` 时处理下表面符号；尾缘封口单元使用上下尾缘值的平均。展向摩擦先验为零，由三维 residual 预测。这些转换发生在二维输出逆归一化之后。

## 当前低保真 SLD

SLD 通过 `low_fidelity_sld.estimate_spanwise_lift` 调用官方 PyTornado 0.5.0 原生求解器。输入为真实几何和攻角，Mach 接受并记录但未施加压缩性修正；不接收 CFD 压力/摩擦场。默认每半翼 32 个展向面板、4 个弦向面板，镜像后共 256 个面板；按物理展向位置映射为 128 个单元中心的无量纲截面 CL，输出位于后掠变换之前。

这是不可压缩、无黏势流估计，不能解析跨声速激波和分离。完整几何处理、求解器调用、安装记录和数值边界见 [PYTORNADO.md](PYTORNADO.md)。求解失败时直接报错，不自动退回 CFD 积分或常数分布。

当前检查点明确记录 `sld_source=pytornado_vlm`、`sld_network_used=False`、`sld_uses_cfd=False`、`sld_training_steps=0`。完整模型仍包含冻结的 `lift` 模块，以保持结构/检查点兼容；当前 forward 使用外部 SLD，不调用该未训练模块。

### 保留的作者三种 SLD 方案

作者示例 [examples/wing/run_kfold.py](examples/wing/run_kfold.py) 的 `ref_type=1/2/3` 分别保留以下方案，FNN 构造位于 [flowvae/app/wing/models.py](flowvae/app/wing/models.py) 的 `triinput_simplemodel1`：

| 方案 | 作者示例选择 | 计算含义 | 当前任务是否使用 |
|---|---|---|---|
| 纯 VLM | `ref_type=1` | 示例读取预计算 VLM 分布；本项目现用官方 PyTornado 生成该低保真量 | **是** |
| FNN 直接预测 | `ref_type=2` | FNN 从几何/工况直接预测截面 CL | 否，保留算法与入口 |
| FNN 残差加 VLM | `ref_type=3` | FNN 学习 `CFD_CL−VLM_CL`，推理时再加回 VLM | 否，保留算法与入口 |

作者示例还支持 CFD 真值参考分布等对照，不等同于独立低保真估计。两种 FNN 路径的代码保留不表示当前已经有与新 SuperWing/128 站位接口匹配的已训练权重。当前正式三维任务只有纯 VLM 来源，不能将未训练的 FNN 参数用于替代它。

## 三维 residual 的结构与训练前向

这是对三维机翼表面参数网格使用二维卷积的网络，不是体积 Conv3d。

- 输入 5 个通道：3 个几何坐标与 2 个经过后掠变换的二维 Cp/Cf 先验，形状 `(B,5,128,256)`。
- 编码器六组通道为 `32、64、64、128、128、256`。只在周向下采样，展向 128 保持不变，周向由 256 压缩至 4。
- 全局 Mach、AoA 两项归一化后扩展到压缩特征图，与 256 通道拼接为 258 通道。
- 三个独立单通道 ResNet U-Net 解码器，通道为 `128、128、64、64、32、32`，恢复 `(B,3,128,256)` 残差。
- 最终输出为残差加 `[prior_Cp,prior_Cf_tau,0]`，三个通道依次为 `Cp、150×Cf_tau、300×Cf_z`。

输入的 5 个通道、工况和目标分别使用 train 拟合的 min/max buffer。解码器的残差恢复只乘目标范围，不加目标最小值，避免把偏置重复加到先验上。损失是最终场在固定 train 归一化空间的 MSE。

准备阶段对全部选中样本计算 VLM 和冻结二维先验，训练阶段 `cached_forward` 使用同一先验结果，避免每轮重复计算 128 个二维截面。只更新三维 residual；二维与未使用 lift 的参数和 buffer 保持不变，每轮校验冻结状态与归一化状态。

## 原生数据与准备缓存

```text
/home/cvlab/Projects/floGen/
  datasets/SuperWing/                       # 原生55列配置、12列index、几何、场与Re日志
  weights/2d/superfoil_epoch060_e4ae5a3daa5a.pt
  .agent_runs/superwing_200_vlm_20261110_151059/cache/
    geometry.npy                           # (G,3,128,256)，float32
    prior.npy                              # (N,2,128,256)，float32
    conditions.npy                         # (N,2)，Mach、AoA
    geometry_indices.npy                   # (N,)，缓存几何局部下标
    groups.npy                             # (N,)，原几何身份
    sld.npy                                # (N,128)，VLM截面CL
    reynolds.npy                           # (N,)，实际Re
    split/{train_rows,test_rows,excluded_rows}.npy
    split/manifest.json
    manifest.json                          # 源文件、缓存和二维权重SHA
  .agent_runs/superwing_200_bs8_20261111_190048/
    pipeline_config.json
    runtime/                               # HOME、临时文件及各类任务缓存
    training/                              # 当前任务独立输出
```

缓存 `N` 仍按原 28,856 条记录寻址，训练实际使用拆分中的 28,850 条；6 条 excluded 不用于训练/测试。CFD 标签路径记在缓存 manifest 中，当前服务器传输副本为原始目标按训练相同方式转换的 float32；本机原始 float64 数据未修改。缓存和二维权重以内容 SHA 绑定，训练开始、恢复和最终验证均检查来源一致性。

## 运行、同步与检查点

当前配置在 PyCharm 内通过以下本机入口派发；已有 launch marker 时不会重复启动一个新任务：

```powershell
& 'C:/Users/fff/miniconda3/envs/flogen/python.exe' -B build/launch_superwing_bs8.py
```

只读查询当前任务：

```powershell
& 'C:/Users/fff/miniconda3/envs/flogen/python.exe' -B build/check_superwing_bs8_progress.py
```

服务器解释器为 `/home/cvlab/miniconda3/envs/flogen/bin/python`。任务文件、HOME、TMP、Matplotlib、CUDA、Torch 等任务缓存均放在 `Projects/floGen` 内。SSH 使用既有项目连接助手和固定主机密钥；凭据不写入文档或日志。

本机状态为 [weights/3d/server_200_bs8_job.json](weights/3d/server_200_bs8_job.json)，输出目录为 [weights/3d/superwing_200_bs8_20261111_190048](weights/3d/superwing_200_bs8_20261111_190048)。服务器 `latest.pt` 每轮原子更新，包含完整模型、归一化、Adam、随机数状态和历史；`epoch_010.pt` 等十轮快照发布后由监督器下载到本机 `checkpoints/` 并核对 SHA。逐轮日志和图像同步到该本机目录。

训练器支持 `--resume` 恢复同一输出目录，要求 batch、数据 SHA、拆分、二维来源和其他配置一致。当前 batch size 8 新任务未传 resume，旧 batch size 2 权重仅归档。需要另开新训练时，应使用独立的 run/config/output；不能把当前已经派发的配置误当新任务反复覆盖。

完成的判定包括：服务器 200 轮结束、最终检查点重新加载 test 结果一致、`verification.json=passed`，以及本机 20 个十轮检查点全部下载通过校验。

## 指标与其他入口的区别

`history.json` 每轮记录 train MSE、完整 test MSE、训练/测试耗时，以及每个通道的平均绝对误差、RMSE 和百分比统计。主要百分比为每个机翼通道上的 `100×mean(abs(pred−true))/(max(true)−min(true))`，再汇总均值、p95、最差值和极差。相对 L1 以 `sum(abs(true))` 为分母；Cf 物理单位统计先除以 150/300。零分母百分比记为未定义并单列数量，不用任意 epsilon 伪造百分比。不能用 `100%−NMAE` 表示准确率。

`main.py pretrain-2d/train-sld/train-3d` 是原保留入口，其默认设置、官方列表/验证划分和最佳验证权重策略不同：二维默认 300 轮，FNN-SLD/三维默认 900 轮，通用三维默认 batch 2。当前专用训练器默认 200 轮，但 parser 的 batch 默认仍为 2；本次通过任务配置显式传入 train 8、test 4。文档中的当前任务值不是修改了这些通用 CLI 默认值。

论文使用二维 321 点、三维 101×321 网格；当前使用原数据库 401 点和 128×256 单元。论文三维训练设置含 batch 8、900 轮和交叉验证，当前用户要求 200 轮和固定 9∶1 test。论文附录文字与作者源码的解码器表述存在差别；当前三个单通道分支与作者源码形式相近，不声称所有层级和训练实验完全复现论文。

当前 VLM 三维权重必须通过 `predict-foilpara` 或显式提供外部 SLD 的同等前向调用推理；`main.py predict` 和 [bridge.py](bridge.py) 的旧 FNN-SLD 协议不能直接使用它。原生 SuperWing 库格式、三目录单 case 格式和桥接 geometry manifest 格式是不同接口，不能互换目录名称。
