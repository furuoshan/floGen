# 二维神经网络结构与训练记录

当前二维模型是 `main.AirfoilPredictor`，由原作者的一维卷积 U-Net 模块构成。加载和训练流程见[二维神经网络使用说明](二维神经网络使用说明.md)；本文件集中说明结构、真实数据划分和已有权重，避免把历史单轮实验当作当前结果。

## 输入、输出和结构

| 项目 | 当前实现 |
|---|---|
| 几何 | `(B,1,401)`，SuperFoil 原周向顺序的 `y/c` |
| 工况 | `(B,3)`，依次为 `Mach、CL、Re/1e6` |
| 输出 | `(B,2,401)`，依次为 `Cp、1000×Cf`，已经逆归一化 |
| 网络性质 | 确定性条件一维卷积 U-Net，无随机潜变量和 KL 损失 |
| 可训练参数 | 940,424 |
| 激活 | 隐藏层 LeakyReLU，negative_slope=0.01；最终输出层为线性卷积 |
| BatchNorm / Dropout | 当前二维构造未启用 |
| 归一化 | 每个变量一组训练集 min/max，范围作为 buffer 保存在权重中；推理不重新拟合 |

设 B 为 batch 大小，下表省略 B。编码器各组顺序为卷积、LeakyReLU、平均池化；卷积核 `k=3`、步长 `s=2`、填充 `p=1`，池化 `k=3,s=2,p=0`。

| 阶段 | 主要操作 | 输出形状 |
|---|---|---|
| 编码器 1 | Conv1d 1→16，然后池化 | `16×100` |
| 编码器 2 | Conv1d 16→32，然后池化 | `32×24` |
| 编码器 3 | Conv1d 32→64，然后池化 | `64×5` |
| 几何潜变量 | Flatten 320，Linear 320→32 | `32` |
| 工况拼接 | 归一化工况 3 维与潜变量 32 维拼接 | `35` |
| 全连接 | 35→64→128→2560，reshape | `512×5` |
| 解码器 1 | 拼接编码 `64×5`，插值到 24，Conv1d 576→256 | `256×24` |
| 解码器 2 | 拼接编码 `32×24`，插值到 100，Conv1d 288→128 | `128×100` |
| 解码器 3 | 拼接编码 `16×100`，插值到 401，Conv1d 144→64 | `64×401` |
| 输出 | 拼接原归一化几何，Conv1d 65→2，逆归一化 | `2×401` |

解码卷积采用 `k=3,s=1,p=1`，上采样为线性插值，`align_corners=False`。参数分组为编码卷积 7,840、几何投影 10,272、解码全连接 340,864、解码卷积 581,448，总计 940,424。底层构造见 `flowvae/base_model/conv.py` 和 `flowvae/vae.py`。

## 当前数据与固定 9∶1 划分

训练代码为 [train_2d_train_test.py](train_2d_train_test.py)，它复用 [train_2d_once.py](train_2d_once.py) 的 `JointSuperFoil` 数据组织，再通过 [superfoil_train_test_split.py](superfoil_train_test_split.py) 做独立的固定划分。旧脚本仍是依赖，不应因为其单轮入口已被替代就删除。

| 子集 | 源记录 | train | test | 未纳入 |
|---|---:|---:|---:|---:|
| VT-BUF | 25,607 | 23,079 | 2,525 | 3 |
| VT-DD | 17,386 | 15,627 | 1,758 | 1 |
| VT-GU | 16,031 | 14,412 | 1,619 | 0 |
| 合计 | **59,024** | **53,118** | **5,902** | **4** |

VT 原数组为 `data.npy=(N,3,401)` 和 `index.npy=(N,13)`，场通道依次为 `y/c、Cp、1000×Cf`。BUF/GU 的 Mach 和 CL 分别使用 index 零基列 5、9，DD 使用零基列 3、6；DD 第 9 列为实际攻角，不能作为 CL。FT 本地缺少表面场 `data.npy`，未纳入训练。

当前划分使用三个 VT 子集的完整选定记录，不沿用旧单轮训练器的官方测试家族保护划分。种子为 42，按原生元数据中 20 个 CST 系数相同的基准翼型家族分组，train/test 家族不重叠。4 条固定余数记录未纳入，以得到精确 9∶1 样本比例；源数据库未删改。划分数组及清单保存在 `weights/2d/split/`。

家族判定需要 [build/foil_meta_for_condition_audit.npz](build/foil_meta_for_condition_audit.npz)。虽然文件名含 audit，它是重训依赖，默认路径由 `superfoil_train_test_split.DEFAULT_META` 指定。

VT index 没有逐样本 Re 列。本次训练输入统一使用项目配置 `Re=2e7`，在网络工况中为 20；该常量归一化后为零。BUF/DD 的公开来源已支持固定 Re；GU 的本地发布数据缺少 Re 列，因此本权重记录的是统一项目设置，不应将其表述为已经独立核实了 GU 每条流场的物理 Re。保留 Re 输入不代表固定 Re 数据能教会网络对 Re 变化的响应。

## 已完成的 60 轮权重

| 项目 | 已保存记录 |
|---|---|
| 权重 | [weights/2d/latest.pt](weights/2d/latest.pt) |
| run_id | `3b7e346d8aa54e50902e531f92483432` |
| 训练轮数 | 60，最后一轮权重 |
| 优化器 / batch | Adam；train 16、test 128；保留最后的小批次 |
| 优化更新数 | 199,200 |
| 学习率 | 前 20 轮由 `1e-5` 预热至 `1e-4`，之后每轮乘 0.95 |
| 最终 train / test 归一化 MSE | `3.3592422647e-5` / `3.8875771340e-5` |
| test 平均归一化绝对误差 | Cp 0.493917%，Cf 0.655052% |
| test 误差极差 | Cp 5.484339、Cf 6.799360 个百分点 |
| 权重 SHA-256 | `e4ae5a3daa5a1d5b2c6cc025882c6a535e09eee156172f42d3e7ede1a9e295ee` |
| 读回验证 | `verification.json` 为 passed；源文件未变、家族无交集、test 未用于参数更新 |

百分比误差先对单个翼型/通道计算 `100 × mean(abs(pred−true)) / (max(true)−min(true))`，再汇总 test；它不是可用 `100%−误差` 转换的分类准确率。误差极差是测试样本误差百分比的最大值减最小值，单位为百分点。

训练记录与图像分别见 [training_report.json](weights/2d/training_report.json)、[status.json](weights/2d/status.json)、[verification.json](weights/2d/verification.json)、[loss 曲线](weights/2d/loss_vs_epoch.png)和[test 误差曲线](weights/2d/test_error_percent_vs_epoch.png)。检查点同时保存模型参数、归一化 buffer、ModelSpec、Adam 状态和逐轮历史；当前训练器支持受校验的 `--resume`。

## 与论文设置的关系

论文附录 A1 使用同类条件卷积 U-Net，编码通道 16/32/64、潜变量 32、解码通道 512/256/128/64。当前采用现有 SuperFoil 的 401 点而非论文 321 点；对应编码长度为 100/24/5。论文二维训练为 300 轮、batch 16，并包含多次初始化实验；当前用户指定 60 轮、单次初始化、固定 9∶1 train/test，保留最后权重。因此结构方法相近，训练过程和数据分布不等同于论文全部实验。
