# 物理嵌入式三维预测入口：结构分析、论文映射与运行说明

## 2026-11-09：按 readforagent 补齐输入与只前向测试

当前指定入口为 `python main.py predict-foilpara`，默认从 `D:/data/foilpara`
下的 `foiloc / foilgeo / foilsld` 读取数据，输出到
`D:/data/foilperf/network`。详细文件协议和命令见 [FOILPARA.md](FOILPARA.md)。
该入口将外部估计的截面 CL 直接传入物理迁移过程，不调用内部 SLD 网络替代。
旧 `predict` 命令和 `bridge.py` 保持原接口，仍供已有 SuperWing / 多智能体桥接流程使用。

本轮遵循“三维先只测试不训练”：新增 `python main.py test-3d`，
仅进行临时合成输入的三维前向与导出格式检查，优化步数为零。
下文旧 `smoke-test`、`train-sld`、`train-3d` 及其历史验证记录供后续使用，
不是本轮执行的三维工作。

已只读检查真实 SuperWing 与 SuperFoil VT-GU 元数据，并在 VT-GU 上执行
一次二维诊断优化步。诊断检查点位于
`checkpoints/diagnostics/airfoil_vt_gu_20261109_1step.pt`，不能作为充分训练或精度验收的模型。
尚未完成论文的 300 轮二维训练，也没有训练三维模型或发布生产预测。
本轮三维验证的详细结果见 `FOILPARA.md`。

论文架构的边界：当前三维实现沿用三个独立单通道解码器；论文附录 A2
文字描述的是一个解码器最终输出三通道。这是已有实现的架构差异，
本轮为保持 v2 检查点兼容而保留，不将其描述为逐层完全复现论文。

## 依据

阅读的本地论文为桌面 md 目录中的：

2025_AIAAJ_基于物理嵌入式迁移学习的后掠翼快速气动预测.md

对应 Yang 等的 Rapid aerodynamic prediction of swept wings via physics-embedded transfer learning，AIAA Journal 63(6), 2545–2559，2025，DOI 10.2514/1.J064576。同时查看了附录 A1、A2、A4 的网络图。

本次实现是按现有 SuperFoil / SuperWing 数据接口适配该方法，不宣称复现原论文的训练误差、训练数据规模或精度结论。没有可直接加载的本地预训练权重，本次没有启动完整数据集的数百轮训练。

## floGen 结构

| 位置 | 职责及本次使用方式 |
|---|---|
| flowvae/base_model/conv.py | 一维卷积编码器、插值卷积解码器和 U-Net 跳跃连接；用于二维预训练 |
| flowvae/base_model/resnet.py | ResNet 编码器、解码器和 U-Net；用于三维修正 |
| flowvae/base_model/mlp.py | MLP 构造器；用于 SLD 三分支模型 |
| flowvae/vae.py | 条件编码器/解码器与 U-Net；用于二维条件预测 |
| flowvae/dataset.py | 旧数据集封装，倾向一次读入数据；新入口改用原格式只读内存映射 |
| flowvae/post.py | 顶点法向、面积、摩擦基底与表面力积分；用于 SLD 训练标签 |
| flowvae/ml_operator/ | 训练循环、交叉验证、模型配置与权重加载 |
| flowvae/app/wing/models.py | 包含论文使用的模型构造及后续模型；顶层还导入 Transformer 等可选依赖 |
| flowvae/app/wing/api.py | 原有机翼 API 和物理变换；依赖 cfdpost/CST 建模及旧模型配置 |
| examples/wing/run_kfold.py | 论文训练脚本，使用旧路径、旧参数切片和 101×321 网格 |
| main.py | 本次新增：三个模型、物理迁移、只读数据接口、训练、检查点与导出 |
| tests/test_main.py | 本次新增：网络、物理变换、训练和跨项目文件兼容测试 |

复用底层模块，避免让新入口依赖 CFD 几何生成工具或无关 Transformer 依赖。旧训练脚本的 load_encoder_decoder 导入已改为 flowvae.app.multipoint.models，其余旧训练接口问题不由新入口隐式修补。三维分支直接组合原有 ResNet U-Net 编解码器；没有沿用当前 BranchUnet 包装器中访问不存在 decoder 属性的路径。

## 论文到代码的映射

### 二维翼型模型

附录 A1 的一维卷积 U-Net：三组编码通道为 16、32、64，潜变量为 32 维；与工况拼接后，经过 64、128 维全连接层连接解码器。解码通道为 512、256、128、64，最终输出两个通道。

原论文为 321 个点；适配 SuperFoil 的 401 个点后，编码特征长度变为 100、24、5，解码恢复为 401。数据接口接收原始 y/c、工况和 Cp、1000×Cf；网络内部按训练集各变量的最小值和最大值归一化，预测后还原为原存储单位。

完整论文工况为 Mach、CL、Re。用户已确认二维数据根目录为 D:/Superfoil，真实雷诺数为 2×10^7，因此该目录和 Re=20000000 已设为二维默认配置。Re 在网络内以“百万”为单位表示，第三工况值为 20，仅为数值单位转换。SuperFoil VT 原文件没有 Re 列，不会把这个配置写回源数据。

- 默认 condition-dim=3，保留 Mach、CL、Re 三个工况；未传入覆盖参数时，二维适配器采用用户确认的 Re=2e7。
- --reynolds 接受实际 Reynolds 数，并显式覆盖二维默认值。
- --reynolds-file 接受 (N,) NPY，顺序与对应子集 index.npy 相同；显式文件优先于缺省常数，不能与显式 --reynolds 同时使用。
- condition-dim=2 仍是仅供显式选择的 Mach/CL 变体；默认流程不省略 Re。
- --foil-root 默认 D:/Superfoil，可显式指定其他原格式目录；训练仍需通过 --foil-subset 明确选择 VT-BUF、VT-DD 或 VT-GU。
- 二维检查点记录 Re 的来源及数值或文件路径，三维检查点保留前序二维模型的这项配置。此默认值仅属于二维适配器；三维 Re 继续使用显式参数、显式文件或原始日志，截面的有效 Re 仍按论文物理变换计算。

### SLD 模型

采用论文评估过、原仓库 API 也使用的数据驱动 SLD 方案，不默认依赖 VLM 或真实 CFD 的 SLD 来推理。

三分支 FNN：工况分支 16 维，平面/展向参数分支 32、32，翼型分支 64、64，拼接后经过 64 维隐藏层输出 128 个展向站位。SuperWing 的 38 个参数分为 17 个平面/展向参数和 21 个 CST/翼根厚度参数；不把旧 29 维输入切片直接套到新参数上。

训练标签从 origingeom.npy 顶点和 data.npy 表面量计算：按原作者的切向与法向定义恢复摩擦向量，进行单元力积分，再除以截面弦长及条带展向宽度。积分使用 float64，标签提供给训练器时转换为 float32。

真实 SLD 只用于这一训练损失。预测及三维残差模型输入使用 SLD 网络输出，不从 CFD 目标场提取真实 SLD。

### 后掠理论与三维网络

保留论文式 (1)～(4)：

- 二维有效形状的相对 y 除以后掠角余弦。
- 二维有效 Mach 乘以余弦。
- 二维有效 CL 除以余弦平方。
- 二维有效 Re 为真实三维 Re × 局部弦长 × 余弦；SuperWing 使用参考弦长 1。
- 二维输出 Cp/Cf 乘以余弦平方后作为三维先验。

论文使用单段机翼。针对 SuperWing 的折点机翼，本入口采用暴露翼面首末截面四分之一弦点连线的全局后掠角，保持论文单一全局余弦修正形式；额外折点和复杂三维效应留给残差网络学习。这是数据/几何范围扩展，不代表已验证论文结论可直接推广到折点机翼。

截面形状从顶点在展向相邻站位的均值得到，转换到局部弦坐标。沿上下表面分别按物理 x/c 线性插值到 SuperFoil 的 clustcos 参考位置；二维输出再插值回三维单元中心。不会按数组下标盲目把 401 点拉伸为 256 点。

SuperWing 的尾缘封口单元单独使用上下尾缘二维值的平均。二维流向摩擦转换到三维轮廓切向时处理下表面方向，并做 1000→150 的存储倍率转换；展向摩擦由三维残差分支预测。上述均是网络内部物理/数值变换，不会写回源数组。

三维网络输入为 3 个几何坐标通道和 2 个修正先验通道；全局 Mach/AoA 与压缩特征图拼接。编码器六组通道为 32、64、64、128、128、256；解码器为 128、128、64、64、32、32，三个输出分支对应 Cp、150×Cf_tau、300×Cf_z。只在轮廓方向下采样；128 个展向站位保持不变，256 个周向单元依次压缩至 4，再恢复至 256。

三维训练冻结二维模型及 SLD 模型，包括 eval 模式下的统计量；最终结果为修正二维先验加三维网络残差。二维 Cp/Cf 和 SLD 先经过逆归一化，再进行式 (1)～(4) 的物理变换；归一化的网络输出不能直接乘余弦平方。

三维网络输入的五个通道分别按训练样本范围归一化。残差解码器预测 `(目标场 - 二维先验) / 目标场范围`，恢复残差时只乘目标场范围，不加目标场最小值，再与原单位的先验相加。训练和验证的 MSE 都在最终表面场按变量归一化的空间计算。

## 三维数据接口

输入沿用 SuperWing 原布局：

    <wing-root>/
      configs.dat          # G×55，另有原始表头
      index.npy            # N×12
      geom0.npy            # G×3×128×256
      origingeom.npy       # G×3×129×257
      data.npy             # N×3×128×256，仅训练需要
      running_info.jsonl   # 三工况模型的 Re 可从原始日志关联读取
      training_samples_index.txt  # 训练时需要的官方训练样本行号

当前 A/B 接口使用四个主要文件；神经网络物理迁移另需原库已有的 origingeom.npy，避免猜测网格顶点、面积和截面几何。A/B 读取器允许原目录中存在该配套文件。

使用原始编号、形状下标与工况标签各自的含义。Re 日志通过 configs[g,0] 的原始运行编号和工况标签关联，不把原始编号当作几何数组下标。没有 Re 日志时，三工况模型要求显式提供真实数值或文件。

预测不读取输入 data.npy，输入可以只提供几何和工况配套文件。导出到 D:/data/foilperf/network：

    data.npy              # float32 神经网络结果，N×3×128×256
    index.npy             # 原样复制输入元数据
    configs.dat           # 原样复制
    geom0.npy             # 原样复制，保持坐标精度
    origingeom.npy         # 原样复制
    running_info.jsonl    # 输入存在时原样复制
    prediction_info.json # 完成状态、通道、倍率和来源说明

index.npy 的系数列是原数据元数据的副本，不冒充网络积分结果；prediction_info.json 显式记录其角色。网络预测仅在 data.npy 中。不会把 CFD 的场值复制成预测。

导出保留输入完整样本集合与原行号，能直接被当前 D:/dspy 的 SuperWing 比较器配对。中断时只留下 data.npy.partial；全部批次完成才改名为 data.npy。输出目录须为空，禁止覆盖数据源、在源数据目录内导出或覆盖已有预测。

## 二维数据接口

--foil-root 默认指向 D:/Superfoil，其下保留原有 VT 子目录；--foil-subset 明确选择 VT-BUF、VT-DD 或 VT-GU：

    <foil-root>/VT/<subset>/
      data.npy             # N×3×401；y、Cp、1000×Cf
      index.npy            # N×13
      traindataindex.txt   # 官方训练样本行号

保持原始形状、通道、倍率和符号。BUF/GU 读取 Mach 第 5 列、CL 第 9 列；DD 读取 Mach 第 3 列、目标 CL 第 6 列，不能把第 9 列实际攻角当作 CL。以上下标均从 0 开始。几何直接读取 data[:,0]，所以 GU 的实际厚度变化自然保留。

FT 本地缺少流场数组；不伪造 FT 训练场，不把其 14 个 CST 系数套成 VT 的 20 系数。

## 训练与验证

使用官方训练样本列表；从其中按几何身份再划分固定验证组，默认 10%，seed=42。官方测试形状不进入训练/验证集。

按照论文第 III.D 节，输入、输出统一采用逐变量最小最大值归一化 `(x - xmin) / (xmax - xmin)`，替代均值/标准差标准化。范围只由本次选中的训练样本拟合，空间分布通道在所有训练样本和空间点上共享一组范围；SLD 也使用一个覆盖训练样本及展向点的 CL 范围。常量变量使用单位分母，训练范围外的验证/推理值不裁剪。三维先验通道的范围由冻结的二维/SLD 模型生成的先验确定，不使用 CFD 目标冒充输入。范围拟合采用流式读取，不修改源数组，验证集和测试集不参与拟合。

最小值和范围作为模型 buffer 与权重一起保存，加载后使用相同的归一化和逆归一化。检查点格式已更新为 `physics_embedded_superwing_v2`，并记录归一化约定；原 v1 均值/标准差权重缺少输入/输出范围，必须重新训练，不会被当作新模型静默加载。论文未给出逐变量数值范围，本实现采用训练集范围这一明确约定，不能据此宣称已使用作者原始范围。

训练采用 Adam、MSE。二维默认 300 轮、前 20 轮预热为 1e-5→1e-4，随后按 0.95 指数衰减；三维按论文将调度拉长 3 倍，默认 900 轮。SLD 默认 900 轮，按原仓库 warmup_lr_4 使用 4 倍调度。新数据网格更大，三维默认 batch-size=2，可显式设置。

SLD 检查点记录训练几何编号；三维训练前拒绝其与三维验证形状重叠，防止辅助模型通过 SLD 标签泄漏验证信息。默认同一 seed/划分参数会得到一致分组。

各训练命令只在结束时保存验证损失最好的状态；不覆盖已有检查点，记录训练步数、划分与前序模型来源。检查点使用带版本和通道约定的自包含字典，加载采用 weights_only=True。未训练检查点、阶段错误或不匹配的通道约定不能用于真实导出。

完整训练需要用户实际运行，以下示例只是入口，不代表本次已经完成训练。

## 命令

在已安装 PyTorch、NumPy 的环境运行；当前已验证环境是 C:/Users/fff/miniconda3/envs/3.12torch/python.exe。新入口不需要 SciPy、cfdpost、CST 建模库或 Transformer 依赖。

只读检查已确认的数据目录：

    python main.py inspect --wing-root D:/superwing

检查二维数据，默认使用已确认的 D:/Superfoil，并区分原文件缺少 Re 列与已配置 Re：

    python main.py inspect --foil-subset VT-GU

CPU 三阶段前向/反向诊断，不导出生产预测：

    python main.py smoke-test --device cpu

论文三工况二维预训练（默认目录 D:/Superfoil、真实 Re=2e7）：

    python main.py pretrain-2d --foil-subset VT-GU --checkpoint checkpoints/airfoil.pt --device cpu

VT-GU 仅为命令示例，训练子集由调用者明确选择。也可显式写出同一配置：

    python main.py pretrain-2d --foil-root D:/Superfoil --foil-subset VT-GU --condition-dim 3 --reynolds 20000000 --checkpoint checkpoints/airfoil.pt --device cpu

更换二维数据集且 Re 不同时，须通过 --reynolds 或 --reynolds-file 覆盖已确认的默认值；仅在明确需要 Mach/CL 变体时使用 --condition-dim 2。

训练 SLD 和三维残差：

    python main.py train-sld --wing-root D:/superwing --checkpoint checkpoints/sld.pt --device cpu
    python main.py train-3d --wing-root D:/superwing --airfoil-checkpoint checkpoints/airfoil.pt --sld-checkpoint checkpoints/sld.pt --checkpoint checkpoints/wing.pt --device cpu

原始日志提供 SuperWing 的真实 Re；换成其他输入目录时，若无相应日志且二维检查点需要 Re，须传 --reynolds 或 --reynolds-file。

使用已训练检查点导出给当前项目：

    python main.py predict --checkpoint checkpoints/wing.pt --input-root D:/data/foilperf/cfd --output-root D:/data/foilperf/network --device cpu

以 --max-steps 1 --epochs 1 --max-train-samples <数量> 可做明确的短诊断，不应把这种权重当作经充分验证的气动模型。

E 的 D:/data/foilgeo 仍是参数 JSON。新入口的直接输入是表面网格与工况，不新增从参数到 CAD/CFD 网格的生成器。

## 验证范围

- 回归测试涵盖原倍率、DD 列映射、几何分组划分、二维已确认 Re 默认值与显式标量/文件覆盖、三维原始 Re 日志不被二维默认值覆盖、缺失数据拒绝、后掠变换和摩擦方向、插值、冻结参数、三阶段短训练、检查点保存/加载；另覆盖逐变量范围、常量通道、范围外推、训练集独立拟合、物理变换前逆归一化、残差不重复加偏置及旧检查点拒绝。
- 在临时合成数据上实际完成三个训练命令各一步更新，再通过 predict 导出，交由 D:/dspy 的真实 SuperWing 比较器读取，验证四个样本/三通道的格式兼容。
- 只读抽查真实 SuperWing 的 n=0、10829、28855，核对参考网格、128 站位 SLD 标签和 Re 关联；只读检查 SuperFoil 三个 VT 子集的首样本。
- 未修改数据库文件，没有把随机/短训练结果写入生产 network 目录。
- GPU 已复测通过：Python 3.12、PyTorch 2.10.0 / CUDA 13.0、cuDNN 9.25.1.1、RTX 3060。此前 `Cannot load symbol cudnnGetVersion` 的原因是直接启动 Python 的检查进程未包含 Conda `Library/bin` DLL 搜索路径；补齐进程内 PATH 后，最小卷积及三阶段合成数据的 GPU 前向、反向和参数更新均通过。入口仍默认 CPU，可显式选择 --device cuda。该结果是短诊断，不代表完成数百轮训练。

若直接使用绝对路径启动 Python，在 PowerShell 中可按本次已验证方式设置当前会话的 DLL 搜索路径（不修改系统 PATH）：

    $torchEnv = 'C:\Users\fff\miniconda3\envs\3.12torch'
    $env:PATH = "$torchEnv\Library\bin;$torchEnv\Scripts;$torchEnv\DLLs;$torchEnv;$env:PATH"
    & "$torchEnv\python.exe" -B main.py smoke-test --device cuda

测试：

    python -B -m unittest discover -s tests -v

可设置 DSPY_PYTHON 为当前多智能体项目的 Python 可执行文件，以启用导出文件的跨项目读取检查。默认测试均使用临时合成文件。
