# foilpara 输入、神经网络输出与运行说明

当前实际交付约束已按用户要求收紧为单算例：仅一副机翼、一个工况、一条 SLD，
`G=N=1`、`g=k=0`。`D:/data` 仅在五个末级目录保留 `format.md`；
各末级说明是正式单算例交付要求。三目录适配器在代码层强制单例、六列索引和 41 列几何配置。
几何配置仅保留原始编号、38 个几何参数、一份 Mach 和一份攻角，不保留多工况槽位。
输出目录可为空或仅包含普通文件 `format.md`，导出会保留该文档；存在其他内容仍拒绝覆盖。

此入口对应 `readforagent` 指定的数据流：机翼几何、工况和外部估计的展向升力分布输入神经网络，输出机翼表面压强和摩擦分布。二维预训练与三维训练使用独立命令；本文描述的预测/导出入口仅前向，不更新参数。当前三维训练采用冻结二维网络和几何驱动 VLM，详见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。

## 项目结构

| 位置 | 职责 |
|---|---|
| `flowvae/base_model/` | 原有卷积、ResNet、MLP 编解码模块 |
| `flowvae/vae.py` | 条件 U-Net 与编码器/解码器组合 |
| `main.py` | 二维模型、SLD 模型、三维残差模型、后掠理论迁移、独立训练命令和检查点 |
| `foilpara.py` | 三目录数据适配、外部 SLD 预测导出、零优化步三维测试 |
| `bridge.py` | 独立旧增量训练/几何预测协议，要求已训练 FNN-SLD，不接收当前 VLM 权重 |
| `tests/test_foilpara.py` | 数据约定、外部 SLD 生效、只读和纯推理验证 |
| `examples/wing/run_kfold.py` | 原作者论文交叉验证示例，与当前专用 train/test 入口独立 |

当前输入是否完成、来自哪个 case、采用何种 SLD，以各叶目录真实文件和 `source_case.json` 为准。本文的单 case 协议保持 SuperWing 网格及物理量定义，几何配置简化为 41 列。没有添加 CFD 网格生成器；几何参数须连同已经生成的表面网格提供。完整多智能体 A–E 的提示词与调度不属于此次神经网络入口改动。

## 输入文件

默认根目录为 `D:/data/foilpara`，可通过 `--input-root` 指定同格式测试目录。

```text
foilpara/
  foiloc/
    index.npy           # (1,6)，唯一几何/工况，前两列均为 0
    reynolds.npy        # (1,)，唯一实际 Reynolds 数，例如 20000000
  foilgeo/
    configs.dat         # (1,41)，首行表头；Mach、攻角各一列
    geom0.npy           # (1,3,128,256)，单元中心坐标
    origingeom.npy      # (1,3,129,257)，网格顶点坐标
  foilsld/
    index.npy           # (1,6)，必须与 foiloc/index.npy 逐值相等
    sectional_cl.npy    # (1,128)，唯一一条外部估计无量纲截面升力系数
```

当前 `N=G=1`，不接受第二个几何或工况。NPY 必须是真实数值数组，不能是 Git LFS 指针。输入按只读内存映射加载，不会修改数据库，也不读取 CFD `data.npy`。

工况索引前六列按原 SuperWing 顺序保留，零基下标如下：

| 列 | 含义 |
|---|---|
| 0 | 几何数组局部下标，范围 0 到 G−1；不是原始运行编号 |
| 1 | 唯一工况标识，固定 0 |
| 2 | 攻角，单位为度 |
| 3 | Mach |
| 4 | 参考半翼面积，正值 |
| 5 | 半翼展，正值 |

`configs.dat` 只有一行 41 列：第 0 列为原始几何编号；`1:39` 为 38 个几何参数；第 39 列为唯一 Mach；第 40 列为唯一攻角。这里的切片采用 Python 结束下标不包含的约定。工况索引的 Mach/攻角必须与这两个单值一致。原始 SuperWing 训练库的 55 列读取仍独立保留，不要求也不接受在新三目录输入中填充八组数值。

输入校验还包括：Mach 有限且为正，索引与配置中的 Mach/攻角一致，所有坐标有限，周向首末顶点闭合、中心坐标为四顶点平均，展向顶点站位逐段严格递增。用于默认 VLM 准备的 Mach 还须满足该势流适配器的 `[0,1)` 范围。

`sectional_cl.npy` 是在 128 个展向单元中心处、应用后掠变换之前的截面 CL。展向顺序必须与几何网格的展向轴一致；不能直接传入牛顿单位的升力、总机翼 CL，或已经除以后掠角余弦平方的二维有效 CL。升力数据的 `index.npy` 用于绑定样本身份和顺序，错序时直接拒绝，不自动猜测或排序。数据提供方仍需确保每行数值与其索引、展向站位相符。

三工况二维模型需要 `reynolds.npy`：必须是实际 Re，逐样本有限且为正。不会将二维数据的默认 Re 自动套到三维。只有明确采用二维 Mach/CL 两工况检查点时，该文件可省略。

模型收到外部 `sld` 后直接用于物理迁移，绕过内部 SLD 网络；此入口缺失 SLD 会报错，不会静默回退。当前三维检查点记录 `sld_network_used=False`、`sld_source=pytornado_vlm`；`main.load_checkpoint` 为它设置 `requires_external_sld`，未提供外部 SLD 时拒绝前向。`main.py predict` 与 `bridge.py` 的旧 FNN-SLD 协议不能直接使用当前检查点；旧协议继续支持满足其已训练 FNN-SLD 前序要求的历史检查点。

### 当前默认的低保真 SLD 准备方式

`superwing_reader.py` 的 `read` 和 `iter_cases`，以及 `superwing_io.py` 的 `inspect/export`，现已默认选择 `sld_mode="vlm"`。`low_fidelity_sld.py` 根据实际顶点几何和攻角，调用官方 PyTornado 原生 VLM 生成这条 `(1,128)` 截面 CL；使用同局部 x/c 的上下表面平均弯度面，保留扭转与上反，以暴露翼根平面镜像，默认每半翼 32 个展向面板、4 个弦向面板，再按物理展向坐标映射至 128 个参考单元中心。SLD 求解不接收 CFD 场。来源记录为 `sld_source=pytornado_vlm`、`sld_requires_cfd=false`，`sld_details` 记录官方版本及求解网格、映射和限制。

这条默认估计是不可压缩势流低保真结果，Mach 仅记录，未附加 PG 或其他 Mach 修正，不解析激波、黏性或分离。原 SuperWing 读取器为交付对比真值仍读取对应 CFD `field`，这与 SLD 是否利用 CFD 是两件独立的事。显式 `--sld-file` 直接采用调用者的一条外部估计；只有明确选择 `--sld-mode cfd-reference` 才由 CFD 真值积分并标为 `cfd_reference_integral`，不能作为独立估计或盲预测依据。官方扩展缺失或求解失败时会报错，不自动退回参考积分。直接运行 `predict-foilpara` 仍要求已经准备好 `foilsld` 数组，VLM 准备由上述读取/导出步骤或独立适配函数执行。

读取与交付命令见 [SUPERWING_IO.md](SUPERWING_IO.md)，环境、官方求解器调用和数值检查见 [PYTORNADO.md](PYTORNADO.md)。

## 输出文件

默认输出目录为 `D:/data/foilperf/network`，必须为空或仅含普通 `format.md` 文件；禁止覆盖来源目录、其子目录或祖先目录。

```text
network/
  data.npy                  # float32，(N,3,128,256)
  index.npy                 # (N,6)，只保留真实几何与工况元数据
  configs.dat               # 从 foilgeo 原样复制
  geom0.npy                 # 从 foilgeo 原样复制
  origingeom.npy            # 从 foilgeo 原样复制
  source_sectional_cl.npy   # 本次使用的外部截面 CL
  source_reynolds.npy        # 输入提供 Re 时复制
  prediction_info.json      # 状态、通道倍率、SLD 来源与模型状态检查
```

`data.npy` 的三个通道为 `Cp`、`150 × Cf_tau`、`300 × Cf_z`，沿用 SuperWing 存储倍率。摩擦包含表面轮廓切向和展向两个分量，不能把第三通道丢弃或把存储倍率当成真实系数。二维数据原有 `1000 × Cf` 的倍率及下表面方向在模型内部转换。

输出索引只有六列，不填造 CFD 升阻力或力矩系数。`prediction_info.json` 记录 `index_layout=geometry_conditions_v1`、`sld_source=external_foilsld_sectional_cl` 和 `training_steps_performed=0`。读取方应使用已支持六列纯预测索引的接口；旧的仅接受十二列 CFD 数据的读取器需要识别该标记。

导出先写 `data.npy.partial`，完成所有样本、确认模型参数和归一化 buffer 的完整哈希不变后才发布 `data.npy`。只有 `prediction_info.json` 的 `status=complete` 表示导出完成。异常时保留不完整状态供排查。

## 命令

在项目根目录运行，使用本机 Conda `flogen`；解释器为 `C:/Users/fff/miniconda3/envs/flogen/python.exe`。网络推理需要 PyTorch 与 NumPy；只有准备 VLM SLD 时才调用 PyTornado 和 SciPy。

三维前向与文件接口诊断不需要真实输入或已训练权重：

```powershell
python -B main.py test-3d --device cpu
```

该命令在临时目录构造唯一一条单算例，以随机模型验证前向和导出，报告 `diagnostic_untrained`、输出 `(1,3,128,256)`、索引 `(1,6)`、优化步数 0。它不会写生产 `network`，也不用于说明气动精度。

正式预测需提供已训练、已完整下载且验证过的三维检查点。以下路径以当前任务保留的第 80 轮为例，并不表示它是未来的最新轮：

```powershell
python -B main.py predict-foilpara --checkpoint weights/3d/superwing_200_bs8_20261111_190048/checkpoints/epoch_080.pt --device cuda
```

命令默认读取上述 `D:/data/foilpara`，向 `D:/data/foilperf/network` 写唯一算例；`--input-root` 和 `--output-root` 可选择其他同格式目录。已有数据时拒绝覆盖，普通 `format.md` 文件保留。运行前需先完成几何、工况、实际 Re 与外部 SLD 的准备。

当前 VLM 检查点必须记录受过训练的二维前序，以及几何驱动、不使用 CFD 的 PyTornado SLD 来源；不要求训练一个未使用的 FNN-SLD。二维训练与三维训练不由本预测命令触发，分别见[二维神经网络使用说明](二维神经网络使用说明.md)和[IMPLEMENTATION.md](IMPLEMENTATION.md)。

## 接口回归范围

[tests/test_foilpara.py](tests/test_foilpara.py) 覆盖单算例形状、外部 SLD 生效、元数据对应、实际 Re、几何合法性、源数据只读、前向不改模型状态、输出目录冲突和导出协议。[tests/test_wing_vlm_checkpoint.py](tests/test_wing_vlm_checkpoint.py) 覆盖当前 VLM 检查点的前序验证与缺失外部 SLD 拒绝。

```powershell
python -B -m unittest discover -s tests -p test_foilpara.py -v
python -B -m unittest discover -s tests -p test_wing_vlm_checkpoint.py -v
```

这些测试使用临时数据；接口测试通过不等同于模型在所有机翼和工况下都已达到气动精度要求。默认 VLM 的数值与 CFD 隔离检查集中说明在 [PYTORNADO.md](PYTORNADO.md)，不在各接口文档中重复记录历史 pass 数量和临时运行状态。
