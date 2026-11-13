# SuperWing 单 case 读取与输入输出导出

`superwing_reader.py` 读取本地 SuperWing 参考库，`superwing_io.py` 将指定的一条“机翼与工况”样本导出为当前项目的单 case 文件。参考库保持只读，导出目录内每份数据只含一架机翼、一个工况，不将多条样本打包成数据集，不自动覆盖上一条 case。

这里的 CFD 数据来自参考库已经完成的求解结果。读取或导出不会执行新的 CFD，不启动神经网络训练；只有显式提供已训练三维检查点时才调用模型生成网络预测。

## 原始参考库与单 case 文件的区别

参考库默认位于 `D:/Superwing`。读取器沿用其实际文件：`configs.dat` 为带首行表头的 55 列配置表，`index.npy` 为 `(N_source,12)`，`geom0.npy` 为 `(G_source,3,128,256)`，`origingeom.npy` 为 `(G_source,3,129,257)`，`data.npy` 为 `(N_source,3,128,256)`；表面场三个通道依次为 `Cp`、`150×Cf_tau`、`300×Cf_z`。真实 Re 可从原始 `running_info.jsonl` 按原始运行编号和原工况号关联，也可通过显式参数指定。

单 case 配置文件为 `(1,41)`：零基第 0 列是 `original_id`，第 1～38 列保留原 38 个几何参数，第 39 列是唯一的 `mach`，第 40 列是唯一的 `aoa_deg`。Mach 和攻角各只保存一份。几何参数仍按原顺序包含平面参数、厚度/弯度/扭转变化参数、上下表面各十个 CST 系数以及 `tcroot`。单 case 坐标文件保留原网格分辨率，形状分别为 `(1,3,128,256)` 和 `(1,3,129,257)`。

输出的工况索引前六列依次为 `g`、`case_id`、`aoa_deg`、`mach`、`ref_area`、`half_span`。单 case 只有第 0 行，局部几何下标 `g=0`，`case_id=0`。原始库中的几何数组下标、工况号与原始运行编号另存入来源信息，不能因局部重编号而丢失。原始库的样本行号 `row` 与几何编号是不同概念，同一机翼的不同工况位于不同样本行。

完整 41 列定义、表头与坐标示例见输出位置的 `foilpara/foilgeo/format.md`；工况及展向升力的逐列格式见对应叶目录的 `format.md`。这些说明文件是人工文档，不是读取器的数值输入，也不能替代真实网格文件。

## Python 读取接口

使用 `SuperWingReader(root="D:/Superwing", reynolds=None, reynolds_file=None)` 创建读取器，并通过 `with` 管理生命周期。命令行对应参数名为 `--source-root`。数组以只读内存映射方式访问，每次读取只复制所选 case 的坐标、场值等数据，不将整个约 22.7 GB 的表面场文件一次装入内存；退出上下文后释放原库文件句柄。

```python
from superwing_reader import SuperWingReader

with SuperWingReader(root="D:/Superwing") as reader:
    case = reader.read(0)
    print(case.metadata)
    print(case.configs.shape)       # (1,41)
    print(case.centers.shape)       # (1,3,128,256)
    print(case.vertices.shape)      # (1,3,129,257)
    print(case.field.shape)         # (1,3,128,256)
    print(case.sectional_cl.shape)  # (1,128)
```

`read(row, sld=None, sld_mode="vlm")` 的 `row` 是参考库 `index.npy` 和 `data.npy` 的同一零基样本行号，不能直接传原始几何运行编号。返回 case 的主要字段如下；数值数组统一复制为 C 连续 `float64`，第一维长度均限定为 1，确保下游每次处理一个 case。

| case 字段 | 形状或类型 | 内容与对应关系 |
|---|---|---|
| `configs` | `(1,41)` 数值数组 | 原始编号、原 38 个几何参数、唯一 Mach 和攻角 |
| `index` | `(1,6)` 数值数组 | 前两列重编号为 0，随后为攻角、Mach、参考半面积、半翼展 |
| `cfd_index` | `(1,12)` 数值数组 | 同一 case 的完整 CFD 索引，前两列同样重编号为 0；其余列保留参考值 |
| `centers` | `(1,3,128,256)` 数值数组 | 对应唯一几何的单元中心 x/y/z 坐标 |
| `vertices` | `(1,3,129,257)` 数值数组 | 对应唯一几何的参考网格顶点 x/y/z 坐标 |
| `field` | `(1,3,128,256)` 数值数组 | 原始参考 CFD 表面场，通道倍率为 1、150、300 |
| `reynolds` | `(1,)` 数值数组 | 此 case 的实际 Re，例如 `20000000`，不是以百万为单位的 `20` |
| `sectional_cl` | `(1,128)` 数值数组 | 同一 case 的 128 个展向单元中心截面 CL |
| `metadata` | 字典 | 原始样本行、原始编号、原几何下标、原工况号、Re 与 SLD 来源等追踪信息 |

`cfd_index` 的第 6～8 列为原始求解器给出的 `CL/CD/CMz`，第 9～11 列为参考表面网格积分得到的同三项系数。两组三项对应同一工况，不是两个工况。坐标与场值保留参考文件的物理位置、周向顺序及原存储倍率；同一个 `j,i` 指向同一表面单元中心。x 主要沿弦向，y 表示厚度与上反方向，z 为展向；几何长度沿用参考弦长为 1 的模型尺度。

### 默认 PyTornado VLM、外部 SLD 与显式参考积分

如果不提供 `sld`，读取器默认使用 `sld_mode="vlm"`，由 `low_fidelity_sld.py` 调用官方 PyTornado 原生涡格求解器，根据机翼几何和攻角估计截面 CL。来源标记为 `sld_source="pytornado_vlm"`、`sld_requires_cfd=false`，`sld_details` 保存求解器版本、网格、弯度面、镜像与插值约定。这条 SLD 计算路径不接受 CFD 场作为参数，不从真实压力或摩擦提取升力；SuperWingReader 仍复制 `field` 用于并行交付参考真值，这不表示 VLM 使用了它。

VLM 将上下表面在相同局部 x/c 处平均成弯度面，保留参考几何的扭转与上反，将原 `[x,y高度,z展向]` 映射为 VLM `[x,y展向,z高度]`，以实际暴露翼根平面镜像。默认每半翼 32 个展向面板、4 个弦向面板，沿物理展向位置插值为 128 个参考单元中心的 CL。当前使用不可压缩势流 VLM，Mach 仅记录，未添加 Prandtl–Glauert 或其他 Mach 修正，不能解析激波、分离或黏性效应。实现和环境说明见 [PYTORNADO.md](PYTORNADO.md)。

只有显式选择 `sld_mode="cfd-reference"` 时，才从本条参考 CFD 表面场和顶点几何积分计算截面 CL，来源标记为 `cfd_reference_integral`、`sld_requires_cfd=true`。该模式用于参考诊断，已使用真实 CFD，不能当作独立估计或盲预测精度证据；它也不是 VLM 缺失时的自动回退。

如果已有独立估计的截面 CL，可以向 `read` 传入形状为 `(128,)` 或 `(1,128)` 的有限数值数组；返回值统一为 `(1,128)`，来源为 `external_provided`。显式数组优先直接使用，不再调用 VLM 或 CFD 积分。它必须是对应本 case、按参考网格展向顺序排列、后掠修正之前的无量纲截面升力系数，不能传入牛顿单位的升力、总机翼 CL，或者已经经过二维有效 CL 变换的数据。

```python
import numpy as np
from superwing_reader import SuperWingReader

external_sld = np.load("my_case_sectional_cl.npy", allow_pickle=False)
with SuperWingReader("D:/Superwing") as reader:
    case = reader.read(0, sld=external_sld)
    print(case.sectional_cl.shape)  # (1,128)
    print(case.metadata)           # 核对本次 SLD 的外部来源
```

显式 `reynolds` 应传实际正数；`reynolds_file` 应提供按参考库样本行顺序排列的逐样本 Re 数组。两个覆盖方式不能同时提供，缺省使用该 case 在原始日志中关联的真实 Re，不将二维翼型训练的默认值自动套用到三维。

### 逐条遍历

`iter_cases(start=0, stop=None, sld_mode="vlm")` 是生成器，按零基样本行逐条产生独立 case；`start` 包含，`stop` 不包含，省略 `stop` 时遍历到参考库末尾。默认逐条计算几何驱动的 PyTornado VLM；如要参考积分必须明确传 `sld_mode="cfd-reference"`。可边读取边处理，不创建包含多条数据的大数组。下例只展示前三个 case 的来源信息，不写输出文件：

```python
from superwing_reader import SuperWingReader

with SuperWingReader("D:/Superwing") as reader:
    for case in reader.iter_cases(start=0, stop=3):
        print(case.metadata)
        # 在此独立处理这一份 case；不要把 case 追加到列表后整体打包。
```

## 命令行检查与导出

在 floGen 项目根目录运行，环境需安装 NumPy 与 PyTorch。`inspect` 读取指定 case 并打印其结构和来源，不导出数值文件，不训练：

```powershell
python -B superwing_io.py inspect --row 0 --source-root D:/Superwing
```

默认 VLM 还需官方 PyTornado 原生扩展。本机使用已准备的 Conda `flogen` 环境，例如 `C:/Users/fff/miniconda3/envs/flogen/python.exe`；普通未安装 PyTornado 的解释器会明确报错，不会静默读取 CFD 真值代替。`inspect` 与 `export` 均接受 `--sld-mode vlm`（默认）或 `--sld-mode cfd-reference`。参考积分的显式诊断示例为：

```powershell
python -B superwing_io.py inspect --row 0 --source-root D:/Superwing --sld-mode cfd-reference
```

将一条参考样本导出到当前项目的数据根目录：

```powershell
python -B superwing_io.py export --row 0 --output-root D:/data --source-root D:/Superwing
```

可选择覆盖实际 Re，或者提供单 case 的外部截面 CL 文件；下面是参数示例，文件路径需替换为实际存在且顺序正确的文件：

```powershell
python -B superwing_io.py export --row 0 --output-root D:/data --source-root D:/Superwing --reynolds 20000000 --sld-file my_case_sectional_cl.npy
```

需要逐样本 Re 文件时，将 `--reynolds 20000000` 替换为 `--reynolds-file actual_reynolds.npy`。两种 Re 参数互斥，Re 文件按原始参考样本行编号取值；`--sld-file` 则只提供本次选中 case 的 `(128,)` 或 `(1,128)` 数组。

如需同时生成神经网络预测，显式传入实际已训练的三维检查点：

```powershell
python -B superwing_io.py export --row 0 --output-root D:/data --source-root D:/Superwing --sld-file my_case_sectional_cl.npy --checkpoint checkpoints/trained_wing.pt --device cpu
```

有检查点时执行真实模型前向，不能使用随机数、CFD 真值复制或占位数组充当网络结果。没有 `--checkpoint` 时不生成 `foilperf/network/data.npy`，已有 `format.md` 保留；不会为了补齐目录而构造假的预测。

也可在 Python 中复用同一个导出函数：

```python
from superwing_reader import SuperWingReader
from superwing_io import export_case

with SuperWingReader("D:/Superwing") as reader:
    case = reader.read(0)
    export_case(case, output_root="D:/data")  # 默认 VLM SLD，交付输入与参考 CFD，无网络预测
```

## 导出文件树

`--output-root` 指定的是数据根目录，例如 `D:/data`，其下采用固定相对路径。各实际写入的 case 叶目录均附带 `source_case.json` 记录来源，并保留已有 `format.md`。

```text
<output-root>/
  foilpara/
    foilgeo/
      format.md                 # 已有说明文件保留
      configs.dat               # 一行表头与一行 41 个数值
      geom0.npy                 # (1,3,128,256)
      origingeom.npy            # (1,3,129,257)
      source_case.json
    foiloc/
      format.md
      index.npy                 # (1,6)，g=case_id=0
      reynolds.npy              # (1,)，实际 Re
      source_case.json
    foilsld/
      format.md
      index.npy                 # (1,6)，与 foiloc 前六列完全相等
      sectional_cl.npy          # (1,128)
      source_case.json
  foilperf/
    cfd/
      format.md
      configs.dat               # 同一份单 case 41 列配置
      index.npy                 # (1,12)，真实参考 CFD 元数据
      geom0.npy
      origingeom.npy
      data.npy                  # (1,3,128,256)，参考 CFD 真值
      running_info.jsonl        # 单条日志，0_running_info 记录同一实际 Re
      source_case.json
    network/
      format.md
      # 只有显式提供已训练检查点时，才写入真实预测和来源文件。
```

`source_case.json` 的 `source_root`、`source_row`、`original_id`、`source_geometry_index`、`source_condition_id` 分别追踪参考路径、原始样本行号、原始几何编号、原几何数组下标和原工况号；`sld_source`、`sld_requires_cfd` 和 `sld_details` 记录展向升力的来源与求解设置，默认 PyTornado 时分别为 `pytornado_vlm`、`false` 和求解器设置对象。`reference_sld_uses_cfd` 默认 `false`，只有显式参考积分时才为 `true`。`actual_Re` 与 `reynolds_source` 记录实际 Re 及来源。各叶目录共享同次 `export_id`，`output_role` 区分目录角色，`status=complete` 表示本次文件交付完成。不能只看局部 `g=0`、`case_id=0` 判断两次交付是否来自同一参考 case。

CFD 目录的 `running_info.jsonl` 只写本单 case 的 Re 日志：`running_id` 采用原始几何编号，`0_running_info.working_conditions.re` 保存实际 Re；原参考工况号保留在 `source_case.json`，不与重编号后的 0 混淆。该日志不是一次新 CFD 求解记录，不声称本次导出重新完成求解。

## 样本对照示例

选择参考库 `row=0` 时，原始编号为 0，原几何数组下标为 0，原工况号为 0；单 case 工况表为下列一行。该 Markdown 表只用于阅读，实际文件是 `(1,6)` 的数值 NPY，不把表头写入 NPY。

| g | case_id | aoa_deg | mach | ref_area | half_span |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 9.3724910087 | 0.7519974134 | 1.7633693114510065 | 3.1161224020635268 |

同一 case 的参考 CFD 十二列表包含下面两组真实系数；这里只说明参考数据，不把它们复制成网络预测积分结果。

| g | case_id | aoa_deg | mach | ref_area | half_span | cl_solver | cd_solver | cm_solver | cl_surface | cd_surface | cm_surface |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 9.3724910087 | 0.7519974134 | 1.7633693114510065 | 3.1161224020635268 | 0.46849161091134833 | 0.01785562762371063 | 0.4988891191704631 | 0.46833910536356366 | 0.017818920255398732 | 0.49862396903929807 |

## 输出冲突、单 case 生命周期与边界

导出前对本次全部目标路径统一预检。任何目标 case 数据或来源文件已经存在时，整次导出先拒绝，不先写几个目录再发现另一个目录冲突，也不覆盖已有数据。说明文件 `format.md` 可以预先存在并保留，它不能被误判成需要覆盖的 case 数值数据。

同一 `output-root` 代表当前正在交付的一条 case。处理下一条参考样本前，应明确归档上一条 case 的完整相关目录并重新准备空目标，或者显式选择另一个独立输出根目录。遍历读取不会自动覆盖、自动合并或自动将输出变成多 case 文件；不要直接在循环内反复使用同一个非空输出根目录。

来源数据的抽取与数值格式正确不等于新 CFD 工程已就绪。参考翼面坐标是展开的参考表面网格，实际 CFD 所需的确定性根尖封口、水密与正体积检查、体网格质量、边界条件、尺度、物性及求解配置仍需按几何叶目录说明验收。该工具不会运行这些新求解步骤，不改动 `D:/Superwing`，也不把短诊断或参考真值泄漏的结果称为盲预测精度。

## 当前检查点与验证范围

`superwing_io.py export --checkpoint ...` 在交付几何、工况、SLD 和参考 CFD 之后，通过正式三目录入口进行真实网络前向，因此支持记录了 PyTornado 外部 SLD 来源的当前三维检查点。该路径与 `main.py predict` 的原生库旧接口不同，不能因二者都读取 SuperWing 数据就互换使用。没有 `--checkpoint` 时只交付参考输入/真值，不产生网络预测。

当前三维任务与二维前序权重见 [IMPLEMENTATION.md](IMPLEMENTATION.md)，生产输入/输出完整协议见 [FOILPARA.md](FOILPARA.md)。模型运行状态、训练完成轮数及检查点 SHA 应读取真实文件，不通过此文档固定某个运行中的 epoch。

回归范围包括原始行号与局部编号的区分、41/55 列配置、6/12 列索引、Re 关联、来源身份共享、只读源数组、全部目标统一冲突预检，以及 VLM/外部提供/CFD 参考积分三种 SLD 来源的区分。对应 [tests/test_superwing_io.py](tests/test_superwing_io.py)、[tests/test_foilpara.py](tests/test_foilpara.py) 和 [tests/test_low_fidelity_sld.py](tests/test_low_fidelity_sld.py)。物理低保真假设和数值例子集中在 [PYTORNADO.md](PYTORNADO.md)。
