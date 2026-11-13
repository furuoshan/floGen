# 官方 PyTornado 低保真展向升力估计

当前默认的展向升力估计由 `low_fidelity_sld.py` 调用官方 PyTornado 原生 VLM 完成。输入只有机翼参考几何、Mach 与攻角，输出为单 case 的 `(1,128)` 无量纲截面 CL 和求解来源信息。它既用于准备 `foilpara/foilsld/sectional_cl.npy`，也用于当前三维训练的 VLM 缓存。SLD 求解本身不训练神经网络、不执行新的 CFD，也不使用参考 CFD 压力或摩擦场；三维训练读取它的结果作为固定低保真输入。

## 环境与官方求解器

本机安装环境为 Conda `flogen`，Python 3.12，解释器为 `C:/Users/fff/miniconda3/envs/flogen/python.exe`；已安装 PyTornado 0.5.0。服务器同名环境使用 Python 3.11，原生扩展为独立的 Linux 构建，不能与本机 Windows wheel 混用。官方五个 C++ 源文件按原算法编译为本地 wheel，其涡格数学算法保持不变。围绕旧版本 Python 层及接口做的兼容处理包括现代 matplotlib 版本判断、`collections.abc`、Windows 的 C `long` 与 NumPy 2 整数类型匹配，以及移除依赖已废弃 `scipy.misc.derivative` 的导数调用方式。

`low_fidelity_sld.py` 构造官方原生例程所需的几何与状态对象，实际依次调用：

| 官方调用 | 本流程中的作用 |
|---|---|
| `pytornado.aero.c_vlm.py2c_lattice` | 从适配后的面板角点建立官方涡格、控制点、法向和面板面积 |
| `pytornado.aero.c_vlm.py2c_downwash` | 生成官方诱导速度影响矩阵 |
| `pytornado.aero.c_vlm.py2c_boundary` | 生成法向速度边界条件右端项 |
| `scipy.linalg.lapack.dgesv` | 求解环量线性方程，采用官方 Python driver 同一 LAPACK 解法 |
| `pytornado.aero.c_vlm.py2c_results` | 根据求得环量计算官方面板力和结果 |

本项目负责参考机翼的弯度面提取、坐标映射、面板打包与 128 个站位的结果映射；没有另写一套诱导流公式来代替 PyTornado。求解器加载失败、矩阵或面板出现非法值、线性求解失败时直接报错，不自动退回 CFD 积分或常数分布。Windows 原生函数使用的整数数组按 `ctypes.c_long` 的实际类型创建，避免把 NumPy 2 默认的 64 位整数直接传给需要 32 位 C `long` 的接口。

## 本机安装包与构建记录

实际安装位置为 `C:/Users/fff/miniconda3/envs/flogen/Lib/site-packages/pytornado`。
安装包来自 [官方 PyTornado](https://github.com/airinnova/pytornado) 的 PyPI 0.5.0 源码，
原始五个 C++ 求解文件及头文件的逐字一致性已用 SHA256 核对，Windows 构建使用该环境中的 Zig 0.13。

| 本地文件 | 用途 |
|---|---|
| [PyTornado wheel](build/pytornado-wheels/pytornado-0.5.0-cp312-cp312-win_amd64.whl) | 已安装的 Windows / Python 3.12 原生包 |
| [airfoils 兼容 wheel](build/pytornado-wheels/airfoils-0.2.2-py3-none-any.whl) | 保留旧 SciPy 默认三点中心差分的计算顺序，修复被移除的 derivative 导入 |
| [PyTornado 构建记录](build/pytornado-build-info.json) | 官方源包、wheel 及原始 C++ 文件 SHA256 |
| [PyTornado 兼容清单](build/pytornado-build-compatibility.json) | matplotlib 约束、C long 整数数组和 collections.abc 修补记录 |
| [airfoils 兼容清单](build/airfoils-scipy-compatibility.json) | SciPy 导数接口的局部兼容修补及验证 |
| [Windows 构建脚本](build/build_pytornado_windows.py) | 从已下载源码重建本机 wheel 的脚本 |

环境运行依赖包括 NumPy 2.5.4、SciPy 1.18.1、Matplotlib 3.11.2、commonlibs 0.3.4、
airfoils 0.2.2、aeroframe 0.1.2、ambiance 1.3.1；兼容 wheel 仍保留原许可证。
直接重新安装未经修补的旧 PyPI 发布包会重新遇到旧依赖问题，不能将它等同于上述本机已验证构建。

```powershell
conda activate flogen
python -c "import pytornado.aero.vlm; import pytornado.aero.c_vlm; print('PyTornado ready')"
python -m pip check
python -B superwing_io.py inspect --row 0
```

服务器当前加载项目内 `.agent_runs/superwing_flogen_runtime/deps` 中的官方 CPython 3.11 原生扩展，使用 `/home/cvlab/miniconda3/envs/flogen/bin/python`，所有任务缓存均位于 `Projects/floGen`。完整服务器配置见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。

## 从真实表面几何到弯度面

适配器接收单个机翼、不含样本轴的 `centers=(3,128,256)` 和 `vertices=(3,129,257)`，先通过当前几何校验确认坐标有限、周向闭合、中心为四顶点平均以及展向站位递增。几何仍以参考文件为权威，不能只用少量平面参数生成与真实翼型无关的矩形替代。

每个展向顶点截面先识别前缘和平均尾缘，建立局部弦向及法向坐标。沿上下表面分别按相同的局部 x/c 插值，再将两侧局部高度取平均，得到弯度面；这样保留该参考翼的弯度、扭转、后掠及上反。若尾缘封口包含中间点，定位平均尾缘时排除封口中点造成的额外上表面点。弦向面板边界采用余弦分布，展向面板边界在实际暴露翼面范围内按物理展向坐标等距布置。

| 原参考坐标 | PyTornado 坐标 | 含义 |
|---|---|---|
| x | x | 主要沿弦向或流向 |
| z | y | 展向；减去原参考暴露翼根的 z_min 后建立镜像平面 |
| y | z | 厚度、弯度、上反所在的高度方向 |

当前将暴露翼根 `z=z_min` 平面作为对称镜像平面，生成另一半翼，并调整镜像面板角点顺序以保持一致方向。**这是暴露半翼的镜像假设，不包含未交付的中心翼段、机身及其干扰效应**；不能将它描述为完整飞机或中心线翼根模型。

## 网格、截面 CL 和物理范围

默认每半翼使用 32 个展向面板和 4 个弦向面板，两侧合计 256 个面板。官方求解器采用无量纲等价的来流密度与速度均为 1，因此动压为 0.5。将一半翼的面板力按攻角旋转成升力，再沿弦向求和为各展向条带升力，除以动压、当地弦长和对应条带展向宽度，得到无量纲截面 CL。

原生 VLM 面板站位与参考网格的 128 个单元中心不同，输出按真实物理展向坐标进行线性插值，暴露翼根取对称条带值、自由翼尖令循环趋零。结果统一为 C 连续 `float64` 的 `(1,128)` 数组，保持从内翼到外翼的顺序；没有以数组下标盲目拉伸。输出是后掠理论变换之前的截面 CL，不预先除以后掠余弦平方，也不做网络 min/max 归一化。

此处是不可压缩的势流马蹄涡格估计。Mach 在接口中接受并记录为来源信息，目前不参与压缩性修正，未添加 Prandtl–Glauert 或其他经验 Mach 修正；Re 也不进入本势流求解。VLM 不能解析跨声速激波、黏性、分离和完整厚度效应，所得结果属于低保真估计。`sld_details.total_cl` 采用由弯度面弦长条带计算的参考面积归一化，不能在未核对参考面积与模型范围时直接等同于原 CFD 索引中的整翼系数。

## Python 接口与几何独立使用

```python
import numpy as np
from low_fidelity_sld import estimate_spanwise_lift

# 只读几何；不打开 data.npy 或接受 CFD 场参数。
centers = np.load("D:/data/foilpara/foilgeo/geom0.npy",
                  mmap_mode="r", allow_pickle=False)
vertices = np.load("D:/data/foilpara/foilgeo/origingeom.npy",
                   mmap_mode="r", allow_pickle=False)
sectional_cl, details = estimate_spanwise_lift(
    centers[0], vertices[0], mach=0.7519974134, aoa_deg=9.3724910087,
    chord_panels=4, span_panels=32,
)
print(sectional_cl.shape)  # (1,128)
print(details)
```

网格参数的当前允许范围为：`chord_panels` 是 1～16 的整数，`span_panels` 是 4～128 的整数，布尔值不当作整数；Mach 必须有限且在 `[0,1)`，攻角必须有限。输出数组和来源对象由调用者保存或交付，以上示例不自动覆盖目录中的 SLD。

## 读取器与 CLI 的三种来源

| 选择 | 调用方式 | `sld_source` | 是否使用 CFD 场生成 SLD |
|---|---|---|---|
| 默认 VLM | `reader.read(row)` 或 `--sld-mode vlm` | `pytornado_vlm` | 否 |
| 调用者提供 | `reader.read(row,sld=array)` 或 `--sld-file one_case.npy` | `external_provided` | 读取器不再估计；来源物理方法由调用者负责 |
| 显式参考积分 | `reader.read(row,sld_mode="cfd-reference")` 或 `--sld-mode cfd-reference` | `cfd_reference_integral` | 是，只能作为参考诊断 |

`reader.iter_cases` 同样默认 `sld_mode="vlm"`。调用者提供数组时优先直接使用，不再执行 VLM 或参考积分。`source_case.json` 中默认 `sld_requires_cfd=false`、`reference_sld_uses_cfd=false`；显式参考积分时这两项为 true。VLM 的 `sld_details` 包括 solver/version/backend、弦向和展向面板数、总面板数、Mach/攻角、弯度面定义、镜像平面、坐标映射、插值方式、归一化面积和适用限制。

```powershell
& 'C:/Users/fff/miniconda3/envs/flogen/python.exe' -B superwing_io.py inspect --row 0 --source-root D:/Superwing
```

该命令打印参考 case 及默认 VLM 来源，不写生产文件。正式导出仍执行单 case 冲突预检，已有数据不会被此默认模式悄悄覆盖；完整交付命令见 [SUPERWING_IO.md](SUPERWING_IO.md)。网络 `predict-foilpara` 继续读取已经准备好的 `foilsld`，三通道输出的 `Cp、150×Cf_tau、300×Cf_z` 定义不变。

## 当前三维训练中的使用

`prepare_superwing_3d_training.py` 对选中的每条 SuperWing 工况计算这套相同 VLM，并存入 `sld.npy`；随后通过原后掠理论和冻结二维网络生成 `prior.npy`。三维训练只更新 residual，缓存文件及二维权重通过 SHA-256 绑定。它没有训练 SLD FNN，也没有在训练期间改换 VLM 网格、压缩性模型或插值方法。

单算例纯预测仍读取已经准备好的 `foilsld`；读取、交付和训练缓存是三个调用场景，求解算法都在同一个 `low_fidelity_sld.py` 中。当前完整三维检查点记录 `sld_source=pytornado_vlm`、`sld_network_used=False` 和 `sld_uses_cfd=False`。作者三种方案——纯 VLM、FNN 直接预测、FNN 预测 `CFD_CL−VLM_CL` 后加回 VLM——均保留，来源与 `ref_type=1/2/3` 对应关系见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。当前只使用纯 VLM；不能把参考 CFD 积分或 FNN 输出误标为 PyTornado，也不把保留的 FNN 算法当作已有训练权重。

## 数值回归与范围

[tests/test_low_fidelity_sld.py](tests/test_low_fidelity_sld.py) 使用真实官方原生后端检查以下行为，不用模拟输出替代 VLM：

| 检查 | 需要保持的行为 |
|---|---|
| 平直对称翼、零攻角 | 升力接近零 |
| 规则测试翼、正攻角 | 升力为正；细化展向面板时结果收敛 |
| 不存在 CFD 场文件 | 仅几何和工况仍能计算 SLD |
| 改写临时副本中的参考 CFD 场 | 几何/工况相同则 VLM 结果不变 |
| 来源元数据 | 记录官方后端、网格、坐标映射和 CFD 隔离标志 |

```powershell
conda activate flogen
python -B -m unittest discover -s tests -p test_low_fidelity_sld.py -v
```

规则测试翼的参考收敛记录中，4° 攻角、每半翼 16/32/64 个展向面板时，面积加权总 CL 约为 0.26312 / 0.26430 / 0.26482；这些数值说明该测试几何的势流计算行为，不代表 SuperWing 跨声速黏性 CFD 的误差水平。

原生求解链路和单算例交付正确，不等同于完整高层 PyTornado 飞机 JSON API 已完成验收，也不等同于网络预测在任意几何/工况下都准确。生产文件的实际来源应读取各 case 的 `source_case.json`，不依赖文档中某次历史导出的临时目录或 export_id。
