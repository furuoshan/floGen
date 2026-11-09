# foilpara 输入、神经网络输出与运行说明

此入口对应 `readforagent` 指定的数据流：机翼几何、工况和外部估计的展向升力分布输入神经网络，输出机翼表面压强和摩擦分布。二维预训练与三维流程使用独立命令；本次三维只进行前向测试，不更新参数。

## 项目结构与本次改动

| 位置 | 职责 |
|---|---|
| `flowvae/base_model/` | 原有卷积、ResNet、MLP 编解码模块 |
| `flowvae/vae.py` | 条件 U-Net 与编码器/解码器组合 |
| `main.py` | 二维模型、SLD 模型、三维残差模型、后掠理论迁移、独立训练命令和检查点 |
| `foilpara.py` | 本次新增：三目录数据适配、外部 SLD 预测导出、零优化步三维测试 |
| `bridge.py` | 已有增量训练及独立几何预测协议，继续保持兼容 |
| `tests/test_foilpara.py` | 本次新增：数据约定、外部 SLD 生效、只读和纯推理验证 |
| `examples/wing/run_kfold.py` | 旧论文交叉验证示例，本轮不执行 |

`D:/data/foilpara/foiloc`、`foilgeo`、`foilsld` 当前均为空。因此下面定义的是沿用现有 SuperWing 数组布局的拆分接口，并非从这些空目录中发现了现成文件协议。没有添加 CFD 网格生成器；几何参数须连同已经生成的表面网格提供。完整多智能体 A–E 的提示词与调度不属于此次神经网络入口改动。

## 输入文件

默认根目录为 `D:/data/foilpara`，可通过 `--input-root` 指定同格式测试目录。

```text
foilpara/
  foiloc/
    index.npy           # (N,6)，兼容原始 (N,12)
    reynolds.npy        # (N,)，逐样本的实际 Reynolds 数，例如 20000000
  foilgeo/
    configs.dat         # (G,55)，首行原格式表头
    geom0.npy           # (G,3,128,256)，单元中心坐标
    origingeom.npy      # (G,3,129,257)，网格顶点坐标
  foilsld/
    index.npy           # (N,6)，必须与 foiloc/index.npy 的前六列逐值相等
    sectional_cl.npy    # (N,128)，外部估计的无量纲截面升力系数
```

`N` 为工况样本数，`G` 为几何数；多个工况可以引用同一几何。NPY 必须是实际下载的数值数组，不能是 Git LFS 指针。输入按只读内存映射加载，不会修改数据库，也不读取 CFD `data.npy`。

工况索引前六列按原 SuperWing 顺序保留，零基下标如下：

| 列 | 含义 |
|---|---|
| 0 | 几何数组局部下标，范围 0 到 G−1；不是原始运行编号 |
| 1 | 原始工况槽位，范围 0 到 7 |
| 2 | 攻角，单位为度 |
| 3 | Mach |
| 4 | 参考半翼面积，正值 |
| 5 | 半翼展，正值 |

`configs.dat` 保持 55 列：第 0 列为原始几何编号；`1:39` 为 38 个几何参数；`39:47` 为八个工况的 Mach；`47:55` 为对应攻角。这里的切片采用 Python 结束下标不包含的约定。索引的 Mach/攻角必须与相应槽位一致。

`sectional_cl.npy` 是在 128 个展向单元中心处、应用后掠变换之前的截面 CL。展向顺序必须与几何网格的展向轴一致；不能直接传入牛顿单位的升力、总机翼 CL，或已经除以后掠角余弦平方的二维有效 CL。升力数据的 `index.npy` 用于绑定样本身份和顺序，错序时直接拒绝，不自动猜测或排序。数据提供方仍需确保每行数值与其索引、展向站位相符。

三工况二维模型需要 `reynolds.npy`：必须是实际 Re，逐样本有限且为正。不会将二维数据的默认 Re 自动套到三维。只有明确采用二维 Mach/CL 两工况检查点时，该文件可省略。

模型收到外部 `sld` 后直接用于物理迁移，绕过内部 SLD 网络；此入口缺失 SLD 会报错，不会静默回退。现有完整 SuperWing 和 `bridge.py` 入口未传外部 SLD 时继续使用原 SLD 网络。

## 输出文件

默认输出目录为 `D:/data/foilperf/network`，必须为空；禁止覆盖来源目录、其子目录或祖先目录。

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

以下命令在项目根目录运行，环境需安装 PyTorch 与 NumPy。本机已验证的解释器是 `C:/Users/fff/miniconda3/envs/3.12torch/python.exe`。

三维前向与文件接口测试，无需检查点、真实输入或 CFD：

```powershell
python -B main.py test-3d --device cpu
```

此命令在临时目录构造符合上述协议的两条合成样本，用随机初始化模型测试前向与导出，然后删除临时文件。报告明确标注 `diagnostic_untrained`，不创建优化器，不训练，不写生产目录，也不作气动精度结论。

二维预训练保持独立，可直接使用 SuperFoil 原始文件：

```powershell
python -B main.py pretrain-2d --foil-subset VT-GU --checkpoint checkpoints/airfoil_vt_gu.pt --device cpu
```

默认二维配置：300 轮、batch size 16、Adam、MSE、三工况 Mach/CL/Re；前 20 轮学习率从 `1e-5` 预热至 `1e-4`，随后按 0.95 指数衰减。默认数据根目录 `D:/Superfoil`，默认真实 Re 为已有配置 `2e7`，可用 `--reynolds` 或 `--reynolds-file` 显式覆盖。仅从训练样本拟合逐变量 min/max 范围，官方训练列表内按几何划分验证集。VT-DD 与 BUF/GU 的工况列不同，读取器已分别处理。

有充分训练并验证过的三维 v2 检查点后，执行实际预测：

```powershell
python -B main.py predict-foilpara --checkpoint checkpoints/wing.pt --device cpu
```

该命令要求已训练的 wing 检查点及其二维/SLD 前序记录，不能使用本轮生成的二维一步诊断权重代替。三维训练命令虽保留，本轮未运行。

本机 `network` 下已经存在历史目录 `training-audit-20261108`。实际交付新一轮预测时，应指定一个新的空轮次目录，例如 `--output-root D:/data/foilperf/network/round-001`；不需要清理历史结果。

新增接口回归测试：

```powershell
python -B -m unittest discover -s tests -p test_foilpara.py -v
```

原有 `smoke-test` 和部分旧测试会执行 SLD/三维优化步。因此当前“三维只测试”阶段应使用上述 `test-3d` 和新测试文件。

## 本轮实际验证与限制

- 只读检查真实 SuperWing：28,856 个工况、4,239 个几何、128×256 单元网格。真实 SuperFoil VT-GU：16,031 个样本、每个翼型 401 点。
- 在真实 VT-GU 上完成二维诊断：最多选取 32 个训练样本，独立几何验证比例 1%，batch size 16，运行一轮且最多一个优化步。学习率 `1e-5`，归一化空间训练 MSE 为 `0.2817867100`，验证 MSE 为 `0.2807653739`。这是链路检查，不能用来估计完整训练后的精度。
- 新接口的 12 项回归测试全部通过；另外运行原有 `test_main.py` 的 20 项非训练测试，全部通过。没有运行其中会更新 SLD/三维权重的两项旧测试，也没有运行完整桥接训练测试。
- `test-3d --device cpu` 实际通过：输出 `(2,3,128,256)`、索引 `(2,6)`，所有输出有限，优化步数为 0，完整模型状态和归一化状态保持不变。测试仅使用临时合成文件。
- 保存二维诊断检查点 `checkpoints/diagnostics/airfoil_vt_gu_20261109_1step.pt`。检查点目录已加入忽略规则，避免将权重纳入代码版本。
- 二维与三维的论文网格分别被现有数据的 401 点和 128×256 单元替代。三维保留已有三分支解码器，与论文附录 A2 的单解码器三通道有架构差异，未宣称严格复现论文精度。
- 没有完成二维 300 轮训练，没有训练 SLD/三维模型，没有把随机模型结果写入生产 `network` 目录。当前 `foilpara` 三个空目录需要数据提供方按上述协议填入真实输入后才能进行生产推理。
