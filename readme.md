# FloGen：物理嵌入式机翼表面预测

本项目使用 SuperFoil 训练二维翼型模型，将冻结的二维预测与后掠理论、官方 PyTornado 低保真展向升力估计结合，再由三维残差网络预测机翼表面的 Cp、轮廓切向摩擦和展向摩擦。模型、原始数据适配和单算例文件接口的实现均位于当前工作区。

## 使用入口

| 任务 | 代码或文档 |
|---|---|
| 二维权重加载、输入输出和训练命令 | [二维神经网络使用说明](二维神经网络使用说明.md) |
| 二维网络结构、数据划分与 60 轮权重记录 | [MODEL_2D.md](MODEL_2D.md) |
| 三维网络、后掠变换、实际训练配置与检查点 | [IMPLEMENTATION.md](IMPLEMENTATION.md) |
| 正式单算例输入与网络输出协议 | [FOILPARA.md](FOILPARA.md) |
| 项目目录、运行材料与保留依赖 | [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) |
| 从 SuperWing 读取一条参考 case 并交付输入/CFD | [SUPERWING_IO.md](SUPERWING_IO.md) |
| 官方 PyTornado SLD 算法、环境和适用范围 | [PYTORNADO.md](PYTORNADO.md) |
| 原有增量训练/几何预测协议及兼容边界 | [BRIDGE.md](BRIDGE.md) |

当前二维权重为 [weights/2d/latest.pt](weights/2d/latest.pt)，已完成 60 轮训练。当前三维任务配置为 [build/superwing_200_bs8_config.json](build/superwing_200_bs8_config.json)：训练 batch size 8，test batch size 4，目标 200 轮，每轮 test，每 10 轮下载一个检查点到本机。训练完成情况应以任务状态文件为准，不能仅凭配置判断已经完成 200 轮。

正式推理命令为 `python -B main.py predict-foilpara --checkpoint <已训练的三维权重>`，默认从 `D:/data/foilpara/{foilgeo,foiloc,foilsld}` 读取单算例，输出到 `D:/data/foilperf/network`。当前 VLM 三维检查点要求外部 SLD；详细形状、单位和目录冲突规则见 FOILPARA 文档。

## 项目边界

当前本机数据根目录为 `D:/SuperFoil` 和 `D:/SuperWing`，源数组保持只读。服务器使用 Conda `flogen`，所有任务代码、数据、缓存、日志和输出均位于 `/home/cvlab/Projects/floGen`。服务器训练由本机 PyCharm 中的 subprocess/SSH 监督进程启动和同步。

`readforagent` 保留项目最初要求，原文保持不变；当前训练规模和 batch 设置以之后用户明确要求及实际任务配置为准。[多智能体系统框架](多智能体系统框架.md)描述外部 A–E 协作目标，不等同于本项目已经实现全部智能体。原作者算法及底层模型库按当前功能依赖保留；其通用文档描述的历史应用，与本机当前训练入口的参数和数据协议需要区分。

项目内保留[物理嵌入论文译文](2025_AIAAJ_基于物理嵌入式迁移学习的后掠翼快速气动预测.md)，用作方法参考。当前实现适配了现有数据库网格，并采用用户指定的训练规模，未宣称复现论文全部训练实验或误差结论。论文原稿中的插图 `assets` 原本未随项目提供，原文完整保留；这些缺图不是训练或推理的运行依赖。

## 原项目、数据来源与引用

原项目通用文档：[floGen documentation](https://flogen.readthedocs.io/en/latest/)。数据来源：[SuperFoil](https://huggingface.co/datasets/yunplus/SuperFoil)、[SuperWing](https://huggingface.co/datasets/yunplus/SuperWing)。原作者 Web-wing 应用及其部署代码分别位于 [TUM Web-wing](https://webwing.pbs.cit.tum.de/) 和 [webwing repository](https://github.com/YangYunjia/webwing)。这些外部应用与本工作区的单算例接口独立。

**Author:** 

- Yunjia Yang, Tsinghua University (2020-2025), Technical University of Munich (2025-), yyj980401@126.com

**Thanks to:** 

- Runze Li
- Mengxin Liu and Weishao Tang, for wing dataset estabilishment
- Yuqi Cheng, UI for Webwing

**Citation:**

- Multipoint airfoil flow field prediction
    ```
    Yang, Yunjia, Runze Li, Yufei Zhang, and Haixin Chen*. 2022. “Flowfield Prediction of Airfoil Off-Design Conditions Based on a Modified Variational Autoencoder.” AIAA Journal 60 (10): 5805–20. https://doi.org/10.2514/1.J061972.
    ```

- Multipoint airfoil shape optimization
    ```
    Yang, Yunjia, Runze Li, Yufei Zhang, and Haixin Chen*. 2024. “Fast Buffet-Onset Prediction and Optimization Method Based on Pretrained Flowfield Prediction Model.” AIAA Journal 62 (8): 2979–95. https://doi.org/10.2514/1.J063634.

    Yang, Yunjia, Runze Li, Yufei Zhang, and Haixin Chen*. 2026. "Uncertainty-aware data-based method for fast and reliable shape optimization." Structural and Multidisciplinary Optimization 69 (4): 95. https://link.springer.com/10.1007/s00158-026-04259-0.
    ```

- Wing flow field prediction
    ```
    Yang, Yunjia, Runze Li, Yufei Zhang, Lu Lu, and Haixin Chen*. 2024. “Transferable Machine Learning Model for the Aerodynamic Prediction of Swept Wings.” Physics of Fluids 36 (7): 076105. https://doi.org/10.1063/5.0213830.

    Yang, Yunjia, Runze Li, Yufei Zhang, Lu Lu, and Haixin Chen*. 2025. "Rapid aerodynamic prediction of swept wings via physics-embedded transfer learning." AIAA Journal 63 (6): 2545-59. 

    Yang, Yunjia*, Babak Gholami, Caglar Gurbuz, Mohammad Rashed, Nils Thuerey. 2026. "Towards a Foundation-Model Paradigm for Aerodynamic Prediction in Three-dimensional Design." Accepted by AIAA Journal.
    ```

原代码许可证见 [LICENSE](LICENSE)。
