# floGen 增量训练文件接口

入口：python -B bridge.py --job <绝对路径job.json>。

bridge.py 独立复用 main.py 的 SuperWingData、PhysicsEmbeddedWingPredictor、v2 检查点和 predict_directory，不改变原训练命令。协议为 `wing-loop-v1`，operation 支持 `train_predict` 与 `predict`。

## 与当前主流程的兼容边界

该文件描述保留的旧增量训练和独立几何协议，其形状/manifest 契约保持不变；它不是当前 `foilpara/{foilgeo,foiloc,foilsld}` 的单算例入口。

当前三维权重使用冻结二维网络加 PyTornado 外部 SLD，记录 `sld_network_used=False`、`sld_training_steps=0`。`bridge.py` 的两个 operation 都要求已训练的 FNN-SLD 前序，因此**不能直接加载当前 VLM 三维权重执行桥接**。当前权重的正式纯预测使用 `main.py predict-foilpara`，见 [FOILPARA.md](FOILPARA.md)；当前 200 轮训练使用专用训练器，见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。

下面的训练/预测规则适用于满足旧协议前序和数据要求的检查点。文件协议和代码功能在文档整理中没有修改。

## 接续语义

必须输入已训练的 wing 检查点，并具备 airfoil_training_steps、sld_training_steps、历史 train_shape_ids 与 sld_train_shape_ids。首次桥接原 main.py 检查点时保留权重并新建 Adam，记录 bridge_optimizer_reset=true；桥接检查点则恢复 Adam 状态，并将本轮学习率设成 plan.learning_rate。每轮恰好执行 plan.steps 个优化步。device 接受 cpu、cuda 和 cuda:0 等非负 GPU 编号；seed 限定为 0..2^32-1。可选 checkpoint_in_sha256 将任务绑定到指定祖先检查点，读取模型前先验证实际文件哈希。当前实现保留最终一步权重及同一步 Adam，不按验证结果退回最佳权重，让验证退化可真实反馈给动态训练策略。

不重新拟合归一化范围；全部 FeatureNorm buffer 和二维/SLD 网络在训练前后进行哈希验证。只有三维 residual 参数被优化。原始 Cp、150Cf_tau、300Cf_z 数据约定保持不变。

## 固定验证和数据隔离

validation 对象含 root、rows 和可空 expected_fingerprint。首轮计算指纹，后续必须与 job 指定值和输入 checkpoint 内已有值一致。指纹涵盖所选原始行号、index、configs、geom0、origingeom、CFD 真值及实际 Re；不依赖目录绝对位置。训练后再次检查验证指纹和输入 checkpoint 哈希，防止执行中来源改变。

本轮训练形状、历史 wing 训练形状、SLD 训练形状均不得出现在验证形状中。train 是多个原数据选择的列表，不合并或改写原数组。所有重复 (原始形状ID, 工况ID) 都会被拒绝，包括内容一致的重复，以防意外加权。

Re 只从各 selection 的显式 reynolds、显式 reynolds_file 或原始三维日志读取；二维默认 2e7 不会覆盖三维工况。

## 幂等和输出

result_path 的同名 .started.json 由 x 模式独占创建。已完成且 request_sha256 与原 job 字节相同的任务直接返回完成结果；未完成 claim、失败结果或不同请求均不得自动重新训练。人工处理故障后应创建新 job_id 和新输出路径。

检查点先写同目录临时文件，在预测完成后通过不覆盖已有文件的原子发布方式生成 checkpoint_out。预测复用 main.py 的完整 SuperWing 导出，训练/验证数据与旧检查点只读；result JSON 最后发布，status=completed 才代表整项任务完成。失败会保留 claim 和错误信息，不伪造反馈。因停机或失败遗留的临时检查点和预测目录用于人工排查，不自动清理或覆写。

feedback 精确匹配 D:/dspy/design.py 的 TrainingFeedback：round_id、validation_error_before、validation_error_after、metrics、notes。before/after 是同固定验证集、同原检查点归一化范围的 MSE；metrics 中含原存储单位的三通道 RMSE、优化步数、学习率和 optimizer_reset，所有值均为 float。

## 小数据真实测试

运行 python -B -m unittest discover -s tests -p test_bridge.py -v。

tests/test_bridge.py 的 bootstrap_fixture(base) 可供跨项目测试复用：在调用者给出的临时目录里，创建 4 样本原格式数据并让二维、SLD、三维模型各实际优化 1 步，返回 cfd_root、checkpoint、train_rows=[0,1]、validation_rows=[2,3]。训练形状为 101，验证形状为 509。job_fixture 可构造共同协议的任务。

回归覆盖 1 步训练及下一轮接续 2 步、固定归一化与先验、真实预测导出、源数据未变、Re/真值指纹、形状泄漏、重复样本、失败反馈和 claim 防重跑。测试只使用临时合成数组，未运行完整训练。


## 独立几何预测分支

operation=predict 与 train_predict 共存。纯预测只接收已经训练的 checkpoint 和 D:/data/foilgeo/<round>/geometry 的几何工况，不等待 CFD，不读取任何 data.npy，不创建优化器，也不产生 training feedback。每轮新几何可以同时交付服务器 CFD 分支和此预测分支。

geometry.root 必须包含 configs.dat（55 列）、index.npy（N×6；也兼容 N×12 但只读前六列）、geom0.npy（G×3×128×256）、origingeom.npy（G×3×129×257）和 geometry_manifest.json。index 前六列仍是原 SuperWing 含义：局部形状下标、工况标签、攻角、Mach、参考半面积和半展长。

manifest 的 schema_version=wing-geometry-v1，files 映射记录四个原始几何文件的 SHA256。实际 Re 使用 manifest.reynolds 正数，或 manifest.reynolds_file 的本目录 NPY 文件（须在 files 中声明），或 files 中声明的 running_info.jsonl。manifest 全部文件均进行哈希校验。共享 geometry_fingerprint 严格等于原 manifest 文件字节的 SHA256，与 geometry.expected_fingerprint 对齐。可选 job.geometry 的 Re 参数只能验证 manifest，不能覆盖共享几何的工况。

纯预测 job 不含 train/validation/plan/checkpoint_out；使用 checkpoint_in、可选 checkpoint_in_sha256、geometry={root,expected_fingerprint}、prediction={output_root}、batch_size、device、seed、result_path，以及通用 schema_version/job_id/round_id。纯预测 completed 结果包含 operation=predict、geometry_fingerprint、checkpoint_sha256、prediction_dir、steps_performed=0、normalization_unchanged=true，无 feedback。

输出 data.npy 沿用三通道原倍率，index.npy 固定为 N×6，绝不伪造 CFD 升阻力或力矩系数。prediction_info.json 声明 index_layout=geometry_conditions_v1、index_coefficients_role=unavailable_not_measured_not_predicted、stored_multipliers=[1,150,300]、共享 geometry_fingerprint。A/B 只应对带此标记的预测目录接受六列 index，系数字典应为空；标准 CFD/数据库仍为原十二列接口。

原输入 manifest 以 source_geometry_manifest.json 复制到预测目录，并由 prediction_info.source_geometry_manifest 指明它描述的是来源。这样即便兼容输入从十二列 index 截成六列输出，也不会把原 manifest 冒充输出文件清单。完整发布前后还会复核来源 manifest/各文件及输入模型未变。

新增独立预测测试仅调用一次短 bootstrap，再在完全没有 data.npy 的几何目录上执行真实模型前向；测试禁止创建 Adam/拟合归一化/打开名为 data.npy 的源文件，检查所有预测有限、Nx6元数据和共享指纹，并验证几何篡改在加载模型前即被拒绝。桥接回归使用临时模型与临时数据，与当前运行中的 200 轮训练无关。
