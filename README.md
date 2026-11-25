# Assignment 3：网络入侵检测分类任务

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)

本作业要求独立实现 XGBoost 风格的梯度提升树（GBDT），根据网络连接特征区分正常流量和攻击流量，并通过实验比较优化前后的训练和推理速度。

## 1. 任务概述与作业要求

| 数据文件 | 样本数量 | 标签 | 用途 |
|---|---:|---|---|
| `train.csv` | 311874 | 包含 `label` | 模型训练与训练集内验证 |
| `test_public.csv` | 40000 | 包含 `label` | 本地自评与方案比较 |

1. **模型实现。** 独立实现二阶梯度提升树，包括梯度和二阶导数的计算、分裂增益、带正则的叶节点权重，以及多棵树的训练与预测。
2. **类别特征处理。** 处理三个类别特征，并在报告中说明编码方式。
3. **精度与速度。** 以 ROC-AUC ≥ 0.98 为效果门槛。在相同数据和运行环境下比较优化前后的 ROC-AUC、训练时间、推理时间和 Latency，并说明优化带来的变化。

模型核心算法可使用 NumPy、pandas 和 Python 标准库。禁止直接导入或调用以下机器学习库及其现成模型：

`sklearn` / `scikit-learn`、`tensorflow`、`torch` / `pytorch`、`keras`、`xgboost`、`lightgbm`、`catboost`、`statsmodels`、`libsvm`、`liblinear`、`thundersvm`、`cuml`、`shotgun`。

`analysis.py` 可使用 `matplotlib` 等分析和可视化库。实验应使用提交的模型。

## 2. 数据格式说明

两个 CSV 文件均包含以下字段：

| 字段 | 含义 |
|---|---|
| `id` | 记录编号，不作为模型特征 |
| 41 列输入特征 | `protocol_type`、`service`、`flag` 为类别特征，其余为数值特征 |
| `label` | 二分类标签，0 表示正常流量，1 表示攻击流量 |

## 3. 评测指标与评分方式

### 3.1 评测指标

| 指标 | 含义 |
|---|---|
| ROC-AUC | 根据正类概率衡量排序质量 |
| Accuracy | 分类准确率 |
| Training Time | `fit` 的运行时间 |
| Testing Time | 10 个线程调用 `forward` 完成全部预测的时间 |
| Latency | √(训练时间 × 推理时间)，越低越好 |

### 3.2 评分方式

总分为 100 分。模型效果占 80 分，速度优化实验占 20 分。

参考基线成绩：ROC-AUC 0.99994，Latency 1.11 秒（`test_public.csv` 全量）。本地耗时会随设备和运行环境变化。

## 4. 项目结构

```text
ML2026-Assignment-3/
├── train.csv          # 训练数据
├── test_public.csv    # 公开验证数据
├── model.py           # 模型算法接口，需修改
├── solution.py        # 训练与推理接口，需修改
├── evaluate.py        # 本地评测脚本
├── requirements.txt  # 实验环境依赖
└── README.md          # 作业说明
```

提交时还需提供自行编写的 `analysis.py`、`report.md` 和评测生成的 `result.json`。

## 5. 模型实现与接口规范

### 5.1 模型训练

评测器调用 `Solution.fit(X_train, y_train)`。`X_train` 是包含 41 个特征的 pandas DataFrame，`y_train` 是 NumPy 标签数组。在 `fit` 中完成特征处理和模型训练，并在内存中保留推理所需的状态。

模型训练仅使用 `train.csv`。公开验证标签可用于自评和误差分析，不得并入训练数据。不得读取其他测试文件、联网获取额外数据、加载外部训练产物或硬编码答案。

### 5.2 单样本推理接口

评测器调用 `Solution.forward(sample)`。输入为包含 41 个特征的字典，返回 `{"prediction": 0 或 1, "probability": 正类概率}`。正类概率须为 0 至 1 之间的有限数值。

评测器使用 10 个线程并发调用同一个 `Solution` 实例。请在训练阶段完成模型准备，保持训练与推理的特征处理方式一致，避免在 `forward` 中修改共享状态。

教师运行环境提供 NumPy、pandas 和 Python 标准库。运行所需源码应放在 `solution.py` 和 `model.py` 中。教师每次评测的预算为 1800 秒，包含训练和推理，内存上限为 4 GiB。

## 6. 环境配置

建议使用 Python 3.11。

```bash
conda create -n ML2026 python=3.11
conda activate ML2026
pip install -r requirements.txt
```

## 7. 本地评测

在项目根目录执行：

```bash
# 完整评测公开验证集
python evaluate.py

# 快速检查前 5000 行
python evaluate.py --rows 5000

# 在结果文件中记录学号和姓名
python evaluate.py --student-id 20260001 --name 张三
```

评测脚本输出 ROC-AUC、Accuracy、训练时间、推理时间和 Latency，并生成 `result.json`。接口模板的固定预测仅用于检查评测能否运行。修改源码后，提交前须重新生成完整自评结果。

## 8. 提交要求

| 文件 | 提交内容 |
|---|---|
| `solution.py` | 训练入口与单样本推理接口 |
| `model.py` | 梯度提升树核心算法 |
| `analysis.py` | 速度优化对照实验代码 |
| `report.md` | 算法与类别编码说明、优化前后对照表、结论及 AI 工具使用声明 |
| `result.json` | 最终源码的完整自评结果 |

请固定实验中的随机种子，注明计时环境，确保实验结果与提交代码对应。

## 9. 学术诚信与参考资料

可以查阅 XGBoost 论文和算法资料，请在报告中注明来源。使用 AI 工具时，请说明辅助范围。

复制历年解答、获取或硬编码测试标签、探测教师评测数据、伪造实验结果等行为，将按课程规则核查处理。
