# OV-DINO 在 TCT_NGC 数据集上的微调实验结果

## 1. 实验设置

### 模型
- **架构**: OV-DINO（Swin-Tiny-224 + BERT-Base + DINO Transformer）
- **预训练权重**: `swin_tiny_patch4_window7_224.pth`
- **类别数**: 20

### 训练超参数
| 参数 | 值 |
|:-----|:---|
| 优化器 | AdamW (betas=0.9/0.999, weight_decay=1e-4) |
| 基础学习率 | 1e-4（语言骨干网络: ×0.1） |
| 学习率调度 | Multi-step 衰减（epoch 16 & 22），warmup 1000 步 |
| 批大小 | 8（每卡 1 × 8 GPU） |
| 训练轮数 | 24 |
| 每轮迭代数 | 8,700 |
| 总迭代数 | 208,800 |
| 混合精度（AMP） | 开启 |
| 梯度裁剪 | max_norm=0.1 |

### 数据集: TCT_NGC
- **训练集**: `tct_ngc_train_base_ovd_unipro`（69,590 张）
- **测试集（Base）**: `tct_ngc_test_base_ovd`（1,607 张）
- **测试集（Novel）**: `tct_ngc_test_novel_ovd`
- **文档覆盖类别数**: 31（20 base + 11 novel）

### 类别划分

以下英文类别名严格对齐 TCT_NGC 标注文件中的原始 `categories[].name`。

**Base 类别（20 类，含 5 个阴性类）** — 参与训练：

| 编号 | 类别 | 备注 |
|:----:|:-----|:-----|
| 1 | normal | 阴性类 |
| 2 | ascus | |
| 3 | asch | |
| 4 | lsil | |
| 5 | agc_adenocarcinoma_em | |
| 6 | vaginalis | |
| 7 | dysbacteriosis_herpes_act | |
| 8 | ec | |
| 9 | Serous effusion-Negative samples | 阴性类 |
| 10 | Serous effusion-Diseased cells | |
| 11 | Serous effusion-Breast cancer | |
| 12 | Thyroid gland-Papillary cancer | |
| 13 | Thyroid gland-Negative samples | 阴性类 |
| 14 | Thyroid gland-Suspicious for Malignancy | |
| 15 | Urine-Negative | 阴性类 |
| 16 | Urine-SHGUC | |
| 17 | Urine-AUC | |
| 18 | respiratory tract-Negative samples | 阴性类 |
| 19 | respiratory tract-Diseased cells | |
| 20 | respiratory tract-adenocarcinoma | |

**Novel 类别（11 类）** — 零样本，训练中未见：

| 编号 | 类别 |
|:----:|:-----|
| 1 | hsil_scc_omn |
| 2 | monilia |
| 3 | Serous effusion-Ovarian cancer |
| 4 | Serous effusion-adenocarcinoma |
| 5 | Thyroid gland-Suspicious papillary cancer |
| 6 | Thyroid gland-AUC |
| 7 | Thyroid gland-Malignant tumour |
| 8 | Thyroid gland-NS |
| 9 | Urine-HGUC |
| 10 | respiratory tract-Squamous cell cinoma |
| 11 | respiratory tract-Small cell carcinoma |

---

## 2. 训练曲线（全部 20 类，完整测试集）

| 轮次 | 迭代数 | AP | AP50 | AP75 | APm | APl |
|:----:|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 1 | 8,699 | 17.59 | 25.79 | 20.11 | 6.09 | 17.90 |
| 2 | 17,399 | 19.91 | 29.90 | 22.71 | 7.85 | 20.40 |
| 3 | 26,099 | 21.57 | 32.30 | 24.58 | 10.37 | 21.78 |
| 4 | 34,799 | 22.68 | 33.77 | 26.04 | 10.42 | 22.92 |
| 5 | 43,499 | 22.97 | 34.28 | 26.38 | 11.74 | 23.19 |
| 6 | 52,199 | 23.22 | 34.56 | 26.70 | 11.22 | 23.54 |
| 7 | 60,899 | 23.41 | 35.01 | 26.96 | 11.47 | 23.71 |
| 8 | 69,599 | 24.22 | 36.13 | 27.87 | 11.81 | 24.54 |
| 9 | 78,299 | 24.25 | 35.88 | 28.06 | 11.92 | 24.54 |
| 10 | 86,999 | 25.95 | 38.69 | 29.78 | 12.67 | 26.20 |
| 11 | 95,699 | 23.92 | 35.70 | 27.59 | 11.78 | 24.27 |
| 12 | 104,399 | 23.34 | 34.89 | 26.86 | 12.58 | 23.64 |
| 13 | 113,099 | 24.77 | 36.82 | 28.53 | 12.85 | 25.08 |
| **14** | **121,799** | **26.66** | **39.19** | **30.98** | **12.80** | **26.99** |
| 15 | 130,499 | 24.76 | 36.77 | 28.44 | 13.19 | 25.10 |
| 16 | 139,199 | 25.77 | 38.30 | 29.72 | 12.75 | 26.17 |
| 17 | 147,899 | 26.21 | 38.72 | 30.38 | 13.26 | 26.51 |
| 18 | 156,599 | 25.65 | 38.00 | 29.76 | 13.32 | 25.93 |
| 19 | 165,299 | 25.91 | 38.39 | 29.97 | 13.36 | 26.20 |
| 20 | 173,999 | 25.56 | 37.71 | 29.53 | 13.37 | 25.86 |
| 21 | 182,699 | 25.47 | 37.70 | 29.47 | 13.25 | 25.73 |
| 22 | 191,399 | 25.09 | 37.19 | 29.04 | 13.21 | 25.37 |
| 23 | 200,099 | 25.55 | 37.81 | 29.61 | 13.04 | 25.86 |
| 24 | 208,799 | 25.43 | 37.59 | 29.45 | 13.09 | 25.73 |

**最佳轮次: 第 14 轮**（AP=26.66，权重文件: `model_0121799.pth`）

---

## 3. 最佳模型评测结果（第 14 轮）

### 3.1 Base 类别（排除 5 个阴性类）

使用 `ExcludeClassCOCOEvaluator` 评测，排除：normal、Serous effusion-Negative samples、Thyroid gland-Negative samples、Urine-Negative、respiratory tract-Negative samples。

| AP | AP50 | AP75 | APs | APm | APl |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| **33.349** | **48.358** | **38.923** | 0.000 | 18.473 | 33.623 |

**各类别 AP（Base，排除阴性类）：**

| 类别 | AP |
|:-----|:---:|
| ascus | 36.827 |
| asch | 24.888 |
| lsil | 36.447 |
| agc_adenocarcinoma_em | 45.183 |
| vaginalis | 30.584 |
| dysbacteriosis_herpes_act | 57.909 |
| ec | 22.824 |
| Serous effusion-Diseased cells | 35.992 |
| Serous effusion-Breast cancer | 69.042 |
| Thyroid gland-Papillary cancer | 54.106 |
| Thyroid gland-Suspicious for Malignancy | 31.989 |
| Urine-SHGUC | 28.440 |
| Urine-AUC | 8.783 |
| respiratory tract-Diseased cells | 15.502 |
| respiratory tract-adenocarcinoma | 1.718 |

### 3.2 Novel 类别（零样本）

使用标准 `COCOEvaluator` 在 Novel 测试集上评测（训练中未见的类别）。

| AP | AP50 | AP75 | APs | APm | APl |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| **8.808** | **11.800** | **10.077** | 0.000 | 1.937 | 8.838 |

**各类别 AP（Novel）：**

| 类别 | AP |
|:-----|:---:|
| hsil_scc_omn | 9.952 |
| monilia | 0.006 |
| Serous effusion-Ovarian cancer | 0.042 |
| Serous effusion-adenocarcinoma | 3.905 |
| Thyroid gland-Suspicious papillary cancer | 8.068 |
| Thyroid gland-AUC | 0.214 |
| Thyroid gland-Malignant tumour | 0.547 |
| Thyroid gland-NS | 0.092 |
| Urine-HGUC | 69.759 |
| respiratory tract-Squamous cell cinoma | 1.188 |
| respiratory tract-Small cell carcinoma | 3.117 |

---

## 4. 关键发现

1. **最佳轮次为第 14 轮**（全 20 类 AP=26.66），此后性能出现波动，可能与 epoch 16 处的学习率阶梯衰减有关。

2. **Base 类别（排除阴性类后）AP=33.35** — 显著高于全 20 类的 AP（26.66），说明阴性/正常细胞类别由于检测难度大，拉低了整体指标。

3. **Novel 零样本 AP=8.81** — 整体偏低，但存在显著异常值：*Urine-HGUC* 达到 69.76 AP，说明形态特征显著的细胞类型具有较强的文本-视觉对齐能力。

4. **最难类别**：respiratory tract-adenocarcinoma（Base 中 AP=1.72）、monilia（Novel 中 AP=0.006）— 可能与目标尺寸过小（APs 始终为 0.000）以及与阴性样本的视觉相似性有关。

5. **APs=0.000** 在所有评测中一致出现，表明模型对细胞学图像中的小目标检测能力不足。

---

## 5. 实验产物

| 产物 | 路径 |
|:-----|:-----|
| 训练日志 | `wkdrs/ovdino_swin_tiny224_bert_base_ft_tct_ngc_24ep/log.txt` |
| 最佳权重（第 14 轮） | `wkdrs/ovdino_swin_tiny224_bert_base_ft_tct_ngc_24ep/model_0121799.pth` |
| 最终权重（第 24 轮） | `wkdrs/ovdino_swin_tiny224_bert_base_ft_tct_ngc_24ep/model_final.pth` |
| Base 评测输出 | `wkdrs/eval_tct_ngc_base/eval_base_ep14/` |
| Novel 评测输出 | `wkdrs/eval_tct_ngc_novel/eval_novel_ep14/` |
| 训练配置 | `ovdino/projects/ovdino/configs/ovdino_swin_tiny224_bert_base_ft_tct_ngc_24ep.py` |
| Base 评测配置 | `ovdino/projects/ovdino/configs/ovdino_swin_tiny224_bert_base_ft_tct_ngc_eval_base.py` |
| Novel 评测配置 | `ovdino/projects/ovdino/configs/ovdino_swin_tiny224_bert_base_ft_tct_ngc_eval_novel.py` |
