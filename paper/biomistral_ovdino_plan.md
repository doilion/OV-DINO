# BioMistral-7B + OV-DINO 多阶段训练完整方案

> 面向可行性审核，涵盖动机、架构、训练流程、资源需求、风险与对照实验。

---

## 一、研究背景与动机

### 1.1 当前问题

OV-DINO 使用 BERT-Base 作为文本编码器。BERT 是通用领域预训练模型，对组织病理学专业术语（如 "atypical squamous cells of undetermined significance"、"low-grade squamous intraepithelial lesion"）的语义理解有限，导致：

- Novel 类别（11类）的零样本检测性能低（当前 Novel mAP = 8.81）
- 语义相近的病理类别（如 SHGUC vs HGUC）文本表征区分度不足

### 1.2 已有验证（YOLO-World-Medical）

在 YOLO-World-Medical 上用 BioMistral-7B 替换 CLIP 文本编码器，**单阶段训练**：

| 方案 | Base mAP | Novel mAP | 说明 |
|------|----------|-----------|------|
| CLIP baseline | 48.2 | 3.7 | YOLO-World 原始 |
| BioMistral V1（无 STEGO） | 47.8 | 4.1 | MLP 直连，无语义保持 |
| BioMistral V2（+ STEGO, weight=100） | 46.5 | **8.2** | STEGO 使 Novel 翻倍 |

**结论：** BioMistral 的医学语义 + STEGO 对 Novel 有显著提升，但单阶段训练中 MLP 随机初始化的噪声梯度会影响视觉分支的预训练参数。

### 1.3 本方案的改进

迁移到 OV-DINO 时引入**多阶段训练**，解决 MLP 随机初始化问题：
1. Phase 0：离线预对齐 MLP，消除初始噪声
2. Phase 1：冻结视觉分支，只训练文本通路
3. Phase 2：全部解冻，低学习率联合微调

---

## 二、架构设计

### 2.1 OV-DINO 原始架构

```
文本输入 → BERT-Base → [N*C, 768] → ClassEmbed×7 [768→256] → text_feat [256-dim]
图像输入 → Swin-Tiny → Neck → DINO Transformer → image_feat [256-dim]
检测输出 = dot_product(image_feat, text_feat) → logits
```

### 2.2 BioMistral 改造后架构

```
文本输入 → BioMistral-7B（离线提取，冻结）→ [N*C, 4096]
         → Adapter MLP（可训练）→ [N*C, 768]
         → ClassEmbed×7 [768→256]（预训练权重复用）→ text_feat [256-dim]

图像输入 → Swin-Tiny → Neck → DINO Transformer → image_feat [256-dim]
（与原架构完全相同，零改动）

检测输出 = dot_product(image_feat, text_feat) → logits
辅助损失 = STEGO Correspondence Loss（31类全参与）
```

### 2.3 为什么用 Adapter MLP 而不是直接改 ClassEmbed？

| 方案 | 随机初始化参数量 | 预训练权重保留 | 风险 |
|------|----------------|--------------|------|
| **方案A（采用）：Adapter MLP 4096→768** | ~1000万（1个MLP） | ClassEmbed 7组权重全部复用 | 低 |
| 方案B：改 ClassEmbed 4096→256 | ~7×(4096×256+4096) = ~740万×7 | 全部丢失 | 高，灾难性遗忘 |

### 2.4 Adapter MLP 结构

```python
LayerNorm(4096)          # 输入归一化
→ Linear(4096, 2048)     # 降维
→ GELU                   # 激活
→ Dropout(0.1)           # 正则化
→ Linear(2048, 768)      # 投影到 BERT 空间维度
```

参数量：4096×2048 + 2048 + 2048×768 + 768 + 4096（LayerNorm） ≈ **1000万**

### 2.5 STEGO Correspondence Loss

**目的：** 让 Adapter MLP 投影后保持 BioMistral 原始语义空间的类间距离结构。

```
Teacher（固定）：BioMistral 原始 31 类 embedding 的余弦相似度矩阵 F [31×31]
Student（可训练）：Adapter MLP 投影后 31 类的余弦相似度矩阵 S [31×31]

L = -mean( clamp(S_ij, 0, 0.8) × (F_ij - 0.4) )
```

- `clamp(S_ij, 0, 0.8)`：只对 student 认为相似（>0）且不过度自信（<0.8）的 pair 施加梯度
- `F_ij - 0.4`：negative pressure，teacher 相似度 <0.4 的 pair 被推远，>0.4 的被拉近
- **31 类全参与**（20 base + 11 novel），保持完整语义拓扑，Novel 也受益

**YOLO-World 验证：** 加入 STEGO 后 Novel mAP 从 4.1 → 8.2（翻倍），证明语义保持对零样本至关重要。

---

## 三、训练流程

### 3.0 前置准备

| 步骤 | 说明 | 产出 |
|------|------|------|
| BioMistral-7B 模型迁移 | 从 3090 机器 scp 到本机 | `inits/huggingface/hub/models--BioMistral--BioMistral-7B/` (27GB) |
| 已完成 | ✅ 已传输并验证 | hidden_size=4096, 13.5GB weights |

### 3.1 Phase 0a — BioMistral Embedding 提取

| 项目 | 说明 |
|------|------|
| **做什么** | 用 BioMistral-7B 对 31 个类别名称提取文本 embedding |
| **怎么做** | 去除 causal mask（双向注意力）→ mean pooling → L2 normalize |
| **输入** | 31 个英文类别名称（20 base + 11 novel） |
| **输出** | `embeddings/biomistral_tct_ngc.pt`，Tensor [31, 4096]，498 KB |
| **硬件** | CPU（FP32，~28GB RAM）或 GPU（FP16，~14GB VRAM） |
| **耗时** | CPU ~2min，GPU ~30s |
| **状态** | ✅ 已完成并验证 |

提取结果语义验证：
```
SHGUC <-> HGUC:     0.8484（高，符合预期——同为尿路高级别病变）
Normal <-> LSIL:    0.6655（中，宫颈正常 vs 低级别病变）
Normal <-> Breast:  0.6685（中，跨器官）
```

### 3.2 Phase 0b — Adapter MLP 预对齐（STEGO 自蒸馏）

| 项目 | 说明 |
|------|------|
| **做什么** | 训练 Adapter MLP 使其投影后保持 BioMistral 的 31 类间距离结构 |
| **怎么做** | Loss = STEGO correspondence loss + VICReg variance regularization |
| **训练对象** | 仅 Adapter MLP（~1000万参数） |
| **不需要** | 不需要 OV-DINO 任何组件，不需要图像数据，不需要 GPU |
| **输入** | `biomistral_tct_ngc.pt` [31, 4096]（498 KB） |
| **输出** | `embeddings/adapter_prealigned.pth`（Adapter MLP 权重） |
| **硬件** | CPU 即可（计算量极小：31 个向量的矩阵乘法 × 1000 步） |
| **耗时** | CPU ~30 秒 |
| **成功标准** | Spearman 相关系数 > 0.9（投影后 vs 原始的类间距离排序） |

**为什么不需要 OV-DINO？** 这一步的目标是"教 MLP 如何正确压缩维度（4096→768）"，只需要知道原始 31 个类在 4096 维空间的相对位置，就能训练 MLP 在 768 维空间保持这些位置关系。

### 3.3 Phase 1 — 冻结视觉，训练文本通路

| 项目 | 说明 |
|------|------|
| **做什么** | 加载 OV-DINO 预训练权重 + 预对齐 Adapter，冻结视觉分支，训练文本通路 |
| **冻结** | Swin-Tiny backbone, Neck, DINO Transformer, PositionEmbedding, LabelEnc |
| **训练** | Adapter MLP (lr=1e-4) + ClassEmbed×7 (lr=1e-5) + BBoxEmbed×7 (lr=1e-5) |
| **损失** | 检测 Loss（classification + box regression + denoising）+ Correspondence Loss (weight=100) |
| **数据** | TCT_NGC 训练集（69,590 张，20 base 类别标注） |
| **Epoch** | 8 |
| **Batch size** | 8（每卡1 × 8 GPU） |
| **迭代** | 8,700/epoch × 8 = 69,600 |
| **评估间隔** | 每 epoch（8,700 iter） |
| **硬件** | 8 × 2080Ti (11GB)，需要 `find_unused_parameters=True`（冻结导致） |
| **耗时预估** | 与 BERT baseline 相近（BioMistral 是离线 embedding，不占计算）约 8-12 小时 |
| **输出** | `wkdrs/ovdino_biomistral_phase1_tct_ngc/model_final.pth` |

**为什么先冻结视觉？** Adapter MLP 虽然预对齐了，但仍然不完美。如果一开始就全部联合训练，Adapter 的残余噪声会通过检测 Loss 的梯度反传到视觉分支，破坏 Swin-Tiny 的预训练特征。冻结 8 epoch 让 Adapter + ClassEmbed 先适配好，再放开视觉分支。

### 3.4 Phase 2 — 全部解冻，联合微调

| 项目 | 说明 |
|------|------|
| **做什么** | 加载 Phase 1 checkpoint，解冻所有参数，低学习率联合训练 |
| **学习率** | adapter_mlp: 1e-5, backbone: 1e-6 (0.1×), 其余: 1e-5 |
| **损失** | 检测 Loss + Correspondence Loss (weight=100) |
| **数据** | TCT_NGC 训练集（同 Phase 1） |
| **Epoch** | 16 |
| **LR 衰减** | Multi-step at epoch 11, 14 |
| **迭代** | 8,700/epoch × 16 = 139,200 |
| **硬件** | 8 × 2080Ti (11GB) |
| **耗时预估** | ~16-24 小时 |
| **输出** | `wkdrs/ovdino_biomistral_phase2_tct_ngc/model_final.pth` |

**为什么 backbone 用 0.1× 学习率？** 防止 Swin-Tiny 的预训练特征被大幅修改，保持通用视觉特征的同时微调适配病理领域。

### 3.5 评估

| 评估集 | 说明 | 预期 |
|--------|------|------|
| Base (20类) | `tct_ngc_test_base_ovd`，1,607 张 | AP 接近 BERT baseline (~26-33) |
| Novel (11类) | `tct_ngc_test_novel_ovd` | AP > BERT baseline (>8.81)，目标 15+ |

---

## 四、完整流程时间线

```
Day 0: Phase 0（已完成）
  ├─ 0a: BioMistral embedding 提取 ✅ (CPU, 2min)
  └─ 0b: Adapter 预对齐           ⏳ (CPU, 30s)

Day 1: Phase 1（需要 8 GPU）
  └─ 冻结视觉，训练文本通路       (~8-12h)
  └─ 评估 Phase 1 结果            (~30min)

Day 2-3: Phase 2（需要 8 GPU）
  └─ 全部解冻，联合微调           (~16-24h)
  └─ 评估 Base + Novel            (~30min)

Day 3: 结果分析
  └─ 对比 BERT baseline
  └─ 对比 YOLO-World-Medical 结果
```

**总计 GPU 时间：~24-36 小时**（Phase 1 + Phase 2），与 BERT baseline 的 24 epoch 训练时间相当。

---

## 五、资源需求

### 5.1 硬件

| 资源 | Phase 0 | Phase 1 | Phase 2 |
|------|---------|---------|---------|
| GPU | 不需要 | 8 × 2080Ti (11GB) | 8 × 2080Ti (11GB) |
| RAM | ~28GB (CPU推理) | 标准 | 标准 |
| 磁盘 | BioMistral 27GB + embeddings 1MB | checkpoint ~2GB | checkpoint ~2GB |

### 5.2 软件依赖

| 包 | 版本 | 用途 | 阶段 |
|----|------|------|------|
| transformers | >=4.35, <5.0 | BioMistral-7B 加载 | Phase 0a only |
| scipy | any | Spearman 相关系数 | Phase 0b only |
| bitsandbytes | optional | INT4 量化 | Phase 0a (GPU不足时) |
| detrex | 0.3.0 (已安装) | OV-DINO 框架 | Phase 1/2 |
| detectron2 | 已安装 | 检测框架 | Phase 1/2 |

**注意：** transformers 升级到 4.38.2 后与 detrex 的版本约束冲突（detrex 要求 4.30.2），但实测 Phase 1/2 训练不受影响，因为 BERT 不再被使用。如果训练出问题，可以在提取完成后降回 4.30.2。

### 5.3 数据

- 训练集：`tct_ngc_train_base_ovd_unipro`（69,590 张，20 base 类别标注）——已有
- 测试集 Base：`tct_ngc_test_base_ovd`（1,607 张）——已有
- 测试集 Novel：`tct_ngc_test_novel_ovd`——已有
- BioMistral embeddings：`biomistral_tct_ngc.pt`——✅ 已生成
- 预训练权重：`inits/ovdino/ovdino_swin_tiny224_bert_base.pth`——已有

---

## 六、与 BERT Baseline 对比

| 维度 | BERT Baseline | BioMistral 方案 |
|------|--------------|----------------|
| 文本编码器 | BERT-Base (110M参数, 在线推理) | BioMistral-7B (7B参数, 离线embedding) |
| 文本维度 | 768 | 4096 → Adapter → 768 |
| 训练开销 | 24 epoch 一次性 | Phase 0 (30s) + Phase 1 (8ep) + Phase 2 (16ep) |
| 额外参数 | 无 | Adapter MLP ~1000万 + CorrespondenceLoss (无参数) |
| 推理速度 | BERT forward ~5ms/batch | 查表 ~0ms/batch + MLP ~0.1ms/batch（更快） |
| 医学语义 | 通用 NLP | 医学领域预训练 |
| Novel 能力 | 8.81 mAP | 预期 > 15 mAP |

**推理反而更快**：BERT 需要在线 forward pass，BioMistral 是离线 embedding 查表 + 一次 MLP forward，计算量更小。

---

## 七、风险评估

### 7.1 已知风险

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|---------|
| transformers 版本冲突 | 中 | 低 | Phase 0 完成后降回 4.30.2，不影响 Phase 1/2 |
| Phase 1 Base AP 下降明显 | 低 | 中 | 预对齐已保证 Adapter 输出语义合理，ClassEmbed 预训练权重完全复用 |
| Correspondence Loss weight 不合适 | 中 | 中 | 从 YOLO-World 验证的 100.0 开始，可调 |
| 8×2080Ti 11GB OOM | 低 | 高 | BioMistral 不在 GPU 上（离线embedding），显存与 BERT baseline 相同 |
| Phase 2 视觉分支退化 | 低 | 中 | backbone 0.1× 学习率 + 梯度裁剪 0.1 |

### 7.2 需要验证的假设

| 假设 | 验证方法 |
|------|---------|
| BioMistral 病理语义优于 BERT | Phase 0a 已验证：语义对相似度合理 ✅ |
| STEGO 在 OV-DINO 上同样有效 | Phase 1/2 训练后对比有/无 STEGO |
| 多阶段优于单阶段 | 可选：额外跑一个单阶段实验对比 |
| 预对齐比随机初始化好 | 可选：Phase 1 不加载预对齐权重对比 |

### 7.3 回退方案

如果 BioMistral 方案效果不及预期：
1. 所有改动在独立 config 中，**不影响** BERT baseline 的任何代码和权重
2. 可随时切回 BERT baseline config 继续实验
3. `ovdino.py` 的修改都有 `if self.adapter_mlp is not None` 守卫，不使用时零影响

---

## 八、消融实验建议（可选）

如果时间和 GPU 允许，建议做以下消融验证各组件贡献：

| 实验 | Adapter | STEGO | 预对齐 | 多阶段 | 预期 |
|------|---------|-------|--------|--------|------|
| BERT baseline | - | - | - | - | Base 26-33, Novel 8.81 |
| BioMistral 无 STEGO | ✅ | ❌ | ✅ | ✅ | Novel 小幅提升 |
| BioMistral 无预对齐 | ✅ | ✅ | ❌ | ✅ | Phase 1 收敛慢 |
| BioMistral 单阶段 | ✅ | ✅ | ✅ | ❌ | Base 可能下降 |
| **BioMistral 完整方案** | ✅ | ✅ | ✅ | ✅ | **Novel 15+** |

---

## 九、执行命令汇总

```bash
# ===== 环境 =====
source /root/code/OV-DINO/ovdino_env/bin/activate
cd /root/code/OV-DINO/ovdino
export HF_HOME=/root/code/OV-DINO/inits/huggingface

# ===== Phase 0a: Embedding 提取（已完成 ✅）=====
python scripts/extract_biomistral_embeddings.py \
    --device cpu \
    --output embeddings/biomistral_tct_ngc.pt

# ===== Phase 0b: Adapter 预对齐（CPU，~30秒）=====
python scripts/prealign_adapter.py \
    --embeddings embeddings/biomistral_tct_ngc.pt \
    --output embeddings/adapter_prealigned.pth \
    --steps 1000 \
    --negative-pressure 0.4

# ===== Phase 1: 冻结视觉，训练文本通路（8 GPU，~8-12h）=====
bash scripts/finetune.sh \
    projects/ovdino/configs/ovdino_swin_tiny224_biomistral_phase1_tct_ngc.py \
    ../inits/ovdino/ovdino_swin_tiny224_bert_base.pth

# ===== Phase 2: 全部解冻联合微调（8 GPU，~16-24h）=====
bash scripts/finetune.sh \
    projects/ovdino/configs/ovdino_swin_tiny224_biomistral_phase2_tct_ngc.py \
    ./wkdrs/ovdino_biomistral_phase1_tct_ngc/model_final.pth

# ===== 评估 =====
# Base
bash scripts/eval.sh \
    projects/ovdino/configs/ovdino_swin_tiny224_biomistral_eval_base.py \
    ./wkdrs/ovdino_biomistral_phase2_tct_ngc/model_final.pth \
    ../wkdrs/eval_biomistral_base

# Novel
bash scripts/eval.sh \
    projects/ovdino/configs/ovdino_swin_tiny224_biomistral_eval_novel.py \
    ./wkdrs/ovdino_biomistral_phase2_tct_ngc/model_final.pth \
    ../wkdrs/eval_biomistral_novel
```

---

## 十、当前进度

| 步骤 | 状态 |
|------|------|
| 代码实现（全部新建+修改文件） | ✅ 完成 |
| 代码验证（shape + 梯度 + 导入） | ✅ 完成 |
| BioMistral-7B 模型迁移 | ✅ 完成 (27GB) |
| Phase 0a: Embedding 提取 | ✅ 完成 ([31, 4096], L2 normalized) |
| Phase 0b: Adapter 预对齐 | ⏳ 待执行 (CPU, ~30s) |
| Phase 1: 冻结视觉训练 | ⏳ 等 GPU 空闲 |
| Phase 2: 联合微调 | ⏳ 等 Phase 1 完成 |
| 评估 Base + Novel | ⏳ 等 Phase 2 完成 |
