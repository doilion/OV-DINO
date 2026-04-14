# BioMistral-7B 集成 OV-DINO 变更文档

## 1. 概述

将 OV-DINO 的文本编码器从 BERT-Base (768-dim) 替换为 BioMistral-7B (4096-dim)，用于组织病理学场景的开放词汇目标检测。通过共享 Adapter MLP + STEGO Correspondence Loss + 多阶段训练，解决随机初始化 MLP 破坏视觉预训练参数的问题。

### 核心设计决策

**共享 Adapter MLP (4096→768)** 置于 ClassEmbed 之前，而非直接修改 ClassEmbed 维度：
- 保留 ClassEmbed 的 7 组预训练权重（`lang_embed_proj(768→256)` + `lang_bias(768-dim)`）
- 减少随机初始化参数面积（仅 1 个 MLP vs. 7 组 ClassEmbed）
- `text_embed_dim` 保持 768，模型其余部分零改动

### 数据流

```
BioMistral离线embedding [N*C, 4096]
  → SharedAdapterMLP [N*C, 768]         ← 唯一新增随机参数（Phase 0 预对齐）
  → ClassEmbed × 7 [768→256]            ← 预训练权重复用
  → dot_product(image_256d, text_256d)   → logits

  + CorrespondenceLoss(全部31类, adapter输出) ← 每步训练
```

---

## 2. 新增文件

| 文件路径 | 功能 | 行数 |
|---------|------|------|
| `ovdino/detrex/modeling/language_backbone/precomputed_embedding.py` | BERTEncoder 的替代：加载离线 BioMistral embeddings，`register_buffer` 存储，Forward 查表返回 `[N*C, 4096]` | 97 |
| `ovdino/detrex/layers/biomistral_adapter.py` | 共享 Adapter MLP：`LayerNorm(4096) → Linear(4096,2048) → GELU → Dropout → Linear(2048,768)` | 55 |
| `ovdino/projects/ovdino/modeling/correspondence_loss.py` | STEGO Correspondence Distillation Loss，从 YOLO-World-Medical 迁移并去除 mmyolo 依赖 | 117 |
| `ovdino/scripts/extract_biomistral_embeddings.py` | Phase 0a：BioMistral-7B 离线 embedding 提取（去 causal mask + mean pooling + L2 normalize）| 192 |
| `ovdino/scripts/prealign_adapter.py` | Phase 0b：STEGO 自蒸馏预对齐 Adapter MLP（1000步，Adam lr=1e-3 cosine→1e-5） | 196 |
| `ovdino/projects/ovdino/configs/models/ovdino_swin_tiny224_biomistral.py` | 模型配置：PrecomputedEmbeddingBackbone + BioMistralAdapterMLP + CorrespondenceDistillationLoss | 140 |
| `ovdino/projects/ovdino/configs/ovdino_swin_tiny224_biomistral_phase1_tct_ngc.py` | Phase 1 训练配置：冻结视觉，8 epoch | 76 |
| `ovdino/projects/ovdino/configs/ovdino_swin_tiny224_biomistral_phase2_tct_ngc.py` | Phase 2 训练配置：全部解冻，16 epoch | 68 |
| `ovdino/projects/ovdino/configs/ovdino_swin_tiny224_biomistral_eval_base.py` | Base 类别评估配置 | — |
| `ovdino/projects/ovdino/configs/ovdino_swin_tiny224_biomistral_eval_novel.py` | Novel 类别评估配置 | — |

---

## 3. 修改文件

| 文件路径 | 改动内容 |
|---------|---------|
| `ovdino/projects/ovdino/modeling/ovdino.py` | 新增 `adapter_mlp`、`adapter_init_checkpoint`、`correspondence_loss`、`freeze_visual` 参数；Forward 中 adapter 投影；loss_dict 追加 `loss_corr`；`freeze_visual()` / `unfreeze_visual()` 方法 |
| `ovdino/projects/ovdino/modeling/__init__.py` | 导出 `CorrespondenceDistillationLoss` |
| `ovdino/detrex/modeling/language_backbone/__init__.py` | 导出 `PrecomputedEmbeddingBackbone` |
| `ovdino/detrex/layers/__init__.py` | 导出 `BioMistralAdapterMLP` |
| `ovdino/detrex/modeling/__init__.py` | 更新导出 |

---

## 4. 三阶段训练流程

### Phase 0（离线，约2分钟）

```bash
source /root/code/OV-DINO/ovdino_env/bin/activate
cd /root/code/OV-DINO/ovdino

# Phase 0a: 提取 BioMistral embeddings
python scripts/extract_biomistral_embeddings.py \
    --output embeddings/biomistral_tct_ngc.pt

# 如果 GPU 显存不足，使用 INT4 量化
python scripts/extract_biomistral_embeddings.py --use-4bit \
    --output embeddings/biomistral_tct_ngc.pt

# Phase 0b: STEGO 预对齐 Adapter MLP
python scripts/prealign_adapter.py \
    --embeddings embeddings/biomistral_tct_ngc.pt \
    --output embeddings/adapter_prealigned.pth
```

输出文件：
- `embeddings/biomistral_tct_ngc.pt` — 31 类 BioMistral embeddings [31, 4096]
- `embeddings/adapter_prealigned.pth` — 预对齐后的 Adapter MLP 权重

### Phase 1（独立训练，8 epoch）

冻结视觉分支（Swin + Neck + Transformer + PositionEmb + LabelEnc），训练 Adapter MLP + ClassEmbed + BBoxEmbed。

```bash
bash scripts/finetune.sh \
    projects/ovdino/configs/ovdino_swin_tiny224_biomistral_phase1_tct_ngc.py \
    ../inits/ovdino/ovdino_swin_tiny224_bert_base.pth
```

| 参数 | 值 |
|------|-----|
| 学习率 | adapter_mlp: 1e-4, class_embed/bbox_embed: 1e-5 |
| 总迭代 | 69,600 (~8 epoch) |
| 评估间隔 | 8,700 iter |
| DDP | `find_unused_parameters=True` |
| Loss | 检测 Loss + Correspondence Loss (weight=100.0) |

输出：`wkdrs/ovdino_biomistral_phase1_tct_ngc/model_final.pth`

### Phase 2（独立训练，16 epoch）

加载 Phase 1 checkpoint，全部解冻，视觉分支低学习率联合微调。

```bash
bash scripts/finetune.sh \
    projects/ovdino/configs/ovdino_swin_tiny224_biomistral_phase2_tct_ngc.py \
    ./wkdrs/ovdino_biomistral_phase1_tct_ngc/model_final.pth
```

| 参数 | 值 |
|------|-----|
| 学习率 | base: 1e-5, backbone: 1e-6 (0.1x) |
| 总迭代 | 139,200 (~16 epoch) |
| 衰减节点 | epoch 11, 14 |
| Loss | 检测 Loss + Correspondence Loss (weight=100.0) |

输出：`wkdrs/ovdino_biomistral_phase2_tct_ngc/model_final.pth`

### 评估

```bash
# Base 类别评估
bash scripts/eval.sh \
    projects/ovdino/configs/ovdino_swin_tiny224_biomistral_eval_base.py \
    ./wkdrs/ovdino_biomistral_phase2_tct_ngc/model_final.pth \
    ../wkdrs/eval_biomistral_base

# Novel 类别评估
bash scripts/eval.sh \
    projects/ovdino/configs/ovdino_swin_tiny224_biomistral_eval_novel.py \
    ./wkdrs/ovdino_biomistral_phase2_tct_ngc/model_final.pth \
    ../wkdrs/eval_biomistral_novel
```

---

## 5. 关键技术细节

### STEGO Correspondence Loss

从 YOLO-World-Medical 验证的核心组件，保持 BioMistral 原始语义空间的类间结构：
- `F_ij` = BioMistral 原始 embedding 余弦相似度（teacher，frozen）
- `S_ij` = Adapter 投影后余弦相似度（student，trainable）
- `L = -mean(clamp(S_ij, 0, 0.8) * (F_ij - b))`，其中 `b = negative_pressure = 0.4`
- 覆盖全部 31 类（含 11 novel），每个训练步都计算

在 YOLO-World-Medical 上的验证：Novel mAP 从 4.1% → 8.2%（使用 STEGO 后翻倍）。

### Adapter MLP 预对齐（Phase 0b）

解决 MLP 随机初始化产生噪声梯度影响后续训练的问题：
- Loss = STEGO correspondence loss + VICReg variance regularization
- 1000 步，Adam lr=1e-3 cosine→1e-5
- 最佳 checkpoint 由 Spearman 相关系数选取（目标 > 0.9）

### Checkpoint 兼容性

- OV-DINO 预训练 checkpoint 正常加载（ClassEmbed 768→256 权重完全匹配）
- 新增的 `adapter_mlp` 和 `correspondence_loss` 键在 DetectionCheckpointer 中自动跳过（mismatched keys）
- `adapter_init_checkpoint` 单独加载预对齐权重

---

## 6. 验证清单

- [ ] Phase 0 输出验证：`biomistral_tct_ngc.pt` shape [31, 4096]，L2 normalized
- [ ] Phase 0b 验证：预对齐后 Spearman 相关系数 > 0.9
- [ ] Shape 验证：fast_dev_run 20 iter，adapter 输出 768-dim
- [ ] 梯度验证：Phase 1 视觉分支梯度 None，adapter + class_embed 非零
- [ ] Correspondence Loss 验证：训练中 loss_corr 稳定下降
- [ ] Phase 1 评估：Base AP 接近 BERT baseline（~26-33 AP）
- [ ] Phase 2 评估：Novel AP > BERT baseline（>8.81 AP）

---

## 7. 依赖

- Python 3.10, PyTorch 2.1.2+cu118
- transformers（BioMistral-7B 加载）
- scipy（Spearman 相关系数）
- bitsandbytes（可选，INT4 量化用）
- 环境：`/root/code/OV-DINO/ovdino_env/bin/activate`

## 8. 参考

- YOLO-World-Medical: https://github.com/doilion/YOLO-WORLD-MEDICAL
- BioMistral: `BioMistral/BioMistral-7B` (HuggingFace)
- STEGO: Hamilton et al., ICLR 2022
- OV-DINO: Li et al., 2024 (arXiv: 2407.07844)
