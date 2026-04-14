import copy

import torch.nn as nn
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import SwinTransformer
from detrex.layers import BioMistralAdapterMLP, PositionEmbeddingSine
from detrex.modeling.language_backbone import PrecomputedEmbeddingBackbone
from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.neck import ChannelMapper
from projects.ovdino.modeling import (
    CorrespondenceDistillationLoss,
    DINOCriterion,
    DINOTransformer,
    DINOTransformerDecoder,
    DINOTransformerEncoder,
    OVDINO,
)

model = L(OVDINO)(
    backbone=L(SwinTransformer)(
        pretrain_img_size=224,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        drop_path_rate=0.1,
        window_size=7,
        out_indices=(1, 2, 3),
    ),
    language_backbone=L(PrecomputedEmbeddingBackbone)(
        embedding_path="embeddings/biomistral_tct_ngc.pt",
        embedding_dim=4096,
    ),
    adapter_mlp=L(BioMistralAdapterMLP)(
        input_dim=4096,
        hidden_dim=2048,
        output_dim=768,
        dropout=0.1,
    ),
    correspondence_loss=L(CorrespondenceDistillationLoss)(
        teacher_embeddings_path="embeddings/biomistral_tct_ngc.pt",
        negative_pressure=0.6,
        loss_weight=100.0,
    ),
    position_embedding=L(PositionEmbeddingSine)(
        num_pos_feats=128,
        temperature=10000,
        normalize=True,
        offset=-0.5,
    ),
    neck=L(ChannelMapper)(
        input_shapes={
            "p1": ShapeSpec(channels=192),
            "p2": ShapeSpec(channels=384),
            "p3": ShapeSpec(channels=768),
        },
        in_features=["p1", "p2", "p3"],
        out_channels=256,
        num_outs=4,
        kernel_size=1,
        norm_layer=L(nn.GroupNorm)(num_groups=32, num_channels=256),
    ),
    transformer=L(DINOTransformer)(
        encoder=L(DINOTransformerEncoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            post_norm=False,
            num_feature_levels="${..num_feature_levels}",
            use_checkpoint=False,
        ),
        decoder=L(DINOTransformerDecoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            return_intermediate=True,
            num_feature_levels="${..num_feature_levels}",
            use_checkpoint=False,
        ),
        num_feature_levels=4,
        two_stage_num_proposals="${..num_queries}",
    ),
    embed_dim=256,
    num_classes=150,
    test_num_classes=31,
    num_queries=900,
    aux_loss=True,
    select_box_nums_for_evaluation=300,
    # text_embed_dim stays 768 (adapter output matches BERT dim)
    text_embed_dim=768,
    criterion=L(DINOCriterion)(
        num_classes="${..num_classes}",
        matcher=L(HungarianMatcher)(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="focal_loss_cost",
            alpha=0.25,
            gamma=2.0,
        ),
        weight_dict={
            "loss_class": 1,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_class_dn": 1,
            "loss_bbox_dn": 5.0,
            "loss_giou_dn": 2.0,
        },
        loss_class_type="focal_loss",
        alpha=0.25,
        gamma=2.0,
        two_stage_binary_cls=False,
    ),
    dn_number=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.120, 57.375],
    vis_period=0,
    input_format="RGB",
    device="cuda",
)

# set aux loss weight dict
base_weight_dict = copy.deepcopy(model.criterion.weight_dict)
if model.aux_loss:
    weight_dict = model.criterion.weight_dict
    aux_weight_dict = {}
    aux_weight_dict.update({k + "_enc": v for k, v in base_weight_dict.items()})
    aux_weight_dict.update({k + "_bcls_enc": v for k, v in base_weight_dict.items()})
    for i in range(model.transformer.decoder.num_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in base_weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    model.criterion.weight_dict = weight_dict
