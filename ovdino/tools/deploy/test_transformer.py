import copy
import sys

import onnxruntime as ort
import torch
import torch.nn as nn
from detrex.layers import MLP, ClassEmbed, PositionEmbeddingSine

sys.path.insert(0, "../..")

from projects.ovdino.modeling import (
    DINOTransformer,
    DINOTransformerDecoder,
    DINOTransformerEncoder,
)


class ONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        feat_level_0,
        feat_level_1,
        feat_level_2,
        feat_level_3,
        mask_level_0,
        mask_level_1,
        mask_level_2,
        mask_level_3,
        pos_level_0,
        pos_level_1,
        pos_level_2,
        pos_level_3,
        class_embed,
    ):
        # 重新组织输入为列表
        multi_level_feats = [feat_level_0, feat_level_1, feat_level_2, feat_level_3]
        multi_level_masks = [mask_level_0, mask_level_1, mask_level_2, mask_level_3]
        multi_level_position_embeddings = [
            pos_level_0,
            pos_level_1,
            pos_level_2,
            pos_level_3,
        ]

        # 重新组装 query_embeds 和 attn_masks
        query_embeds = (None, None)  # 如果是 None，传入零张量
        attn_masks = [None, None, None]  # 简化处理

        return self.model(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            query_embeds,
            attn_masks=attn_masks,
            class_embed=class_embed,
        )


position_embedding = PositionEmbeddingSine(
    num_pos_feats=128,
    temperature=10000,
    normalize=True,
    offset=-0.5,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer = DINOTransformer(
    encoder=DINOTransformerEncoder(
        embed_dim=256,
        num_heads=8,
        feedforward_dim=2048,
        attn_dropout=0.0,
        ffn_dropout=0.0,
        num_layers=6,
        post_norm=False,
        num_feature_levels=4,
        use_checkpoint=False,
    ),
    decoder=DINOTransformerDecoder(
        embed_dim=256,
        num_heads=8,
        feedforward_dim=2048,
        attn_dropout=0.0,
        ffn_dropout=0.0,
        num_layers=6,
        return_intermediate=True,
        num_feature_levels=4,
        use_checkpoint=False,
    ),
    num_feature_levels=4,
    two_stage_num_proposals=900,
)
class_embed = ClassEmbed(768, 256)
bbox_embed = MLP(256, 256, 4, 3)
class_embed = nn.ModuleList([copy.deepcopy(class_embed) for i in range(8)])
bbox_embed = nn.ModuleList([copy.deepcopy(bbox_embed) for i in range(8)])
transformer.decoder.class_embed = class_embed
transformer.decoder.bbox_embed = bbox_embed
transformer_onnx = ONNXWrapper(transformer)
position_embedding.to(device)
transformer_onnx.to(device)
transformer_onnx.eval()

# suppose the image shape is [800, 800]
multi_level_feats = [
    torch.randn(1, 256, 100, 100).to(device),
    torch.randn(1, 256, 50, 50).to(device),
    torch.randn(1, 256, 25, 25).to(device),
    torch.randn(1, 256, 13, 13).to(device),
]
multi_level_masks = [
    torch.zeros(1, 100, 100).to(device).bool(),
    torch.zeros(1, 50, 50).to(device).bool(),
    torch.zeros(1, 25, 25).to(device).bool(),
    torch.zeros(1, 13, 13).to(device).bool(),
]
multi_level_position_embeddings = [
    position_embedding(mask) for mask in multi_level_masks
]
query_embeds = (None, None)
attn_mask = None
text_embed = torch.randn(1, 80, 768).to(device)

with torch.no_grad():
    (
        inter_states,
        init_reference,
        inter_references,
        enc_state,
        enc_reference,  # [0..1]
    ) = transformer_onnx(
        *multi_level_feats,
        *multi_level_masks,
        *multi_level_position_embeddings,
        class_embed=text_embed,
    )

# 修改输入值
input_values = (
    multi_level_feats[0],
    multi_level_feats[1],
    multi_level_feats[2],
    multi_level_feats[3],
    multi_level_masks[0],
    multi_level_masks[1],
    multi_level_masks[2],
    multi_level_masks[3],
    multi_level_position_embeddings[0],
    multi_level_position_embeddings[1],
    multi_level_position_embeddings[2],
    multi_level_position_embeddings[3],
    text_embed,
)

input_names = [
    "feat_level_0",
    "feat_level_1",
    "feat_level_2",
    "feat_level_3",
    "mask_level_0",
    "mask_level_1",
    "mask_level_2",
    "mask_level_3",
    "pos_level_0",
    "pos_level_1",
    "pos_level_2",
    "pos_level_3",
    "class_embed",
]

output_names = [
    "inter_states",
    "init_reference",
    "inter_references",
    "enc_state",
    "enc_reference",
]

# script model is not supported yet (much error), use tracing instead
# transformer_script = torch.jit.script(transformer_onnx)

torch.onnx.export(
    transformer_onnx,
    input_values,
    "transformer_script.onnx",
    opset_version=12,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={"image": {1: "height", 2: "width"}},
)

session_options = ort.SessionOptions()
session_options.inter_op_num_threads = 4
session_options.intra_op_num_threads = 4
ort_session = ort.InferenceSession(
    "transformer.onnx",
    sess_options=session_options,
    provider=[
        "CPUExecutionProvider",
    ],
)
ort_inputs = {
    "feat_level_0": multi_level_feats[0].cpu().numpy(),
    "feat_level_1": multi_level_feats[1].cpu().numpy(),
    "feat_level_2": multi_level_feats[2].cpu().numpy(),
    "feat_level_3": multi_level_feats[3].cpu().numpy(),
    "mask_level_0": multi_level_masks[0].cpu().numpy(),
    "mask_level_1": multi_level_masks[1].cpu().numpy(),
    "mask_level_2": multi_level_masks[2].cpu().numpy(),
    "mask_level_3": multi_level_masks[3].cpu().numpy(),
    "pos_level_0": multi_level_position_embeddings[0].cpu().numpy(),
    "pos_level_1": multi_level_position_embeddings[1].cpu().numpy(),
    "pos_level_2": multi_level_position_embeddings[2].cpu().numpy(),
    "pos_level_3": multi_level_position_embeddings[3].cpu().numpy(),
    "class_embed": text_embed.cpu().numpy(),
}
# ort_inputs = {
#     "onnx::Shape_0": multi_level_feats[0].cpu().numpy(),
#     "onnx::Shape_1": multi_level_feats[1].cpu().numpy(),
#     "onnx::Shape_2": multi_level_feats[2].cpu().numpy(),
#     "onnx::Shape_3": multi_level_feats[3].cpu().numpy(),
#     "m.1": multi_level_masks[0].cpu().numpy(),
#     "m.3": multi_level_masks[1].cpu().numpy(),
#     "m.5": multi_level_masks[2].cpu().numpy(),
#     "m": multi_level_masks[3].cpu().numpy(),
#     "lang_embeds.1": text_embed.cpu().numpy(),
# }
ort_outputs = ort_session.run(None, ort_inputs)

print(inter_states, init_reference, inter_references, enc_state, enc_reference)
print(ort_outputs)
