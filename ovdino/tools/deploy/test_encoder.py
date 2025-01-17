import sys

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from detrex.layers import PositionEmbeddingSine

sys.path.insert(0, "../..")

from projects.ovdino.modeling import DINOTransformerEncoder


def get_reference_points(spatial_shapes, valid_ratios, device):
    """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The ratios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
    reference_points_list = []
    for lvl, (H, W) in enumerate(spatial_shapes):
        #  TODO  check this 0.5
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
        )
        ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
        ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    return reference_points


def get_valid_ratio(mask):
    """Get the valid ratios of feature maps of all levels."""
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1)
    valid_W = torch.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W
    valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio


class ONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    # @torch.jit.script_method
    def forward(
        self,
        feat_flatten,
        lvl_pos_embed_flatten,
        mask_flatten,
        spatial_shapes,
        reference_points,
        level_start_index,
        valid_ratios,
    ):
        feat_flatten = feat_flatten.contiguous()
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.contiguous()
        mask_flatten = mask_flatten.contiguous()
        spatial_shapes = spatial_shapes.contiguous()
        reference_points = reference_points.contiguous()
        level_start_index = level_start_index.contiguous()
        valid_ratios = valid_ratios.contiguous()

        return self.model(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )


class EncoderONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        query,
        key,
        value,
        query_pos,
        query_key_padding_mask,
        spatial_shapes,
        reference_points,
        level_start_index,
        valid_ratios,
    ):
        return self.model(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            query_key_padding_mask=query_key_padding_mask,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
position_embedding = PositionEmbeddingSine(
    num_pos_feats=128,
    temperature=10000,
    normalize=True,
    offset=-0.5,
).to(device)
encoder = DINOTransformerEncoder(
    embed_dim=256,
    num_heads=8,
    feedforward_dim=2048,
    attn_dropout=0.0,
    ffn_dropout=0.0,
    num_layers=6,
    post_norm=False,
    num_feature_levels=4,
    use_checkpoint=False,
).to(device)
# encoder_onnx = ONNXWrapper(encoder).to(device)
encoder.eval()
position_embedding.eval()

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
multi_level_pos_embeds = [position_embedding(mask) for mask in multi_level_masks]


feat_flatten = []
mask_flatten = []
lvl_pos_embed_flatten = []
spatial_shapes = []
for lvl, (feat, mask, pos_embed) in enumerate(
    zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)
):
    bs, c, h, w = feat.shape
    spatial_shape = (h, w)
    spatial_shapes.append(spatial_shape)

    feat = feat.flatten(2).transpose(1, 2)  # bs, hw, c
    mask = mask.flatten(1)
    pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
    # lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
    lvl_pos_embed = pos_embed
    lvl_pos_embed_flatten.append(lvl_pos_embed)
    feat_flatten.append(feat)
    mask_flatten.append(mask)
feat_flatten = torch.cat(feat_flatten, 1)
mask_flatten = torch.cat(mask_flatten, 1)
lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
spatial_shapes = torch.as_tensor(
    spatial_shapes, dtype=torch.long, device=feat_flatten.device
)
level_start_index = torch.cat(
    (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
)
valid_ratios = torch.stack([get_valid_ratio(m) for m in multi_level_masks], 1)

reference_points = get_reference_points(
    spatial_shapes, valid_ratios, device=feat.device
)
with torch.no_grad():
    torch_memory = encoder(
        query=feat_flatten,
        key=None,
        value=None,
        query_pos=lvl_pos_embed_flatten,
        query_key_padding_mask=mask_flatten,
        spatial_shapes=spatial_shapes,
        reference_points=reference_points,  # bs, num_token, num_level, 2
        level_start_index=level_start_index,
        valid_ratios=valid_ratios,
    )

# Create wrapped model for ONNX export
wrapped_encoder = EncoderONNXWrapper(encoder)

input_values = [
    feat_flatten,
    None,  # key
    None,  # value
    lvl_pos_embed_flatten,
    mask_flatten,
    spatial_shapes,
    reference_points,
    level_start_index,
    valid_ratios,
]
input_names = [
    "query",
    "key",
    "value",
    "query_pos",
    "query_key_padding_mask",
    "spatial_shapes",
    "reference_points",
    "level_start_index",
    "valid_ratios",
]

torch.onnx.export(
    wrapped_encoder,  # Use wrapped model instead of encoder directly
    tuple(input_values),
    "encoder.onnx",
    input_names=input_names,
    output_names=["memory"],
    # dynamic_axes={
    #     "query": {0: "batch_size", 1: "num_tokens"},
    #     "key": {0: "batch_size", 1: "num_tokens"},
    #     "value": {0: "batch_size", 1: "num_tokens"},
    #     "query_pos": {0: "batch_size", 1: "num_tokens"},
    #     "query_key_padding_mask": {0: "batch_size", 1: "num_tokens"},
    #     "spatial_shapes": {0: "num_levels"},  # This is a 2D tensor [num_levels, 2]
    #     "reference_points": {
    #         0: "batch_size",
    #         1: "num_tokens",
    #         2: "num_levels",
    #     },  # This is a 4D tensor
    #     "level_start_index": {0: "num_levels"},
    #     "valid_ratios": {0: "batch_size", 1: "num_levels"},
    # },
    opset_version=11,
    do_constant_folding=False,
    keep_initializers_as_inputs=True,
    training=torch.onnx.TrainingMode.EVAL,
)

import onnx

model = onnx.load("encoder.onnx")
onnx.checker.check_model(model)
print("Model inputs:", [input.name for input in model.graph.input])

session_options = ort.SessionOptions()
session_options.inter_op_num_threads = 4
session_options.intra_op_num_threads = 4
ort_session = ort.InferenceSession(
    "encoder.onnx",
    sess_options=session_options,
    provider=[
        "CPUExecutionProvider",
    ],
)
ort_inputs = {
    "query": feat_flatten.cpu().numpy(),
    "key": (
        feat_flatten.cpu().numpy() if feat_flatten is not None else None
    ),  # or appropriate null value
    "value": (
        feat_flatten.cpu().numpy() if feat_flatten is not None else None
    ),  # or appropriate null value
    "query_pos": lvl_pos_embed_flatten.cpu().numpy(),
    "query_key_padding_mask": mask_flatten.cpu().numpy(),
    "spatial_shapes": spatial_shapes.cpu().numpy(),
    "reference_points": reference_points.cpu().numpy(),
    "level_start_index": level_start_index.cpu().numpy(),
    "valid_ratios": valid_ratios.cpu().numpy(),
}
ort_outputs = ort_session.run(None, ort_inputs)

diff = np.abs(torch_memory.cpu().numpy() - ort_outputs[0])
print(
    f"max diff: {np.max(diff)}, min diff: {np.min(diff)}, mean diff: {np.mean(diff)}, std diff: {np.std(diff)}"
)
