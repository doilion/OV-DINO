import itertools

import detectron2.data.transforms as T
import onnxruntime as ort
import torch
import torch.nn as nn
from detectron2.data import detection_utils
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import SwinTransformer
from detrex.data.datasets import clean_words_or_phrase, template_meta
from detrex.modeling.language_backbone import BERTEncoder
from detrex.modeling.neck import ChannelMapper


def preprocess_image(image_name):
    original_image = detection_utils.read_image(image_name, format="RGB")
    # Do same preprocessing as DefaultPredictor
    aug = T.ResizeShortestEdge(short_edge_length=800, max_size=1333)
    height, width = original_image.shape[:2]
    image = aug.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

    return image, height, width


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_backbone = SwinTransformer(
    pretrain_img_size=224,
    embed_dim=96,
    depths=(2, 2, 6, 2),
    num_heads=(3, 6, 12, 24),
    drop_path_rate=0.1,
    window_size=7,
    out_indices=(1, 2, 3),
)
neck = ChannelMapper(
    input_shapes={
        "p1": ShapeSpec(channels=192),
        "p2": ShapeSpec(channels=384),
        "p3": ShapeSpec(channels=768),
    },
    in_features=["p1", "p2", "p3"],
    out_channels=256,
    num_outs=4,
    kernel_size=1,
    norm_layer=nn.GroupNorm(num_groups=32, num_channels=256),
)
image_encoder = nn.Sequential(image_backbone, neck)
image_encoder.to(device)
image_encoder.eval()

# image_path = ""
# image, height, width = preprocess_image(image_path)
height, width = (800, 1024)
image = torch.randn(3, height, width)
image = image.unsqueeze(0).to(device)
with torch.no_grad():
    image_embed = image_encoder(image)

# script model is not supported yet (much error), use tracing instead
# image_encoder_script = torch.jit.script(image_encoder)

torch.onnx.export(
    image_encoder,
    image,
    "swin.onnx",
    input_names=["image"],
    output_names=["image_embed"],
    dynamic_axes={"image": {1: "height", 2: "width"}},
)

session_options = ort.SessionOptions()
session_options.inter_op_num_threads = 4
session_options.intra_op_num_threads = 4
ort_session = ort.InferenceSession(
    "swin.onnx",
    sess_options=session_options,
    provider=[
        "CPUExecutionProvider",
    ],
)
ort_inputs = {"image": image.cpu().numpy()}
ort_outputs = ort_session.run(None, ort_inputs)

print(image_embed)
print(ort_outputs)
