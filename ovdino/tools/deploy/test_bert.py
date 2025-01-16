import itertools

import onnxruntime as ort
import torch
from detrex.data.datasets import clean_words_or_phrase, template_meta
from detrex.modeling.language_backbone import BERTEncoder


def tokenize_category_names(torch_model, category_names):
    category_names = [
        [
            [
                template.format(clean_words_or_phrase(cat_name))
                for template in template_meta["full"]
            ]
            for cat_name in batch_cat_names
        ]
        for batch_cat_names in category_names
    ]
    category_names = list(itertools.chain(*category_names))
    if isinstance(category_names[0], list):
        input_ids = torch.stack(
            [
                torch_model.tokenizer(name, return_mask=True)["input_ids"].squeeze(0)
                for name in category_names
            ],
            dim=0,
        )

        # with torch.no_grad():
        #     text_embed = torch.stack(
        #         [torch_model.language_backbone(name) for name in category_names], dim=0
        #     )  # [bs*num_classes, num_templates, embed_dim]
        #     text_embed = text_embed.mean(1).cpu()

    return input_ids


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
language_backbone = BERTEncoder(
    tokenizer_cfg=dict(tokenizer_name="bert-base-uncased"),
    model_name="bert-base-uncased",
    output_dim=256,
    padding_mode="max_length",
    context_length=48,
    pooling_mode="mean",
    post_tokenize=False,
    is_normalize=False,
    is_proj=False,
    is_freeze=False,
    return_dict=False,
)
language_backbone.to(device)
language_backbone.eval()

category_names = [["person", "cat", "dog"]]
input_ids = tokenize_category_names(language_backbone, category_names)
input_ids = input_ids.to(device)
with torch.no_grad():
    # text_embed = torch.stack(
    #     [language_backbone(input_id) for input_id in input_ids[0]], dim=0
    # )
    text_embed = language_backbone(input_ids)
    # text_embed = text_embed.mean(1)


torch.onnx.export(
    language_backbone,
    input_ids,
    "bert.onnx",
    input_names=["input_ids"],
    output_names=["text_embed"],
    dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"}},
)

session_options = ort.SessionOptions()
session_options.inter_op_num_threads = 4
session_options.intra_op_num_threads = 4
ort_session = ort.InferenceSession(
    "bert.onnx",
    sess_options=session_options,
    provider=[
        "CPUExecutionProvider",
    ],
)
ort_inputs = {"input_ids": input_ids.cpu().numpy()}
ort_outputs = ort_session.run(None, ort_inputs)

print(text_embed)
print(ort_outputs)
