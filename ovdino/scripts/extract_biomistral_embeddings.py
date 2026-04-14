#!/usr/bin/env python3
"""Phase 0a: BioMistral-7B offline embedding extraction for OV-DINO.

Extracts text embeddings for all TCT_NGC categories (20 base + 11 novel)
using BioMistral-7B with bidirectional attention, mean pooling, and L2 normalization.

Usage:
    python scripts/extract_biomistral_embeddings.py
    python scripts/extract_biomistral_embeddings.py --use-4bit
    python scripts/extract_biomistral_embeddings.py --output embeddings/biomistral_tct_ngc.pt

Output:
    .pt file containing:
        - "embeddings": Tensor [31, 4096]
        - "categories": list[str] (31 category names)
        - "model": "BioMistral-7B"
        - "pooling": "mean"
        - "normalized": True
"""
import argparse
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


# TCT_NGC category definitions (from register_tct_ngc_ovd.py)
# Base categories (20)
BASE_CATEGORIES = [
    "cervical normal cells",
    "cervical atypical squamous cells of undetermined significance",
    "cervical atypical squamous cells cannot exclude high-grade lesion",
    "cervical low-grade squamous intraepithelial lesion",
    "cervical atypical glandular cells and adenocarcinoma with endometrial origin",
    "cervical trichomonas vaginalis infection",
    "cervical dysbacteriosis with herpes and actinomyces",
    "cervical endocervical cells",
    "serous effusion negative samples",
    "serous effusion diseased cells",
    "serous effusion breast cancer cells",
    "thyroid gland papillary cancer",
    "thyroid gland negative samples",
    "thyroid gland suspicious for malignancy",
    "urine cytology negative samples",
    "urine suspicious for high-grade urothelial carcinoma",
    "urine atypical urothelial cells",
    "respiratory tract negative samples",
    "respiratory tract diseased cells",
    "respiratory tract adenocarcinoma",
]

# Novel categories (11)
NOVEL_CATEGORIES = [
    "cervical high-grade squamous intraepithelial lesion and squamous cell carcinoma",
    "cervical candida infection",
    "serous effusion ovarian cancer cells",
    "serous effusion adenocarcinoma cells",
    "thyroid gland suspicious for papillary cancer",
    "thyroid gland atypia of undetermined significance",
    "thyroid gland malignant tumour",
    "thyroid gland non-diagnostic specimen",
    "urine high-grade urothelial carcinoma",
    "respiratory tract squamous cell carcinoma",
    "respiratory tract small cell carcinoma",
]

ALL_CATEGORIES = BASE_CATEGORIES + NOVEL_CATEGORIES


def load_biomistral(device="cuda", use_4bit=False):
    """Load BioMistral-7B with bidirectional attention (no causal mask)."""
    from transformers import AutoModel, AutoTokenizer

    model_name = "BioMistral/BioMistral-7B"
    print(f"Loading {model_name} ({'INT4' if use_4bit else 'FP16'}) ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModel.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto"
        )
    else:
        dtype = torch.float32 if device == "cpu" else torch.float16
        model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device)

    model.eval()

    # Remove causal mask for bidirectional attention
    def _no_causal_mask(self_inner, *args, **kwargs):
        return None

    model._update_causal_mask = types.MethodType(_no_causal_mask, model)

    return model, tokenizer


def encode_texts(model, tokenizer, texts, device="cuda", max_length=256):
    """Encode texts with BioMistral: mean pooling + L2 normalization.

    Args:
        texts: list[str]

    Returns:
        Tensor [N, 4096], L2-normalized
    """
    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    hidden = outputs.last_hidden_state  # [N, seq_len, 4096]

    # Attention mask weighted mean pooling
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    normed = F.normalize(pooled.float(), p=2, dim=-1)
    return normed.cpu()


def main():
    parser = argparse.ArgumentParser(description="Phase 0a: BioMistral offline embedding extraction")
    parser.add_argument("--use-4bit", action="store_true", help="Use INT4 quantization")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output",
        type=str,
        default="embeddings/biomistral_tct_ngc.pt",
        help="Output .pt file path",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Categories: {len(ALL_CATEGORIES)} ({len(BASE_CATEGORIES)} base + {len(NOVEL_CATEGORIES)} novel)")

    # Load model
    model, tokenizer = load_biomistral(args.device, args.use_4bit)

    # Encode all categories
    print("\nEncoding categories ...")
    embeddings = encode_texts(model, tokenizer, ALL_CATEGORIES, args.device)
    print(f"Embeddings shape: {embeddings.shape}")  # [31, 4096]

    # Verify L2 norms
    norms = torch.norm(embeddings, dim=1)
    print(f"L2 norms: [{norms.min():.4f}, {norms.max():.4f}]")

    # Semantic sanity check
    sim = embeddings @ embeddings.t()
    print("\nSemantic pairs:")
    pairs = [
        (0, 3, "Normal vs LSIL"),
        (15, 28, "SHGUC vs HGUC"),
        (0, 10, "Normal vs Breast cancer"),
    ]
    for i, j, desc in pairs:
        print(f"  {ALL_CATEGORIES[i][:35]:35s} <-> {ALL_CATEGORIES[j][:35]:35s}: {sim[i,j]:.4f} ({desc})")

    # Save
    data = {
        "embeddings": embeddings,  # [31, 4096]
        "categories": ALL_CATEGORIES,
        "model": "BioMistral-7B",
        "pooling": "mean",
        "normalized": True,
        "num_base": len(BASE_CATEGORIES),
        "num_novel": len(NOVEL_CATEGORIES),
    }
    torch.save(data, output_path)
    print(f"\nSaved to {output_path}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
