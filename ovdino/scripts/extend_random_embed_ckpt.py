"""Extend a random-embed checkpoint's embedding table from 20 (base) to 31 (base+novel).

The 11 appended novel rows are freshly sampled from N(0, 0.02^2) — the same
init distribution used during training. They have never been trained, so novel
AP will be near zero. This is intentional: the extended checkpoint exists only
to let the eval harness resolve novel class names without crashing.

Usage:
    python scripts/extend_random_embed_ckpt.py \
        --input  wkdrs/ovdino_swin_tiny224_bert_base_random_embed_tct_ngc/model_0078299.pth \
        --output wkdrs/ovdino_swin_tiny224_bert_base_random_embed_tct_ngc/model_0078299_extended31.pth
"""
import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--novel-classes", type=int, default=11)
    parser.add_argument("--init-std", type=float, default=0.02)
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location="cpu")
    state = ckpt["model"]

    key = "random_text_embedding.weight"
    old_weight = state[key]
    n_base, dim = old_weight.shape
    print(f"Original embedding: {old_weight.shape}")

    torch.manual_seed(args.seed)
    novel_weight = torch.empty(args.novel_classes, dim)
    torch.nn.init.normal_(novel_weight, mean=0.0, std=args.init_std)

    state[key] = torch.cat([old_weight, novel_weight], dim=0)
    print(f"Extended embedding: {state[key].shape}  (appended {args.novel_classes} novel rows, seed={args.seed})")

    ckpt["model"] = state
    torch.save(ckpt, args.output)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
