"""Build a ``{prompt_str: Tensor[D]}`` text-embedding cache.

The cache is consumed by ``PseudoLanguageBackbone`` to bypass the in-loop
BERT encoder. Two input modes:

1. **Per-class prompts** (recommended for medical detectors):

       --ann       COCO ann json (defines the class set)
       --prompts   JSON {class_name: "descriptive prompt sentence", ...}

   The tool emits one embedding per ``prompts[class_name]`` value, plus
   a sibling ``<out>.name_to_prompt.json`` reverse-map you can wire into
   ``PseudoLanguageBackbone(name_to_prompt_path=...)``.

2. **Prompts only**:

       --prompts-list  JSON ``[str, ...]`` — flat list of prompt strings.

   Useful when you want to build a generic cache divorced from any
   specific dataset.

Encoder defaults to ``bert-base-uncased`` (matches OV-DINO's stock
``BERTEncoder``), but the script accepts any HuggingFace model that
exposes ``last_hidden_state`` so you can swap in BiomedCLIP /
PubMedBERT / SapBERT without code changes.

Usage::

    python projects/ovdino/data/build_text_embeddings.py \\
        --ann      datas/custom/annotations/train.json \\
        --prompts  data/texts/tct_ngc_class_prompts.json \\
        --encoder  microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \\
        --pool     mean \\
        --out      data/texts/tct_ngc_class_emb_pubmedbert.pth

The output ``.pth`` is plug-compatible with WeDetect's cache files, so
WeDetect-side caches built against the same prompts can be loaded by
``PseudoLanguageBackbone`` directly (no need to re-run encoding).
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

_logger = logging.getLogger("build_text_embeddings")


def _resolve_prompts(args) -> Dict[str, str]:
    """Return {class_name: prompt}. Plain prompts-list mode keys by prompt
    itself (class_name == prompt)."""
    if args.ann and args.prompts:
        ann = json.loads(Path(args.ann).read_text())
        class_names = [c["name"] for c in sorted(ann["categories"], key=lambda c: c["id"])]
        prompts_map = json.loads(Path(args.prompts).read_text())
        missing = [n for n in class_names if n not in prompts_map]
        if missing:
            raise SystemExit(
                f"--prompts JSON missing entries for {len(missing)} classes:\n  "
                + "\n  ".join(missing)
            )
        return {name: prompts_map[name] for name in class_names}
    if args.prompts_list:
        items = json.loads(Path(args.prompts_list).read_text())
        if not isinstance(items, list):
            raise SystemExit("--prompts-list must be a JSON array of strings")
        return {p: p for p in items}
    raise SystemExit(
        "Must provide either --ann + --prompts, or --prompts-list. "
        "See module docstring for examples."
    )


@torch.no_grad()
def _encode(
    prompts: List[str],
    encoder: str,
    pool: str,
    batch_size: int,
    device: str,
    normalize: bool,
) -> torch.Tensor:
    """Run the HF model over ``prompts`` and return [N, D] tensor."""
    from transformers import AutoModel, AutoTokenizer

    _logger.info("loading %s on %s", encoder, device)
    tok = AutoTokenizer.from_pretrained(encoder)
    model = AutoModel.from_pretrained(encoder).to(device).eval()

    chunks: List[torch.Tensor] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        out = model(**enc)
        hidden = out.last_hidden_state  # [B, L, D]
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        if pool == "mean":
            summed = (hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            feat = summed / denom
        elif pool == "cls":
            feat = hidden[:, 0, :]
        elif pool == "max":
            hidden_masked = hidden.masked_fill(mask == 0, float("-inf"))
            feat = hidden_masked.max(dim=1).values
        else:
            raise ValueError(f"unknown --pool {pool!r}; expected mean/cls/max")
        if normalize:
            feat = torch.nn.functional.normalize(feat, dim=-1)
        chunks.append(feat.cpu().float())
        _logger.info("encoded %d / %d", min(i + batch_size, len(prompts)), len(prompts))

    return torch.cat(chunks, dim=0)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    grp = p.add_mutually_exclusive_group(required=False)
    grp.add_argument("--prompts-list", type=Path,
                     help="JSON file with a flat list of prompt strings")
    p.add_argument("--ann", type=Path,
                   help="COCO ann json (categories[].name defines key set)")
    p.add_argument("--prompts", type=Path,
                   help="JSON {class_name: descriptive_prompt}; "
                        "required when --ann is set")
    p.add_argument("--encoder", default="bert-base-uncased",
                   help="HuggingFace model id (default bert-base-uncased)")
    p.add_argument("--pool", choices=["mean", "cls", "max"], default="mean",
                   help="hidden-state pooling (default mean)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--normalize", action="store_true",
                   help="L2-normalize after pooling (BiomedCLIP convention)")
    p.add_argument("--out", required=True, type=Path,
                   help="output .pth path; sibling .name_to_prompt.json is "
                        "also written when --ann+--prompts are used")
    args = p.parse_args()

    name_to_prompt = _resolve_prompts(args)
    # Encode in input-order, then assemble a dict keyed by prompt string so
    # duplicate prompts share a single embedding entry.
    prompts: List[str] = list(name_to_prompt.values())
    feats = _encode(
        prompts, args.encoder, args.pool, args.batch_size, args.device, args.normalize
    )
    cache: Dict[str, torch.Tensor] = {}
    for prompt, vec in zip(prompts, feats):
        cache[prompt] = vec
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.out)
    _logger.info("wrote cache: %s (%d entries, D=%d)",
                 args.out, len(cache), feats.shape[-1])

    if args.ann:
        sibling = args.out.with_suffix(args.out.suffix + ".name_to_prompt.json")
        sibling.write_text(json.dumps(name_to_prompt, ensure_ascii=False, indent=2))
        _logger.info("wrote name→prompt map: %s", sibling)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
