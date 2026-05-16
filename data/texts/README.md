# PSC text prompts for TCT_NGC

Per-class descriptive prompts used by `PseudoLanguageBackbone`. These map a dataset's bare class names (the ones produced by `detrex.data.datasets.custom_ovd.load_custom_ovd`) to the longer, medically-precise strings that the text encoder actually embeds.

All files mirror the WeDetect `data/texts/` conventions exactly — they are bit-for-bit reusable across the two repos.

## Files

- **`tct_ngc_class_prompts_base30.json`** — `{class_name: descriptive_prompt}` for the 30-class base split.
  Style: *"Respiratory tract cytology - Neutrophil"* — plain, descriptive, encoder-agnostic.
  Use with `PseudoLanguageBackbone(name_to_prompt_path=...)` and `build_text_embeddings.py --prompts` (re-keyed by class name in a JSON of the same shape that `--prompts` expects).
- **`tct_ngc_class_keys_base30.json`** — `{class_name: taxonomy_key}` for the same 30 classes.
  Style: *"PSC Category II: Negative — alveolar macrophages"* / *"Bethesda VI: Malignant — papillary thyroid carcinoma"* — uses the actual Papanicolaou Society / Bethesda / Paris / TIS reporting-system labels.
  Pick this one when you want the text encoder to anchor to clinical taxonomy rather than descriptive phrasing.
- **`tct_ngc_fullnames_32.json`** — flat `[str, ...]` list of the 32-class test set (base30 + 2 novel), in `categories[].id` ascending order. Feed directly to `build_text_embeddings.py --prompts-list` when building a test-only cache.

## Typical workflow

```bash
# Encode the descriptive prompts with BiomedCLIP, producing the .pth cache
# that PseudoLanguageBackbone serves at eval time.
python ovdino/projects/ovdino/data/build_text_embeddings.py \
    --ann      datas/custom/annotations/test.json \
    --prompts  data/texts/tct_ngc_class_prompts_base30.json \
    --encoder  microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
    --pool     mean --normalize \
    --out      data/texts/tct_ngc_class_emb_biomedclip.pth

# The .pth.name_to_prompt.json sibling is also written; export both for eval:
export OVDINO_PSC_CACHE=$(realpath data/texts/tct_ngc_class_emb_biomedclip.pth)
export OVDINO_PSC_NAME2PROMPT=$(realpath data/texts/tct_ngc_class_emb_biomedclip.pth.name_to_prompt.json)
export OVDINO_PSC_DIM=512
export OVDINO_PSC_NORMALIZE=1
```

## Class-name ↔ prompt alignment

Both base30 JSONs are **keyed by the class names** that `MetadataCatalog.thing_classes` produces for the corresponding COCO ann file (sorted by ascending `category_id`). If you regenerate the dataset with renamed classes you must regenerate these JSONs too — `PseudoLanguageBackbone` will raise `KeyError` on a missing entry rather than silently fall back.
