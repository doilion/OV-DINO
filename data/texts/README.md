# PSC text prompts for TCT_NGC

Per-class long descriptive prompts for ablating "bare class names vs.
medically-precise sentences" through **OV-DINO's stock BERT encoder**.
Pure prompt substitution at dataset-registration time — the model and
checkpoint are untouched.

All files mirror the WeDetect `data/texts/` conventions exactly so they
are bit-for-bit reusable across the two repos.

## Files

- **`tct_ngc_class_prompts_base30.json`** — `{class_name: descriptive_prompt}` for the 30-class base split.
  Style: *"Respiratory tract cytology - Neutrophil"* — plain, descriptive English.
- **`tct_ngc_class_keys_base30.json`** — same 30 classes keyed to **clinical reporting-system labels** (Papanicolaou Society / Bethesda / Paris / TIS).
  Style: *"PSC Category II: Negative — alveolar macrophages"* / *"Bethesda VI: Malignant — papillary thyroid carcinoma"*.
- **`tct_ngc_fullnames_32.json`** — flat 32-prompt list (base30 + 2 novel) sorted by `categories[].id`. For test-set evaluation when the dataset has 32 categories.

## Usage — A/B ablation against stock OV-DINO

Drive the eval through the reference config — it pulls the prompt JSON
in at `LazyConfig.load()` time and substitutes every `category_names`
string before the model sees them. No model code changes, no embedding
caches, no extra dependencies.

```bash
# Baseline: original OV-DINO + bare class names
bash ovdino/scripts/eval.sh \
    ovdino/projects/ovdino/configs/ovdino_swin_tiny224_bert_base_ft_custom_24ep.py \
    inits/ovdino/ovdino_swint_ogc.pth \
    wkdrs/eval_bare

# Treatment: original OV-DINO + long PSC prompts (defaults to base30 JSON)
bash ovdino/scripts/eval.sh \
    ovdino/projects/ovdino/configs/ovdino_swin_tiny224_bert_base_eval_psc_prompts.py \
    inits/ovdino/ovdino_swint_ogc.pth \
    wkdrs/eval_psc

# Treatment variant: same model, taxonomy-keyed labels
OVDINO_PSC_NAME2PROMPT=$(realpath data/texts/tct_ngc_class_keys_base30.json) \
bash ovdino/scripts/eval.sh \
    ovdino/projects/ovdino/configs/ovdino_swin_tiny224_bert_base_eval_psc_prompts.py \
    inits/ovdino/ovdino_swint_ogc.pth \
    wkdrs/eval_psc_keys
```

All three runs use the same checkpoint, same backbone, same BERT-base
text encoder, same training-free eval. The only thing that differs is
the string that BERT tokenizes. That isolates the prompt effect cleanly.

## Class-name ↔ prompt alignment

Both base30 JSONs are **keyed by the class names** that
`MetadataCatalog.thing_classes` produces for the corresponding COCO ann
file (sorted by ascending `category_id`). If you regenerate the dataset
with renamed classes you must regenerate these JSONs too —
`attach_psc_prompts_to_dataset` raises on a missing entry by default
(opt out with `OVDINO_PSC_STRICT=0`).
