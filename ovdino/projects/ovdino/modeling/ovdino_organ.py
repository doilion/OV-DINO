"""Organ-conditional OV-DINO: inference-time class restriction by organ.

Mirrors WeDetect's organ-prior mechanism (``YOLOWorldHead.predict_by_feat`` +
``organ_class_mask`` buffer): when a medical patch's organ is known a priori
(e.g. thyroid, urine), zero out cross-organ class scores **before** top-k so
that the detector cannot output a class belonging to a different organ.

Usage in a LazyConfig::

    from projects.ovdino.modeling.ovdino_organ import OVDINOOrganAware
    model._target_ = OVDINOOrganAware
    model.organ_mask_path = "data/texts/tct_ngc_class_organ_mask.pt"

The mask file is produced by ``projects/ovdino/data/build_class_organ_mask.py``
and must list ``class_names`` in the **contiguous index order** that the
dataset's ``thing_classes`` uses (the mask builder enforces this by sorting
the COCO categories the same way ``detrex.data.datasets.custom_ovd`` does).

Per-image organ ids are read from ``batched_inputs[b]["organ_id"]``. They are
populated by attaching an extractor at dataset-registration time — see
``projects/ovdino/data/organ_aware_register.py``.
"""

from typing import List, Optional

import torch

from projects.ovdino.modeling.ovdino import OVDINO


# sigmoid(-1e4) underflows to 0 cleanly in fp16/fp32 without producing NaNs.
_LARGE_NEG = -1e4


class OVDINOOrganAware(OVDINO):
    """OV-DINO that zeros out cross-organ class scores at inference time."""

    def __init__(
        self,
        *args,
        organ_mask_path: Optional[str] = None,
        organ_mask_strict: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._organ_mask_strict = organ_mask_strict
        self._cur_organ_ids: Optional[List[int]] = None
        self._organ_names: Optional[List[str]] = None
        self._mask_class_names: Optional[List[str]] = None
        # Cache of (device, dtype) → mask tensor cast to that combo. Avoids
        # repeating the `.to(...)` allocation on every eval step under AMP
        # where box_cls may alternate dtype across forwards.
        self._mask_cast_cache: Optional[tuple] = None

        if organ_mask_path is None:
            self.organ_class_mask = None
            return

        pkg = torch.load(organ_mask_path, weights_only=False, map_location="cpu")
        mask = pkg["mask"].to(torch.float32)
        # Every class must map to exactly one organ. A row that sums to 0 would
        # silently mask all logits to -inf for that class; a row that sums to
        # >1 means the taxonomy is multi-organ for a single class, which we
        # don't support yet.
        row_sums = mask.sum(dim=1)
        if not torch.all(row_sums == 1.0):
            raise ValueError(
                f"organ mask {organ_mask_path} has rows summing to "
                f"{row_sums.unique().tolist()} (each class must map to "
                "exactly one organ)."
            )
        self.register_buffer("organ_class_mask", mask, persistent=False)
        self._mask_class_names = list(pkg["class_names"])
        self._organ_names = list(pkg["organ_names"])

    # ------------------------------------------------------------------
    # forward: stash per-image organ_ids so inference() can read them
    # ------------------------------------------------------------------
    def forward(self, batched_inputs):
        if (
            not self.training
            and getattr(self, "organ_class_mask", None) is not None
        ):
            self._cur_organ_ids = self._extract_organ_ids(batched_inputs)
        else:
            self._cur_organ_ids = None
        try:
            return super().forward(batched_inputs)
        finally:
            self._cur_organ_ids = None

    def _extract_organ_ids(self, batched_inputs) -> List[int]:
        n_organs = self.organ_class_mask.shape[1]
        organ_ids: List[int] = []
        for b, d in enumerate(batched_inputs):
            raw = d.get("organ_id", None) if isinstance(d, dict) else None
            o = self._coerce_organ_id(raw, n_organs)
            if o is None:
                if self._organ_mask_strict:
                    raise RuntimeError(
                        f"sample {b} has invalid organ_id={raw!r} but model "
                        "was built with organ_mask_path; expected an int in "
                        f"[0, {n_organs}). Either set organ_mask_strict=False "
                        "on the model, or ensure the dataset/mapper attaches "
                        "an 'organ_id' key to every record (see "
                        "projects/ovdino/data/organ_aware_register.py)."
                    )
                # Non-strict: -1 means "do not mask this sample". We mark it
                # with the all-ones organ column built on the fly downstream.
                organ_ids.append(-1)
            else:
                organ_ids.append(o)
        return organ_ids

    @staticmethod
    def _coerce_organ_id(raw, n_organs: int) -> Optional[int]:
        """Return a validated organ_id int, or None if invalid.

        Rejects ``None``, ``bool`` (Python ``True``/``False`` would silently
        coerce to 1/0), non-integral floats (e.g. 2.5 would round to 2), and
        out-of-range values. Accepts plain ints and 0-d ``torch.Tensor``.
        """
        if raw is None or isinstance(raw, bool):
            return None
        if isinstance(raw, torch.Tensor):
            if raw.numel() != 1:
                return None
            raw = raw.item()
        if isinstance(raw, float):
            if not raw.is_integer():
                return None
            raw = int(raw)
        if not isinstance(raw, int):
            return None
        if not (0 <= raw < n_organs):
            return None
        return raw

    # ------------------------------------------------------------------
    # inference: pre-process box_cls so super().inference() does the rest
    # ------------------------------------------------------------------
    def inference(self, box_cls, box_pred, image_sizes, category_start_index=0):
        mask = getattr(self, "organ_class_mask", None)
        organ_ids = self._cur_organ_ids
        if mask is None or organ_ids is None:
            return super().inference(
                box_cls, box_pred, image_sizes, category_start_index
            )

        C = box_cls.shape[-1]
        if mask.shape[0] != C:
            raise RuntimeError(
                f"organ_class_mask has {mask.shape[0]} classes but box_cls "
                f"has {C}. Mask was built for a different class list; rerun "
                "tools/build_class_organ_mask.py against this test ann file."
            )
        # Cache one cast per (device, dtype). With AMP eval, box_cls dtype
        # may alternate; without caching we'd allocate a fresh fp16/fp32 copy
        # every step.
        key = (box_cls.device, box_cls.dtype)
        if self._mask_cast_cache is None or self._mask_cast_cache[0] != key:
            self._mask_cast_cache = (
                key,
                mask.to(device=box_cls.device, dtype=box_cls.dtype),
            )
        mask = self._mask_cast_cache[1]

        # Build per-image [B, C] mask. organ_id == -1 (non-strict skip) gets
        # all-ones so that sample is unaffected.
        ones_col = mask.new_ones(C, 1)
        cols = []
        for o in organ_ids:
            cols.append(mask[:, o : o + 1] if o >= 0 else ones_col)
        per_image_mask = torch.cat(cols, dim=1).T  # [B, C]
        per_image_mask = per_image_mask.unsqueeze(1)  # [B, 1, C]

        # Logit clamp: in-organ logits stay; out-of-organ logits become
        # _LARGE_NEG so sigmoid → 0 before top-k. This keeps super().inference
        # intact (it still applies sigmoid + sqrt + topk).
        box_cls = box_cls * per_image_mask + (1.0 - per_image_mask) * _LARGE_NEG
        return super().inference(
            box_cls, box_pred, image_sizes, category_start_index
        )
