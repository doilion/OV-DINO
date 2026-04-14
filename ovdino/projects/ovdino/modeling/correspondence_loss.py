"""STEGO-inspired correspondence distillation loss for OV-DINO.

Distills inter-class semantic structure from a teacher embedding space
(BioMistral) into the student adapter projection space, using the feature
correspondence formulation from STEGO (Hamilton et al., ICLR 2022).

Core idea: preserve the teacher's pairwise class similarity structure
in the projected space. Computes S matrix from raw embeddings + adapter
in original class order (all 31 classes including novel), avoiding batch
text ordering issues.

NOTE on 31 vs 20 classes: This loss intentionally operates on ALL classes
(base + novel). Preserving the full inter-class structure helps the adapter
maintain BioMistral's medical semantic space, which benefits zero-shot
novel detection. The detection loss only trains on base classes, but the
correspondence regularizer covers the complete class topology.
"""
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CorrespondenceDistillationLoss(nn.Module):
    """STEGO-inspired class-level correspondence distillation.

    Computes student similarity S using the adapter module directly
    on raw teacher embeddings in fixed original order (all classes).

    Args:
        teacher_embeddings_path (str): Path to .pt file containing
            precomputed embeddings. Must have "embeddings" key with
            tensor [C, D] covering ALL classes (base + novel).
        negative_pressure (float): Bias b subtracted from F_ij.
            Controls the balance of attractive vs repulsive forces.
        loss_weight (float): Scalar multiplier for the loss.
    """

    def __init__(
        self,
        teacher_embeddings_path: str,
        negative_pressure: float = 0.4,
        loss_weight: float = 100.0,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.negative_pressure = negative_pressure

        # Load teacher embeddings (all classes: base + novel)
        data = torch.load(teacher_embeddings_path, map_location="cpu")
        raw = data["embeddings"].float()  # [C, D]

        # Store raw embeddings for projection through adapter
        self.register_buffer("raw_embeddings", raw)

        # Precompute teacher similarity matrix F
        raw_normed = F.normalize(raw, p=2, dim=-1)
        F_matrix = torch.mm(raw_normed, raw_normed.t())  # [C, C]
        self.register_buffer("F_matrix", F_matrix)
        self.num_classes = F_matrix.shape[0]

        # Will be set by OVDINO.__init__ via set_adapter()
        self._adapter_ref = None

        logger.info(
            f"[CorrespondenceDistillationLoss] Loaded {self.num_classes} classes, "
            f"b={negative_pressure}, weight={loss_weight}"
        )

    def set_adapter(self, adapter_module: nn.Module):
        """Set reference to the adapter MLP.

        Stored as a plain attribute (not nn.Module) to avoid
        duplicate parameter registration in optimizer groups.
        """
        object.__setattr__(self, "_adapter_ref", adapter_module)

    def forward(self) -> torch.Tensor:
        """Compute correspondence distillation loss.

        Uses adapter module to project ALL raw embeddings in
        original class order, then computes STEGO loss.

        Returns:
            Scalar loss value.
        """
        if self._adapter_ref is None:
            raise RuntimeError(
                "CorrespondenceDistillationLoss: _adapter_ref is not set. "
                "Call set_adapter() after model construction."
            )

        # Project all classes through adapter in fixed order
        # Temporarily disable dropout for stable correspondence
        was_training = self._adapter_ref.training
        self._adapter_ref.eval()
        try:
            projected = self._adapter_ref(self.raw_embeddings)  # [C, 768]
        finally:
            self._adapter_ref.train(was_training)

        # Student similarity: S_ij = cos(feat_i, feat_j)
        feats = F.normalize(projected, p=2, dim=-1, eps=1e-6)
        S = torch.mm(feats, feats.t())  # [C, C]

        # STEGO loss: L = -mean(clamp(S_ij, 0, 0.8) * (F_ij - b))
        mask = ~torch.eye(self.num_classes, dtype=torch.bool, device=feats.device)
        F_off = self.F_matrix[mask]
        S_off = S[mask]

        loss = -(torch.clamp(S_off, min=0.0, max=0.8) * (F_off - self.negative_pressure)).mean()

        return self.loss_weight * loss
