"""Precomputed LLM text backbone for OV-DINO.

Loads offline-extracted LLM embeddings (e.g. BioMistral-7B) and serves
them as a drop-in replacement for BERTEncoder.
"""
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PrecomputedEmbeddingBackbone(nn.Module):
    """Precomputed embedding backbone (drop-in replacement for BERTEncoder).

    Loads precomputed embeddings from a .pt file and maps input text strings
    to embedding vectors via lookup. No trainable parameters.

    Args:
        embedding_path (str): Path to .pt file containing:
            - "embeddings": Tensor [num_classes, embed_dim]
            - "categories": list[str] ordered category names
        embedding_dim (int): Expected embedding dimension. Default 4096.
    """

    def __init__(
        self,
        embedding_path: str,
        embedding_dim: int = 4096,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        # Load precomputed embeddings
        data = torch.load(embedding_path, map_location="cpu")
        embeddings = data["embeddings"]  # [num_classes, embed_dim]
        categories = data["categories"]  # list[str]

        assert embeddings.shape[1] == embedding_dim, (
            f"Expected embedding_dim={embedding_dim}, "
            f"got {embeddings.shape[1]}"
        )

        # Store as buffer (no gradients, moves with .to(device))
        self.register_buffer("embeddings", embeddings.float())
        self.num_classes = embeddings.shape[0]

        # Build text -> index mapping (lowercase, hyphen-normalized for robust matching)
        self.text_to_idx: Dict[str, int] = {}
        for idx, name in enumerate(categories):
            key = name.lower().strip().replace("-", " ")
            self.text_to_idx[key] = idx

        logger.info(
            f"[PrecomputedEmbeddingBackbone] Loaded {self.num_classes} classes, "
            f"dim={embedding_dim}, from {embedding_path}"
        )

    def _resolve_index(self, text: str) -> int:
        """Resolve a text string to embedding index."""
        key = text.lower().strip().replace("-", " ")
        if key in self.text_to_idx:
            return self.text_to_idx[key]
        # Try substring match as fallback
        for stored, idx in self.text_to_idx.items():
            if key in stored or stored in key:
                return idx
        return -1

    def forward(self, x: List[str], *args: Any, **kwargs: Any) -> torch.Tensor:
        """Look up precomputed embeddings for category name strings.

        Args:
            x: list of N*C category name strings (flattened across batch).

        Returns:
            Tensor [N*C, embedding_dim]
        """
        indices = []
        for text in x:
            idx = self._resolve_index(text)
            if idx < 0:
                logger.warning(
                    f"[PrecomputedEmbeddingBackbone] Unknown category: '{text}', "
                    f"using zero embedding"
                )
            indices.append(idx)

        emb_list = []
        for idx in indices:
            if 0 <= idx < self.num_classes:
                emb_list.append(self.embeddings[idx])
            else:
                emb_list.append(torch.zeros(self.embedding_dim, device=self.embeddings.device))

        return torch.stack(emb_list, dim=0)  # [N*C, embedding_dim]
