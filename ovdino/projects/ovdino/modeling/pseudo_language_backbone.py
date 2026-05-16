"""Pre-computed text-embedding "language backbone" for OV-DINO.

Drop-in replacement for ``detrex.modeling.language_backbone.BERTEncoder`` when
the prompts are known in advance — typical for medical detectors where each
class has a long, fixed descriptive sentence ("PSC" = per-class prompt /
prompt-keyed static cache in WeDetect terminology).

Why:
  * BERT-base on a 30-class TBS-style prompt list still costs ~tens of ms per
    forward; PSC lookup costs ~µs.
  * Decouples the text encoder from the detector: you can encode prompts
    offline with **BiomedCLIP / PubMedBERT / SapBERT**, and the detector
    never depends on those packages at training/eval time.
  * Stable text features — no risk of the LM drifting under joint
    optimization.

Cache file format (.pth):

    {prompt_str: torch.FloatTensor of shape [D]}

Exactly the same schema WeDetect uses, so a cache built for WeDetect's
biomedclip path is **directly reusable** here.

Lookup contract:
  * The model passes a flat ``list[str]`` of prompts to ``forward``.
  * Each entry must be present as a key in the cache, **or** match the
    portion before ``prompt_prefix_delimiter`` (default "/", mirroring
    WeDetect's split trick that lets you tag the same prompt with a
    template suffix without bloating the cache).
  * Missing keys raise loudly — silent fallback to zeros would produce
    plausible-looking but garbage AP.

Usage in a LazyConfig::

    from projects.ovdino.modeling.pseudo_language_backbone import (
        PseudoLanguageBackbone,
    )

    model.language_backbone = L(PseudoLanguageBackbone)(
        text_embed_path="data/texts/tct_ngc_class_emb_biomedclip.pth",
        output_dim=512,        # MUST match model.text_embed_dim
        is_normalize=False,    # BiomedCLIP emb already L2-normalized
    )

You must also raise ``model.text_embed_dim`` to match the cache's D, and
(if the text encoder dim differs from BERT-base's 768) wire that through
the rest of the model. The reference config sets this up end-to-end.
"""

import itertools
import json
import logging
from typing import List, Optional

import torch
import torch.nn as nn


_logger = logging.getLogger(__name__)


class PseudoLanguageBackbone(nn.Module):
    """Look-up-table language backbone for fixed-prompt detection.

    API-compatible with ``BERTEncoder``: takes ``list[str]`` (or
    ``list[list[str]]`` when callers split per-image), returns a tensor
    of shape ``[N, D]`` where ``N`` is the total number of prompts.

    Args:
        text_embed_path: ``.pth`` file holding a ``{str: Tensor[D]}`` dict
            (or a dict with ``"text_embed"`` key for compatibility with the
            metadata-style WeDetect files).
        test_embed_path: optional separate cache for eval. Defaults to the
            train cache so a single file works for both. Useful when the
            test class set differs from the train set.
        name_to_prompt_path: optional JSON file mapping
            ``{class_name: prompt_string}`` — if supplied, ``forward`` first
            translates each incoming class name to its prompt before lookup.
            Use this when ``load_custom_ovd`` produces bare class names but
            your cache is keyed by long descriptive prompts.
        prompt_prefix_delimiter: if set (e.g. "/"), keys are stripped at
            this delimiter before lookup. Lets you append a template tag
            to the prompt without duplicating the embedding entry.
            Default "/" to match WeDetect.
        output_dim: declared output dim ``D``. Validated against the cache;
            raises on mismatch so a stale cache cannot silently swap dims.
        is_normalize: if True, L2-normalize embeddings before returning.
            Off by default since most cache builders already normalize.
        is_freeze: accepted for API parity with BERTEncoder; this module
            has no trainable parameters so the flag is a no-op.
    """

    def __init__(
        self,
        text_embed_path: str,
        test_embed_path: Optional[str] = None,
        name_to_prompt_path: Optional[str] = None,
        prompt_prefix_delimiter: Optional[str] = "/",
        output_dim: Optional[int] = None,
        is_normalize: bool = False,
        is_freeze: bool = True,  # noqa: ARG002  (accepted for parity)
    ):
        super().__init__()

        # ---- load cache(s) ----
        self._train_embed = self._load_cache(text_embed_path)
        if test_embed_path is None or test_embed_path == text_embed_path:
            self._test_embed = self._train_embed
        else:
            self._test_embed = self._load_cache(test_embed_path)

        # ---- optional name → prompt translation ----
        if name_to_prompt_path:
            with open(name_to_prompt_path, "r") as f:
                self._name_to_prompt = dict(json.load(f))
            _logger.info(
                "[PseudoLanguageBackbone] loaded name→prompt with %d entries",
                len(self._name_to_prompt),
            )
        else:
            self._name_to_prompt = None

        self._prompt_prefix_delimiter = prompt_prefix_delimiter
        self._is_normalize = is_normalize

        # ---- shape validation ----
        any_vec = next(iter(self._train_embed.values()))
        cached_dim = any_vec.shape[-1]
        if output_dim is not None and output_dim != cached_dim:
            raise ValueError(
                f"PseudoLanguageBackbone: output_dim={output_dim} mismatches "
                f"cache dim={cached_dim} ({text_embed_path}). Update "
                "model.text_embed_dim to the cache's D or rebuild the cache."
            )
        self._embed_dim = int(cached_dim)

        # A dummy buffer purely so .to(device) calls move *something* on this
        # module and we know where to put returned tensors at forward time.
        self.register_buffer("_device_buffer", torch.zeros(1), persistent=False)

        _logger.info(
            "[PseudoLanguageBackbone] %d train keys / %d test keys, D=%d, "
            "normalize=%s, prefix_delim=%r",
            len(self._train_embed),
            len(self._test_embed),
            self._embed_dim,
            is_normalize,
            prompt_prefix_delimiter,
        )

    @staticmethod
    def _load_cache(path: str) -> dict:
        pkg = torch.load(path, map_location="cpu", weights_only=False)
        # Accept either a flat {str: Tensor} dict or a wrapper holding it.
        if isinstance(pkg, dict) and all(
            isinstance(v, torch.Tensor) for v in pkg.values()
        ):
            return pkg
        if isinstance(pkg, dict) and "text_embed" in pkg and isinstance(
            pkg["text_embed"], dict
        ):
            return pkg["text_embed"]
        raise ValueError(
            f"PseudoLanguageBackbone: unsupported cache format in {path}; "
            "expected a dict {prompt_str: Tensor[D]} or a wrapper with key "
            "'text_embed' holding such a dict."
        )

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def _lookup_one(self, raw: str, table: dict) -> torch.Tensor:
        key = raw
        if self._name_to_prompt is not None:
            key = self._name_to_prompt.get(key, key)
        if self._prompt_prefix_delimiter:
            key = key.split(self._prompt_prefix_delimiter, 1)[0]
        if key not in table:
            raise KeyError(
                f"PseudoLanguageBackbone: no cache entry for prompt "
                f"{key!r} (input was {raw!r}). Either rebuild the cache to "
                "include this prompt, or extend name_to_prompt JSON."
            )
        return table[key]

    def forward(self, x) -> torch.Tensor:
        """Look up the cached embedding for each prompt.

        Args:
            x: ``list[str]`` (flat) **or** ``list[list[str]]`` (per-image).
                The latter is what OV-DINO's multi-template inference path
                produces; we flatten internally and let the caller reshape.

        Returns:
            Tensor of shape ``[N, D]`` where ``N == len(flat(x))``.
        """
        if len(x) > 0 and isinstance(x[0], list):
            flat: List[str] = list(itertools.chain(*x))
        else:
            flat = list(x)

        table = self._train_embed if self.training else self._test_embed
        vecs = [self._lookup_one(s, table) for s in flat]
        out = torch.stack(vecs, dim=0)  # [N, D]
        out = out.to(self._device_buffer.device, non_blocking=True).float()
        out.requires_grad_(False)
        if self._is_normalize:
            out = torch.nn.functional.normalize(out, dim=-1)
        return out
