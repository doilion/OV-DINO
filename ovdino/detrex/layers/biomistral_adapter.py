"""BioMistral Adapter MLP for OV-DINO.

Shared projection from BioMistral embedding space (4096d) to
BERT-compatible space (768d), placed before ClassEmbed to preserve
pre-trained ClassEmbed weights.
"""
import torch.nn as nn
import torch.nn.functional as F


class BioMistralAdapterMLP(nn.Module):
    """MLP adapter projecting BioMistral embeddings to BERT-compatible dimension.

    Architecture: LayerNorm -> Linear -> GELU -> Dropout -> Linear

    Args:
        input_dim (int): BioMistral hidden size. Default 4096.
        hidden_dim (int): Intermediate dimension. Default 2048.
        output_dim (int): Output dimension (must match BERT hidden size). Default 768.
        dropout (float): Dropout rate. Default 0.1.
    """

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 2048,
        output_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """Project embeddings: [..., input_dim] -> [..., output_dim].

        No L2 normalization applied (ClassEmbed handles that).
        """
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
