"""
FT-Transformer (Feature Tokenizer Transformer) for Crop Prediction.
Captures complex interactions between soil fertility and climate conditions.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureTokenizer(nn.Module):
    """
    Projects each of the 7 numerical features into a d-dimensional embedding.
    Each feature (N, P, K, pH, Temperature, Humidity, Rainfall) is treated as a token.
    """

    def __init__(self, num_features: int = 7, d_token: int = 64):
        super().__init__()
        self.num_features = num_features
        self.d_token = d_token
        self.weight = nn.Parameter(torch.empty(num_features, 1, d_token))
        self.bias = nn.Parameter(torch.empty(num_features, d_token))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_features)
        Returns:
            tokens: (batch_size, num_features, d_token)
        """
        # x: (B, F) -> (B, F, 1); weight (F, 1, d) -> (1, F, d) for broadcast
        x = x.unsqueeze(-1)  # (B, F, 1)
        out = x * self.weight.permute(1, 0, 2)  # (B, F, 1) * (1, F, d) -> (B, F, d)
        out = out + self.bias.unsqueeze(0)
        return out


class GEGLU(nn.Module):
    """Gated GLU activation: GELU(x[:,:d]) * x[:,d:] used in FFN."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class TransformerBlock(nn.Module):
    """
    Single Transformer layer: MHSA + LayerNorm + Residual, then FFN + LayerNorm + Residual.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_geglu: bool = True,
    ):
        super().__init__()
        # Multi-Head Self-Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        # Position-wise FFN: GEGLU is linear -> 2*d_ff, then GELU(gate)*gate2, then linear -> d
        d_ff_inner = d_ff * 2 if use_geglu else d_ff
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff_inner),
            GEGLU() if use_geglu else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN with residual
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x


class FTTransformer(nn.Module):
    """
    Feature Tokenizer Transformer for tabular crop prediction.
    - Tokenizes 7 numerical features into d-dimensional tokens.
    - Prepends a learnable [CLS] token.
    - Runs a Transformer backbone and uses CLS output for classification.
    """

    def __init__(
        self,
        num_features: int = 7,
        d_token: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 128,
        num_classes: int = 22,
        dropout: float = 0.1,
        use_geglu: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.d_token = d_token
        self.num_classes = num_classes

        self.feature_tokenizer = FeatureTokenizer(
            num_features=num_features,
            d_token=d_token,
        )
        # Learnable [CLS] token (prepended to sequence)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

        encoder_layer = TransformerBlock(
            d_model=d_token,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            use_geglu=use_geglu,
        )
        self.transformer = nn.ModuleList([encoder_layer for _ in range(n_layers)])

        self.head = nn.Linear(d_token, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_features) - scaled numerical features
        Returns:
            logits: (batch_size, num_classes)
        """
        B = x.size(0)
        # Feature tokens: (B, num_features, d_token)
        tokens = self.feature_tokenizer(x)
        # Prepend CLS: (B, 1 + num_features, d_token)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        sequence = torch.cat([cls_tokens, tokens], dim=1)
        # Transformer layers
        for layer in self.transformer:
            sequence = layer(sequence)
        # CLS output (first position)
        cls_output = sequence[:, 0, :]  # (B, d_token)
        logits = self.head(self.dropout(cls_output))
        return logits


def _check_feature_tokenizer():
    """Ensure FeatureTokenizer maps (B, 7) -> (B, 7, d)."""
    ft = FeatureTokenizer(7, 64)
    x = torch.randn(4, 7)
    out = ft(x)
    assert out.shape == (4, 7, 64), out.shape


if __name__ == "__main__":
    _check_feature_tokenizer()
    model = FTTransformer(num_features=7, d_token=64, n_heads=4, n_layers=3, num_classes=22)
    x = torch.randn(8, 7)
    y = model(x)
    assert y.shape == (8, 22)
    print("model.py: FT-Transformer and Feature Tokenizer OK.")
