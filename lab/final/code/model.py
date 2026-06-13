from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        return x + self.pe[:, : x.size(1)]


class TemporalTransformerScorer(nn.Module):
    """
    Shared temporal Transformer for all stocks.

    Input : x of shape (batch, num_stocks, seq_len, num_features)
    Output: score of shape (batch, num_stocks)

    The model intentionally outputs scores. Portfolio weights are produced by a
    deterministic layer in losses.py / predict_weights.py. This is more stable than
    asking the network to learn all trading constraints from scratch.
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_cross_section_attention: bool = False,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)

        self.use_cross_section_attention = use_cross_section_attention
        if use_cross_section_attention:
            cs_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.cross_encoder = nn.TransformerEncoder(cs_layer, num_layers=1, enable_nested_tensor=False)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, t, f = x.shape
        z = x.reshape(b * n, t, f)
        z = self.input_proj(z)
        z = self.pos(z)
        z = self.temporal_encoder(z)
        emb = z[:, -1, :].reshape(b, n, -1)
        if self.use_cross_section_attention:
            emb = self.cross_encoder(emb)
        return self.head(emb).squeeze(-1)


class RecurrentScorer(nn.Module):
    """
    Shared GRU/LSTM scorer for stock time-series windows.

    Input : (batch, num_stocks, seq_len, num_features)
    Output: (batch, num_stocks)
    """

    def __init__(
        self,
        num_features: int,
        kind: str = "gru",
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        kind = kind.lower()
        if kind not in {"gru", "lstm"}:
            raise ValueError(f"Unsupported recurrent scorer kind: {kind}")
        self.kind = kind
        rnn_cls = nn.GRU if kind == "gru" else nn.LSTM
        self.input_norm = nn.LayerNorm(num_features)
        self.rnn = rnn_cls(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, t, f = x.shape
        z = x.reshape(b * n, t, f)
        z = self.input_norm(z)
        out, _ = self.rnn(z)
        emb = out[:, -1, :].reshape(b, n, -1)
        return self.head(emb).squeeze(-1)


class GRUScorer(RecurrentScorer):
    def __init__(self, num_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__(num_features, kind="gru", hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)


class LSTMScorer(RecurrentScorer):
    def __init__(self, num_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__(num_features, kind="lstm", hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)


def build_scorer(
    model_name: str,
    num_features: int,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> nn.Module:
    name = model_name.lower()
    if name == "transformer":
        return TemporalTransformerScorer(
            num_features=num_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            use_cross_section_attention=False,
        )
    if name in {"transformer_cs", "xformer_cs"}:
        return TemporalTransformerScorer(
            num_features=num_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            use_cross_section_attention=True,
        )
    if name == "gru":
        return GRUScorer(num_features=num_features, hidden_size=d_model, num_layers=num_layers, dropout=dropout)
    if name == "lstm":
        return LSTMScorer(num_features=num_features, hidden_size=d_model, num_layers=num_layers, dropout=dropout)
    raise ValueError(f"Unknown model: {model_name}")
