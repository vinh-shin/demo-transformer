"""Transformer Encoder từ scratch — cùng kiến trúc với notebook
`notebook/vihsd_transformer_from_scratch.ipynb`, bổ sung trả về
attention weights của mọi layer để ứng dụng vẽ heatmap. Đổi kiến trúc ở notebook thì
phải sửa cả file này, nếu không checkpoint sẽ không nạp được."""

import json
import math
from pathlib import Path

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        out = self.dropout(attn_weights) @ V
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out), attn_weights


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x, attn_weights


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, num_classes,
                 max_len, dropout=0.1, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):
        """Trả về (logits, list attention weights mỗi layer có shape (B, H, T, T))."""
        pad_mask = input_ids != self.pad_idx
        attn_mask = pad_mask.unsqueeze(1).unsqueeze(2)

        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.dropout(self.pos_encoding(x))

        attentions = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask)
            attentions.append(attn)

        mask_f = pad_mask.unsqueeze(-1).float()
        pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)
        return self.classifier(pooled), attentions


def load_model(model_dir: Path):
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    word2idx = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))

    model = TransformerClassifier(**config)
    state = torch.load(model_dir / "model_state_dict.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, word2idx, config
