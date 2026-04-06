"""
Bi-LSTM + Multi-Head Self-Attention Model for Traffic Congestion Prediction.

Novel contribution:
- Bidirectional LSTM captures both forward and backward temporal patterns
- Multi-Head Self-Attention reveals which time steps matter most (interpretable)
- Designed for lane-level prediction in Indian heterogeneous traffic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention mechanism for temporal importance weighting."""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        residual = x

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        output = self.layer_norm(output + residual)

        return output, attn_weights


class BiLSTMAttentionPredictor(nn.Module):
    """
    Bi-LSTM + Multi-Head Self-Attention for traffic congestion prediction.

    Architecture:
        Input Features → Bi-LSTM → Multi-Head Attention → Fully Connected → Congestion Class
    """

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projections
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Merge bidirectional outputs
        self.bi_merge = nn.Linear(hidden_dim * 2, hidden_dim)

        # Multi-Head Self-Attention
        self.attention = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 4, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch, seq_len, input_dim) — sequence of traffic features
            return_attention: if True, also return attention weights
        Returns:
            logits: (batch, num_classes)
            attn_weights: (batch, num_heads, seq_len, seq_len) [optional]
        """
        # Project input
        x = self.input_proj(x)  # (batch, seq, hidden)

        # Bi-LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)
        lstm_out = self.bi_merge(lstm_out)  # (batch, seq, hidden)

        # Multi-Head Attention
        attended, attn_weights = self.attention(lstm_out)  # (batch, seq, hidden)

        # Use last timestep for classification
        last_hidden = attended[:, -1, :]  # (batch, hidden)

        # Classify
        logits = self.classifier(last_hidden)

        if return_attention:
            return logits, attn_weights
        return logits

    def predict_proba(self, x):
        """Return class probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)


def create_model(input_dim=11, num_classes=3, device="cpu"):
    """Factory function to create and return the model."""
    model = BiLSTMAttentionPredictor(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        num_classes=num_classes,
        dropout=0.3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] BiLSTM+Attention | Params: {total_params:,} | Trainable: {trainable:,}")
    return model
