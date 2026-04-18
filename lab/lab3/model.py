# This file defines the models used in AI3003 lab experiment

import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim=128, max_seq_len=512):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.register_buffer("positional_encoding", 
                             self._generate_positional_encoding())
    
    def _generate_positional_encoding(self):
        pe = torch.zeros(self.max_seq_len, self.embedding_dim)
        id = torch.arange(0, self.max_seq_len).unsqueeze(1) # (max_seq_len, 1)
        dim = torch.arange(0, self.embedding_dim).unsqueeze(0) # (1, embedding_dim)
        angle_rates = 1 / torch.pow(10000, (2 * (dim // 2)) / self.embedding_dim)
        angle_rads = id * angle_rates
        pe[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        pe[:, 1::2] = torch.cos(angle_rads[:, 1::2])
        return pe.unsqueeze(0) # (1, max_seq_len, embedding_dim)
    
    def forward(self, x):
        batch_size, seq_len, embedding_dim = x.size()
        x = x + self.positional_encoding[:, :seq_len]
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self):
        raise NotImplementedError(
            "MultiHeadSelfAttention is not implemented yet.")

class DecoderLayer(nn.Module):
    """
    Single layer of the Transformer decoder
    """
    
    def __init__(self,
                 embedding_dim: int = 128,
                 num_heads: int = 8,
                 ff_dim: int = 64,
                 dropout_rate: float = 0.1
                 ):
        super(DecoderLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim)
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        else:
            self.dropout = None
            
    def forward(self, x):
        # Self-attention
        attn_output, _ = self.self_attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        if self.dropout is not None:
            ffn_output = self.dropout(ffn_output)
        x = self.norm2(x + ffn_output)
        
        return x

class Transformer(nn.Module):
    def __init__(self,
                 vocab_size: int = 16384,
                 embedding_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 ff_dim: int = 64,
                 dropout_rate: float = 0.1,
                 positional_encoding: str = "sinusoidal"
                 ):
        super(Transformer, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.positional_encoding = positional_encoding
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional Encoding layer
        if positional_encoding == "sinusoidal":
            self.positional_encoding_layer = SinusoidalPositionalEncoding()
        else:
            raise ValueError(f"Unknown positional encoding: {positional_encoding}")
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            DecoderLayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate
            ) for _ in range(num_layers)
        ])
        
        self.prob_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.binary_cls_head = nn.Linear(embedding_dim, 1)
        
    def forward(self, x, mode="prob"):
        # x: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = self.positional_encoding_layer(x)
        
        for layer in self.layers:
            x = layer(x)
        
        if mode == "prob":
            return self.prob_head(x)  # (batch_size, seq_len, vocab_size)
        elif mode == "binary_cls":
            return self.binary_cls_head(x[:, 0, :])  # (batch_size, 1)
        else:
            raise ValueError(f"Unknown mode: {mode}")