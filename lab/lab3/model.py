# This file defines the models used in AI3003 lab experiment

import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    # PE(pos, 2i) = sin(pos / 10000^(2i/d_model)
    # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)
    
    def __init__(self, embedding_dim=128, max_seq_len=512):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.register_buffer("positional_encoding", 
                             self._generate_positional_encoding())
    
    def _generate_positional_encoding(self):
        pe = torch.zeros(self.max_seq_len, self.embedding_dim)
        pos = torch.arange(0, self.max_seq_len).unsqueeze(1) # (max_seq_len, 1)
        id = torch.arange(0, self.embedding_dim).unsqueeze(0) # (1, embedding_dim)
        angle_rates = 1 / torch.pow(10000, (2 * (id // 2)) / self.embedding_dim)
        angle_rads = pos * angle_rates
        pe[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        pe[:, 1::2] = torch.cos(angle_rads[:, 1::2])
        return pe.unsqueeze(0) # (1, max_seq_len, embedding_dim)
    
    def forward(self, x):
        _, seq_len, _ = x.size()
        x = x + self.positional_encoding[:, :seq_len]
        return x
    
class RotaryPositionalEncoding(nn.Module):
    # for each q_i = (q_0 + q_1 j, q_2 + q_3 j, ...)
    # q_i' = q_i * exp(i * theta_i) 
    # = (q_0 cos(theta_0) - q_1 sin(theta_0)) + (q_0 sin(theta_0) + q_1 cos(theta_0)) j
    def __init__(self, embedding_dim=128, max_seq_len=512):
        super(RotaryPositionalEncoding, self).__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("Rotary positional encoding requires an even head dimension")
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        sin_pos, cos_pos = self._generate_positional_encoding()
        self.register_buffer("sin_pos", sin_pos)
        self.register_buffer("cos_pos", cos_pos)
    
    def _generate_positional_encoding(self):
        inv_freq = 1 / (10000 ** (torch.arange(0, self.embedding_dim, 2).float()
                                  / self.embedding_dim))
        pos = torch.arange(self.max_seq_len).float()
        angle_rates = torch.outer(pos, inv_freq)
        sin_pos = torch.sin(angle_rates).unsqueeze(0).unsqueeze(0)
        cos_pos = torch.cos(angle_rates).unsqueeze(0).unsqueeze(0)
        return sin_pos, cos_pos  # (1, 1, max_seq_len, embedding_dim//2)
    
    def _rotate(self, x, sin_pos, cos_pos):
        # x: (B, Nh, N, Dh)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rot = torch.stack((
            x_even * cos_pos - x_odd * sin_pos,
            x_even * sin_pos + x_odd * cos_pos
        ), dim=-1)
        return x_rot.flatten(-2)
    
    def forward(self, Q, K):
        # Q, K: (B, Nh, N, Dh)
        N = Q.size(-2)
        sin_pos = self.sin_pos[:, :, :N].to(dtype=Q.dtype)
        cos_pos = self.cos_pos[:, :, :N].to(dtype=Q.dtype)
        Q_rot = self._rotate(Q, sin_pos, cos_pos)
        K_rot = self._rotate(K, sin_pos, cos_pos)
        return Q_rot, K_rot
        

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 embedding_dim: int = 128,
                 num_heads: int = 8,
                 dropout_rate: float = 0.1,
                 positional_encoding: str = "sinusoidal",
                 max_seq_len: int = 512,
                 causal_attention: bool = False
                 ):
        super(MultiHeadSelfAttention, self).__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.dropout_rate = dropout_rate
        self.positional_encoding = positional_encoding
        self.causal_attention = causal_attention
        
        # Add parameters for query, key, value projections
        self.W_Q = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.W_K = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.W_V = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.W_o = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        
        # Register parameters
        self.register_parameter("W_Q", self.W_Q)
        self.register_parameter("W_K", self.W_K)
        self.register_parameter("W_V", self.W_V)
        self.register_parameter("W_o", self.W_o)
        
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:            
            self.dropout = None
            
        if positional_encoding == "rotary":
            self.rotary_positional_encoding = RotaryPositionalEncoding(
                embedding_dim=self.head_dim,
                max_seq_len=max_seq_len
            )
        elif positional_encoding in ("sinusoidal", "none"):
            self.rotary_positional_encoding = None
        else:
            raise ValueError(f"Unknown positional encoding: {positional_encoding}")

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_V)
        nn.init.xavier_uniform_(self.W_o)
    
    def forward(self, x, padding_mask):
        B, N, D = x.size()
        Q, K, V = x @ self.W_Q, x @ self.W_K, x @ self.W_V
        # (B, N, D) -> (B, N, num_heads, D_h) -> (B, num_heads, N, D_h)
        (Q, K, V) = (
            t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            for t in (Q, K, V)
        )
        
        if self.rotary_positional_encoding is not None:
            Q, K = self.rotary_positional_encoding(Q, K)
        
        # scores : (B, num_heads, N, N)
        attn_scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)
        
        if self.causal_attention:
            causal_mask = torch.triu(
                torch.ones(N, N, dtype=torch.bool, device=x.device),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        key_padding_mask = padding_mask[:, None, None, :]
        attn_scores = attn_scores.masked_fill(
            key_padding_mask,
            float("-inf")
        )
        all_masked = torch.isinf(attn_scores).all(dim=-1, keepdim=True)
        attn_scores = attn_scores.masked_fill(all_masked, 0.0)
        
        scores = torch.softmax(attn_scores, dim=-1)
        scores = scores.masked_fill(all_masked, 0.0)
        attn_outputs = scores @ V  # (B, num_heads, N, D_h)
        attn_outputs = attn_outputs.transpose(1, 2).contiguous().view(B, N, D)
        # (B, N, D)
        output = attn_outputs @ self.W_o
        output = output.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        
        # Dropout
        if self.dropout is not None:
            output = self.dropout(output)
        return output

class DecoderLayer(nn.Module):
    """
    Single layer of the Transformer decoder
    """
    
    def __init__(self,
                 embedding_dim: int = 128,
                 num_heads: int = 8,
                 ff_dim: int = 64,
                 dropout_rate: float = 0.1,
                 positional_encoding: str = "sinusoidal",
                 max_seq_len: int = 512,
                 causal_attention: bool = False
                 ):
        super(DecoderLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.causal_attention = causal_attention
        
        self.self_attention = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            positional_encoding=positional_encoding,
            max_seq_len=max_seq_len,
            causal_attention=causal_attention
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
            
    def forward(self, x, padding_mask):
        # Self-attention
        attn_output = self.self_attention(x, padding_mask=padding_mask)
        x = self.norm1(x + attn_output)
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        if self.dropout is not None:
            ffn_output = self.dropout(ffn_output)
        x = self.norm2(x + ffn_output)
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        
        return x

class Transformer(nn.Module):
    def __init__(self,
                 vocab_size: int = 16384,
                 embedding_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 ff_dim: int = 64,
                 dropout_rate: float = 0.1,
                 positional_encoding: str = "sinusoidal",
                 max_seq_len: int = 512,
                 causal_attention: bool = False,
                 pad_token_id: int = 0,
                 use_kaiming_init: bool = True
                 ):
        super(Transformer, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.positional_encoding = positional_encoding
        self.max_seq_len = max_seq_len
        self.causal_attention = causal_attention
        self.pad_token_id = pad_token_id
        
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_token_id
        )
        
        # Positional Encoding layer
        if positional_encoding == "sinusoidal":
            self.positional_encoding_layer = SinusoidalPositionalEncoding(
                embedding_dim=embedding_dim,
                max_seq_len=max_seq_len
            )
        elif positional_encoding in ("rotary", "none"):
            self.positional_encoding_layer = nn.Identity()
        else:
            raise ValueError(f"Unknown positional encoding: {positional_encoding}")
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            DecoderLayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate,
                positional_encoding=positional_encoding,
                max_seq_len=max_seq_len,
                causal_attention=causal_attention
            ) for _ in range(num_layers)
        ])
        
        self.prob_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.binary_cls_head = nn.Linear(embedding_dim, 1)
        
        # Initialize weights
        if use_kaiming_init:
            self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights using Kaiming initialization for linear layers"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm: weight to 1, bias to 0 (default is fine, but explicit)
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                # Standard embedding initialization
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0)
            elif isinstance(module, MultiHeadSelfAttention):
                module.reset_parameters()
        
    def forward(self, x, mode="prob"):
        # x: (batch_size, seq_len)
        padding_mask = x.eq(self.pad_token_id)
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = self.positional_encoding_layer(x)
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)
        
        if mode == "prob":
            return self.prob_head(x)  # (batch_size, seq_len, vocab_size)
        elif mode == "binary_cls":
            # cls_pos = -1 if self.causal_attention else 0
            cls_pos = -1
            return self.binary_cls_head(x[:, cls_pos, :])  # (batch_size, 1)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
class MLPClassifier(nn.Module):
    # Simple MLP classifier as baseline
    # [B, N] --Embedding--> [B, N, D]
    # --Mean Pooling--> [B, D] --MLP--> [B, 1]
    def __init__(self,
                 vocab_size: int = 16384,
                 embedding_dim: int = 128,
                 hidden_dim: int = 64,
                 mlp_depth = 2,
                 dropout_rate: float = 0.1,
                 pad_token_id: int = 0,
    ):
        super(MLPClassifier, self).__init__()
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_token_id
        )
        mlp_layers = []
        for i in range(mlp_depth):
            in_dim = embedding_dim if i == 0 else hidden_dim
            mlp_layers.append(nn.Linear(in_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            if dropout_rate > 0:
                mlp_layers.append(nn.Dropout(dropout_rate))
        self.mlp = nn.Sequential(*mlp_layers)
        self.classifier = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, mode="binary_cls"):
        if mode != "binary_cls":
            raise ValueError(f"MLPClassifier only supports binary_cls mode, got {mode}")
        padding_mask = x.eq(self.pad_token_id)
        token_count = (~padding_mask).sum(dim=1, keepdim=True).clamp_min(1)
        x = self.embedding(x)  # (B, N, D)
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        x = x.sum(dim=1) / token_count  # (B, D)
        x = self.mlp(x)  # (B, hidden_dim)
        return self.classifier(x)  # (B, 1)
        

class RNN(nn.Module):
    def __init__(self,
                 vocab_size: int = 16384,
                 embedding_dim: int = 128,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout_rate: float = 0.1,
                 bidirectional: bool = False,
                 use_kaiming_init: bool = True):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers,
                          batch_first=True, dropout=dropout_rate if num_layers > 1 else 0,
                          bidirectional=bidirectional)
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.prob_head = nn.Linear(rnn_output_dim, vocab_size)
        self.binary_cls_head = nn.Linear(rnn_output_dim, 1)

        if use_kaiming_init:
            self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, x, mode="prob"):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        if mode == "prob":
            return self.prob_head(x)
        elif mode == "binary_cls":
            return self.binary_cls_head(x[:, -1, :])
        else:
            raise ValueError(f"Unknown mode: {mode}")
