from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


##########
# Example how config lookslike. I defined it in config.py
##########
@dataclass
class _ConfigModel:
    d_model: int
    n_layers: int
    vocab_size: int
    output_dim: int
    num_heads: int
    head_dim: int
    mask: bool
    mlp_dim: int


class TSPTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.embeddings = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.layers = nn.ModuleList([TSPLayer(config) for _ in range(config.n_layers)])
        self.proj = nn.Linear(config.d_model, config.output_dim)

    def forward(self, tokens: torch.Tensor):
        x = self.embeddings(tokens)  # now x => (b, seq_len, d_model)
        for layer in self.layers:
            x = layer(x)
        logits = self.proj(x)
        return logits[:, -1, :]


class TSPLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config)
        self.attention = MultiHeadAttention(config)
        self.norm2 = RMSNorm(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = self.norm1(x)
        x = self.attention(x)
        x = self.norm2(x)
        x = self.mlp(x)
        return x


class Mask:

    def __init__(self):
        pass

    def causal_mask(self, x: torch.Tensor):
        seq_len = x.shape[-1]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        while mask.dim() < x.dim():
            mask = mask.unsqueeze(0)
        return mask

    def __call__(self, x: torch.Tensor):
        return x.masked_fill(self.causal_mask(x), float('-inf'))


class MultiHeadAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.d_model % config.num_heads == 0, 'config.d_model % config.num_heads must not be float!'

        self.q_w = nn.Linear(config.d_model, config.d_model)
        self.k_w = nn.Linear(config.d_model, config.d_model)
        self.v_w = nn.Linear(config.d_model, config.d_model)
        self.o_w = nn.Linear(config.d_model, config.d_model)

        self.mask = Mask() if config.mask else None

    def forward(self, x):
        # x = (b, seq_len, d_model)
        q = self.q_w(x)
        k = self.k_w(x)
        v = self.k_w(x)

        # split into heads
        q = q.view(q.shape[0], q.shape[1], self.config.num_heads, self.config.head_dim).transpose(-2, -3)
        k = k.view(k.shape[0], k.shape[1], self.config.num_heads, self.config.head_dim).transpose(-2, -3)
        v = v.view(v.shape[0], v.shape[1], self.config.num_heads, self.config.head_dim).transpose(-2, -3)

        att_scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.config.head_dim)

        if self.mask:
            att_scores = self.mask(att_scores)

        att_scores = att_scores.softmax(dim=-1)
        att_scores = att_scores @ v

        att_scores = att_scores.transpose(-2, -3).contiguous().view(x.shape[0], x.shape[1], -1)

        return self.o_w(att_scores)


class RMSNorm(nn.Module):

    def __init__(self, config, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(config.d_model))
        self.eps = eps

    def forward(self, x):
        rms = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return rms * (1 + self.weight.float())


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.mlp_dim)
        self.up_proj = nn.Linear(config.d_model, config.mlp_dim)
        self.down_proj = nn.Linear(config.mlp_dim, config.d_model)

    def forward(self, x):
        # x = (b, seq_len, d_model)
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate='tanh')  # gate = (b, seq_len, mlp_dim)
        up = self.up_proj(x)  # up = (b, seq_len, mlp_dim)
        fuse = gate * up
        x = self.down_proj(fuse)  # x = (b, seq_len, d_model)
        return x
