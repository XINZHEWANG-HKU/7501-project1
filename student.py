"""Configurable rotary Transformer for the course's independent windows.

The long-training recipe spends the inference budget on a wider RoPE/RMSNorm/
SwiGLU decoder with residual dropout and no evaluation-time cache. Cache, norm,
position and feed-forward variants remain configurable so earlier checkpoints
and mechanism ablations can still be reconstructed. Every prediction remains
strictly causal and no state crosses windows or batch examples.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(x.dtype) * self.weight.to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, context, base=10_000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        inv_freq = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        positions = torch.arange(context, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x):
        length = x.shape[-2]
        cos = self.cos[:length].to(dtype=x.dtype)[None, None]
        sin = self.sin[:length].to(dtype=x.dtype)[None, None]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), dim=-1
        ).flatten(-2)


def _make_norm(width, norm_type):
    if norm_type == "rmsnorm":
        return RMSNorm(width)
    if norm_type == "layernorm":
        return nn.LayerNorm(width)
    raise ValueError(f"unknown norm type: {norm_type}")


class Attention(nn.Module):
    def __init__(self, width, heads, context, use_rope=True, qk_norm=True):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.rope = (
            RotaryEmbedding(self.head_dim, context) if use_rope else nn.Identity()
        )

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q = self.rope(self.q_norm(q))
        k = self.rope(self.k_norm(k))
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(attended.transpose(1, 2).reshape(batch, length, width))


class SwiGLU(nn.Module):
    def __init__(self, width, hidden=None):
        super().__init__()
        hidden = hidden or math.ceil((8 * width / 3) / 64) * 64
        self.gate_up = nn.Linear(width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        gate, value = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class GELUMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.up = nn.Linear(width, 4 * width)
        self.down = nn.Linear(4 * width, width)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(
        self, width, heads, context, norm_type, use_rope, qk_norm, ffn_type,
        residual_dropout=0.0, ffn_width=None
    ):
        super().__init__()
        self.attn_norm = _make_norm(width, norm_type)
        self.ffn_norm = _make_norm(width, norm_type)
        self.attn = Attention(width, heads, context, use_rope, qk_norm)
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(width, ffn_width)
        elif ffn_type == "gelu":
            self.ffn = GELUMLP(width)
        else:
            raise ValueError(f"unknown FFN type: {ffn_type}")
        self.dropout = nn.Dropout(residual_dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


def _effective_shape(config):
    """Scale the supplied baseline unless explicit student dimensions are provided."""
    width = int(config.get(
        "student_width", math.ceil(1.5 * config["width"] / 32) * 32
    ))
    wanted_heads = int(config.get(
        "student_heads", max(config["heads"] + 1, round(1.5 * config["heads"]))
    ))
    valid_heads = [
        h for h in range(1, width + 1)
        if width % h == 0 and (width // h) % 2 == 0
    ]
    heads = min(valid_heads, key=lambda h: abs(h - wanted_heads))
    depth = int(config.get("student_depth", config["depth"] + 2))
    return width, heads, depth


class StudentGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = int(config["context"])
        self.vocab = int(config["vocab"])
        width, heads, depth = _effective_shape(config)
        self.use_cache = bool(config.get("use_cache", True))
        self.use_rope = bool(config.get("use_rope", True))
        norm_type = str(config.get("norm", "rmsnorm")).lower()
        qk_norm = bool(config.get("qk_norm", True))
        ffn_type = str(config.get("ffn", "swiglu")).lower()
        ffn_width = config.get("ffn_width")
        if ffn_width is not None:
            ffn_width = int(ffn_width)
            if ffn_width < 1:
                raise ValueError("ffn_width must be positive")
        residual_dropout = float(config.get("residual_dropout", 0.0))
        if not 0.0 <= residual_dropout < 1.0:
            raise ValueError("residual_dropout must be in [0, 1)")
        scaled_residual_init = bool(
            config.get("scaled_residual_init", True)
        )

        self.token = nn.Embedding(self.vocab, width)
        self.pos = None if self.use_rope else nn.Embedding(self.context, width)
        self.blocks = nn.ModuleList([
            Block(
                width,
                heads,
                self.context,
                norm_type,
                self.use_rope,
                qk_norm,
                ffn_type,
                residual_dropout,
                ffn_width,
            )
            for _ in range(depth)
        ])
        self.norm = _make_norm(width, norm_type)
        self.head = nn.Linear(width, self.vocab, bias=False)

        # Older no-cache checkpoints retained these unused parameters. New
        # inference-only Transformer configs can explicitly omit them.
        retain_cache_parameters = bool(
            config.get("retain_cache_parameters", True)
        )
        if self.use_cache or retain_cache_parameters:
            self.cache_gate = nn.Linear(width, 1)
            initial_temperature = float(config.get("cache_temperature", 0.20))
            fraction = min(
                max((initial_temperature - 0.05) / 0.45, 1e-4), 1 - 1e-4
            )
            self.cache_temperature_logit = nn.Parameter(torch.tensor(
                math.log(fraction / (1 - fraction))
            ))
        else:
            self.cache_gate = None
            self.register_parameter("cache_temperature_logit", None)

        self.apply(self._initialize)
        if self.cache_gate is not None:
            nn.init.zeros_(self.cache_gate.weight)
            nn.init.constant_(self.cache_gate.bias, -2.5)
        if scaled_residual_init:
            residual_std = 0.02 / math.sqrt(2 * depth)
            for block in self.blocks:
                nn.init.normal_(block.attn.out.weight, std=residual_std)
                nn.init.normal_(block.ffn.down.weight, std=residual_std)
        self.head.weight = self.token.weight

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        if ids.ndim != 2:
            raise ValueError("ids must have shape [batch, time]")
        if ids.shape[1] > self.context:
            raise ValueError(
                f"sequence length {ids.shape[1]} exceeds context {self.context}"
            )
        x = self.token(ids)
        if self.pos is not None:
            positions = torch.arange(ids.shape[1], device=ids.device)
            x = x + self.pos(positions)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def _cache_distribution(self, hidden, ids):
        """Return p(next token) from strictly earlier key/value pairs."""
        batch, length, _ = hidden.shape
        if length == 1:
            return hidden.new_zeros(
                (batch, length, self.vocab), dtype=torch.float32
            )

        keys = F.normalize(hidden.float(), dim=-1)
        temperature = 0.05 + 0.45 * torch.sigmoid(
            self.cache_temperature_logit
        )
        scores = torch.matmul(keys, keys.transpose(-1, -2)) / temperature
        causal = torch.ones(
            length, length, device=ids.device, dtype=torch.bool
        ).tril(diagonal=-1)
        weights = F.softmax(
            scores.masked_fill(~causal, -10_000.0), dim=-1
        )
        weights = weights * causal

        # Key j stores token j+1. Since j < query position, every stored value
        # has already been observed by that query and cannot reveal its target.
        values = torch.empty_like(ids)
        values[:, :-1] = ids[:, 1:]
        values[:, -1] = 0
        indices = values[:, None, :].expand(batch, length, length)
        cache = hidden.new_zeros(
            (batch, length, self.vocab), dtype=torch.float32
        )
        return cache.scatter_add(2, indices, weights)

    def forward(self, ids):
        """Training interface: causal next-token scores [batch, time, vocab]."""
        hidden = self.features(ids)
        logits = self.head(hidden).float()
        if not self.use_cache:
            return logits
        base = F.softmax(logits, dim=-1)
        cache = self._cache_distribution(hidden, ids)
        gate = torch.sigmoid(self.cache_gate(hidden).float())
        has_history = (torch.arange(ids.shape[1], device=ids.device) > 0)
        gate = gate * has_history[None, :, None]
        mixture = base * (1.0 - gate) + cache * gate
        return mixture.clamp_min(torch.finfo(torch.float32).tiny).log()

    def predict_log_probs(self, ids):
        """Finite normalized predictions; all temporary cache state is call-local."""
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return StudentGPT(config)
