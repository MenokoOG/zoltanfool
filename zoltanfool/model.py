"""ZoltanFool: a small decoder-only transformer, written from the math."""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config

Tensor = torch.Tensor
Cache = dict[str, Tensor]


class Bigram(nn.Module):
    """Stage 2. The current byte picks a row of logits for the next byte."""

    def __init__(self, vocab_size: int = 256, context_length: int = 256) -> None:
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)
        # n_layer 0 marks it as a bigram; context_length sets the eval window
        self.c = Config(n_layer=0, d_model=vocab_size, n_head=1, context_length=context_length,
                        vocab_size=vocab_size)

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> tuple[Tensor, Optional[Tensor]]:
        logits = self.table(x)
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, ids: Tensor, n: int, temperature: float = 1.0) -> Tensor:
        """Sample n bytes. The bigram only reads the last byte, so a window of one is enough."""
        for _ in range(n):
            logits, _ = self(ids[:, -1:])
            p = (logits[:, -1] / max(temperature, 1e-6)).softmax(-1)
            ids = torch.cat([ids, torch.multinomial(p, 1)], dim=1)
        return ids


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def make_norm(c: Config) -> nn.Module:
    return RMSNorm(c.d_model) if c.rmsnorm else nn.LayerNorm(c.d_model)


def rope_tables(T: int, hd: int, base: float = 10000.0) -> tuple[Tensor, Tensor]:
    inv = 1.0 / base ** (torch.arange(0, hd, 2).float() / hd)
    ang = torch.outer(torch.arange(T).float(), inv)               # (T, hd/2)
    return ang.cos(), ang.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate channel pairs by an angle that grows with position. x is (B, h, T, hd)."""
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    """Explicit Q, K, V, scaled scores, causal mask, softmax, head merge.

    fast=True swaps the explicit steps for PyTorch's fused kernel. Test 1 proves
    both paths agree, so you can train fast and still trust the slow one.
    A cache of past keys and values lets generation process one new byte at a time.
    """

    mask: Tensor
    cos: Tensor
    sin: Tensor

    def __init__(self, c: Config) -> None:
        super().__init__()
        self.n_head, self.hd = c.n_head, c.head_dim
        self.q = nn.Linear(c.d_model, c.d_model, bias=False)
        self.k = nn.Linear(c.d_model, c.d_model, bias=False)
        self.v = nn.Linear(c.d_model, c.d_model, bias=False)
        self.out = nn.Linear(c.d_model, c.d_model, bias=False)
        self.fast = False
        mask = torch.tril(torch.ones(c.context_length, c.context_length, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)
        if c.rope:
            cos, sin = rope_tables(c.context_length, self.hd)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        self.use_rope = c.rope

    def split(self, t: Tensor) -> Tensor:                              # (B,T,d) -> (B,h,T,hd)
        B, T, _ = t.shape
        return t.view(B, T, self.n_head, self.hd).transpose(1, 2)

    def forward(self, x: Tensor, cache: Optional[Cache] = None, pos: int = 0) -> Tensor:
        B, T, d = x.shape
        q, k, v = self.split(self.q(x)), self.split(self.k(x)), self.split(self.v(x))
        if self.use_rope:
            q = apply_rope(q, self.cos[pos:pos + T], self.sin[pos:pos + T])
            k = apply_rope(k, self.cos[pos:pos + T], self.sin[pos:pos + T])
        if cache is not None:
            if "k" in cache:
                k, v = torch.cat([cache["k"], k], dim=2), torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        Tk = k.size(2)
        mask = self.mask[Tk - T:Tk, :Tk]             # query i sees keys 0 .. (Tk - T + i)
        if self.fast:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True) if (cache is None and T == Tk) \
                else F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.hd)      # (B,h,T,Tk)
            scores = scores.masked_fill(~mask, float("-inf"))
            y = scores.softmax(dim=-1) @ v
        return self.out(y.transpose(1, 2).reshape(B, T, d))           # merge heads


class MLP(nn.Module):
    def __init__(self, c: Config) -> None:
        super().__init__()
        self.swiglu = c.swiglu
        h = c.ffn_hidden
        self.up = nn.Linear(c.d_model, h, bias=False)
        self.down = nn.Linear(h, c.d_model, bias=False)
        if c.swiglu:
            self.gate = nn.Linear(c.d_model, h, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.swiglu:
            return self.down(F.silu(self.gate(x)) * self.up(x))
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    """Pre-norm: normalize, transform, add back to the input."""

    def __init__(self, c: Config) -> None:
        super().__init__()
        self.ln1, self.attn = make_norm(c), CausalSelfAttention(c)
        self.ln2, self.mlp = make_norm(c), MLP(c)

    def forward(self, x: Tensor, cache: Optional[Cache] = None, pos: int = 0) -> Tensor:
        x = x + self.attn(self.ln1(x), cache, pos)
        return x + self.mlp(self.ln2(x))


class ZoltanFool(nn.Module):
    def __init__(self, c: Config) -> None:
        super().__init__()
        self.c: Config = c
        self.tok = nn.Embedding(c.vocab_size, c.d_model)
        self.pos = None if c.rope else nn.Embedding(c.context_length, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = make_norm(c)
        self.apply(self._init)
        # scale the two layers that write into the residual stream, GPT-2 style
        for name, p in self.named_parameters():
            if name.endswith(("attn.out.weight", "mlp.down.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def set_fast(self, fast: bool = True) -> None:
        for b in self.blocks:
            assert isinstance(b, Block)
            b.attn.fast = fast

    def forward(self, x: Tensor, y: Optional[Tensor] = None, caches: Optional[list[Cache]] = None,
                pos: int = 0) -> tuple[Tensor, Optional[Tensor]]:
        _, T = x.shape
        h = self.tok(x)
        if self.pos is not None:
            h = h + self.pos(torch.arange(pos, pos + T, device=x.device))
        for i, b in enumerate(self.blocks):
            h = b(h, None if caches is None else caches[i], pos)
        logits = self.ln_f(h) @ self.tok.weight.T                   # tied output head
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, ids: Tensor, n: int, temperature: float = 1.0) -> Tensor:
        """Sample n bytes. Uses a key/value cache, so each new byte costs one position."""
        T0 = self.c.context_length
        caches: list[Cache] = [{} for _ in self.blocks]
        logits, _ = self(ids[:, -T0:], caches=caches)
        pos = min(ids.size(1), T0)
        for _ in range(n):
            p = (logits[:, -1] / max(temperature, 1e-6)).softmax(-1)
            nxt = torch.multinomial(p, 1)
            ids = torch.cat([ids, nxt], dim=1)
            if pos >= T0:                                           # window full: restart the cache
                caches = [{} for _ in self.blocks]
                logits, _ = self(ids[:, -(T0 - 1):], caches=caches)
                pos = T0 - 1
            else:
                logits, _ = self(nxt, caches=caches, pos=pos)
                pos += 1
        return ids


LanguageModel = Bigram | ZoltanFool
