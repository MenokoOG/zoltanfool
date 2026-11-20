"""Model sizes. Only n_layer and d_model change between DEBUG and TARGET.

The three flags switch on the upgrades from Module 9. All off gives the plain
GPT-style model from Modules 3 and 4.
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    n_layer: int
    d_model: int
    n_head: int
    context_length: int = 256
    vocab_size: int = 256
    rope: bool = False      # rotary positions instead of a learned position table
    rmsnorm: bool = False   # RMSNorm instead of LayerNorm
    swiglu: bool = False    # gated MLP instead of up / GELU / down

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"
        return self.d_model // self.n_head

    @property
    def ffn_hidden(self) -> int:
        if not self.swiglu:
            return 4 * self.d_model
        # 8d/3 keeps the MLP at 8d^2 parameters, rounded to a multiple of 64
        return 64 * round(8 * self.d_model / 3 / 64)


DEBUG = Config(n_layer=6, d_model=192, n_head=3)
MID = Config(n_layer=8, d_model=256, n_head=4)
TARGET = Config(n_layer=14, d_model=384, n_head=6)
MODERN_DEBUG = Config(n_layer=6, d_model=192, n_head=3, rope=True, rmsnorm=True, swiglu=True)
MODERN_MID = Config(n_layer=8, d_model=256, n_head=4, rope=True, rmsnorm=True, swiglu=True)
MODERN_TARGET = Config(n_layer=14, d_model=384, n_head=6, rope=True, rmsnorm=True, swiglu=True)


def expected_params(c: Config) -> int:
    """Closed-form count. No bias on linear layers, tied output head."""
    d, h = c.d_model, c.ffn_hidden
    norm = d if c.rmsnorm else 2 * d                    # RMSNorm has a gain only
    mlp = 3 * d * h if c.swiglu else 2 * d * h          # SwiGLU has three matrices
    block = 4 * d * d + mlp + 2 * norm
    pos = 0 if c.rope else c.context_length * d         # RoPE has no position table
    return c.n_layer * block + c.vocab_size * d + pos + norm


CONFIGS = {
    "debug": DEBUG, "mid": MID, "target": TARGET,
    "modern-debug": MODERN_DEBUG, "modern-mid": MODERN_MID, "modern-target": MODERN_TARGET,
}


def right_size(tokens_per_s: float, params: float, hours: float, corpus_bytes: float) -> dict:
    """Module 5 calculator. Compute C = 6 N D; with D = 20 N the best size is N = sqrt(C / 120).

    tokens_per_s and params come from the same measured model. Treat the answer as a guide.
    """
    flops_per_s = 6 * params * tokens_per_s
    compute = flops_per_s * hours * 3600
    n = math.sqrt(compute / 120)
    d = 20 * n
    return {"flops_per_s": flops_per_s, "compute": compute, "params": n, "tokens": d,
            "passes": d / corpus_bytes}
