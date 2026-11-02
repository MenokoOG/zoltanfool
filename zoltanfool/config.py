"""The two model sizes. Only n_layer and d_model are different between the two. The smaller model is used for testing and debugging, while the larger model is used for training and inference.
"""
from dataclasses import dataclass



@dataclass(frozen=True)
class Config:
    n_layer: int
    d_model: int
    n_head: int
    context_length: int = 256
    vocab_size: int = 256
    
    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"
        return self.d_model // self.n_head
    
DEBUG = Config(
    n_layer=6,
    d_model=192,
    n_head=3
)
TARGET = Config(
    n_layer=14,
    d_model=384,
    n_head=6
)


def expected_params(c: Config) -> int:
    """N = L(12d^2 + 4d) + (V + T)d + 2d, no bias on linear layers, tied head."""
    d, L = c.d_model, c.n_layer
    return L * (12 * d * d + 4 * d) + (c.vocab_size + c.context_length) * d + 2 * d
