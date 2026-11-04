"""ZoltanFool: a small decoder-only transformer, written from the math."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import Config


class Bigram(nn.Module):
    """Stage 2. The current byte picks a row of logits for the next byte."""

    def __init__(self, vocab_size: int = 256):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, y=None):
        logits = self.table(x)                      # (B, T, V)
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return logits, loss