class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.d_model, 4 * c.d_model, bias=False)
        self.down = nn.Linear(4 * c.d_model, c.d_model, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    """Pre-norm: normalize, transform, add back to the input."""

    def __init__(self, c: Config):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(c.d_model), CausalSelfAttention(c)
        self.ln2, self.mlp = nn.LayerNorm(c.d_model), MLP(c)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class ZoltanFool(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab_size, c.d_model)
        self.pos = nn.Embedding(c.context_length, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.d_model)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, y=None):
        B, T = x.shape
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        for b in self.blocks:
            h = b(h)
        logits = self.ln_f(h) @ self.tok.weight.T                   # tied output head
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, n, temperature=1.0):
        for _ in range(n):
            logits, _ = self(ids[:, -self.c.context_length:])
            p = (logits[:, -1] / max(temperature, 1e-6)).softmax(-1)
            ids = torch.cat([ids, torch.multinomial(p, 1)], dim=1)
        return ids