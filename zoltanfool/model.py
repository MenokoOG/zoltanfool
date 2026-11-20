class CausalSelfAttention(nn.Module):
    """Stage 3. Explicit Q, K, V, scaled scores, causal mask, softmax, head merge."""

    def __init__(self, c: Config):
        super().__init__()
        self.n_head, self.hd = c.n_head, c.head_dim
        self.q = nn.Linear(c.d_model, c.d_model, bias=False)
        self.k = nn.Linear(c.d_model, c.d_model, bias=False)
        self.v = nn.Linear(c.d_model, c.d_model, bias=False)
        self.out = nn.Linear(c.d_model, c.d_model, bias=False)
        mask = torch.tril(torch.ones(c.context_length, c.context_length, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def split(self, t):                              # (B,T,d) -> (B,h,T,hd)
        B, T, _ = t.shape
        return t.view(B, T, self.n_head, self.hd).transpose(1, 2)

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.split(self.q(x)), self.split(self.k(x)), self.split(self.v(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.hd)      # (B,h,T,T)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
        w = scores.softmax(dim=-1)
        y = (w @ v).transpose(1, 2).reshape(B, T, d)               # merge heads
        return self.out(y)