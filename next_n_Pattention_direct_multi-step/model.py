"""Direct next-N attention model for delta prediction.

Predicts the next N deltas as a vector all at once from the final token
of the attention-encoded history window.

Both the delta history and the PC history are embedded and concatenated at
each position before being passed through MultiheadAttention.
"""

import torch
import torch.nn as nn
from typing import Optional


class EmbeddingPattention(nn.Module):
    def __init__(
        self,
        num_delta_in: int,
        num_pc_in: int,
        delta_embed_dim: int,
        pc_embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        max_steps: int,
    ):
        super().__init__()
        self.num_delta_in = int(num_delta_in)
        self.max_steps = int(max_steps)

        concat_dim = int(delta_embed_dim) + int(pc_embed_dim)
        if concat_dim % int(num_heads) != 0:
            raise ValueError(
                f"concat_dim ({concat_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.delta_embed = nn.Embedding(int(num_delta_in), int(delta_embed_dim))
        self.pc_embed = nn.Embedding(int(num_pc_in), int(pc_embed_dim))

        self.attn = nn.MultiheadAttention(
            embed_dim=concat_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=float(dropout))

        # Vector-output head: predict the full horizon in one shot.
        # Output is reshaped to (B, max_steps, num_delta_in).
        self.delta_head_multi = nn.Linear(concat_dim, int(num_delta_in) * self.max_steps)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        pcs: torch.Tensor,
        deltas: torch.Tensor,
        *,
        steps: Optional[int] = None,
        targets: Optional[torch.Tensor] = None,
    ):
        """Predict next-N delta logits directly from an observed window.

        pcs:     LongTensor (B, T)
        deltas:  LongTensor (B, T)
        targets: LongTensor (B, N) — optional supervision

        Returns:
            logits: (B, steps, num_delta_in)
            loss:   scalar CE over all steps if targets provided, else None
        """
        if steps is None:
            steps = self.max_steps
        steps = int(steps)
        if steps <= 0:
            raise ValueError(f"steps must be > 0, got {steps}")
        if steps > self.max_steps:
            raise ValueError(
                f"steps ({steps}) exceeds max_steps ({self.max_steps}); "
                "increase max_steps when constructing the model"
            )

        x = torch.cat(
            [self.delta_embed(deltas.long()), self.pc_embed(pcs.long())], dim=-1
        )                                                          # (B, T, D)
        x, _ = self.attn(x, x, x)                                 # (B, T, D)
        last = self.dropout(x[:, -1, :])                          # (B, D)

        # (B, max_steps * C) -> (B, max_steps, C) -> (B, steps, C)
        logits_all = self.delta_head_multi(last)
        logits_all = logits_all.view(deltas.size(0), self.max_steps, self.num_delta_in)
        logits = logits_all[:, :steps, :]                         # (B, steps, C)

        loss = None
        if targets is not None:
            t = targets[:, :steps].contiguous().long()
            loss = self.loss_fn(logits.reshape(-1, self.num_delta_in), t.reshape(-1))

        return logits, loss
