"""Direct next-N LSTM for delta prediction.

This predicts the next N deltas as a vector all at once from the history window of the final hidden state.
Both the delta history and the PC history are embedded and concatenated as the LSTM input at each timestep.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class EmbeddingLSTM(nn.Module):
    def __init__(
        self,
        num_delta_in: int,
        num_pc_in: int,
        embed_dim: int,
        pc_embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        max_steps: int,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.max_steps = int(max_steps)
        self.num_delta_in = int(num_delta_in)

        self.delta_embed = nn.Embedding(int(num_delta_in), int(embed_dim))
        self.pc_embed = nn.Embedding(int(num_pc_in), int(pc_embed_dim))

        self.lstm = nn.LSTM(
            input_size=int(embed_dim) + int(pc_embed_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=float(dropout))

        # Vector-output head: predict the full horizon in one shot.
        # Output is reshaped to (B, max_steps, num_delta_in).
        self.delta_head_multi = nn.Linear(self.hidden_dim, int(num_delta_in) * self.max_steps)
        self.loss_fn = nn.CrossEntropyLoss()

    def init_state(self, batch_size: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(self.num_layers, int(batch_size), self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, int(batch_size), self.hidden_dim, device=device)
        return h0, c0

    def forward(
        self,
        deltas: torch.Tensor,
        pcs: torch.Tensor,
        *,
        steps: int,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        targets: Optional[torch.Tensor] = None,
    ):
        """Predict next-N delta logits directly from an observed window.

        deltas:  LongTensor (B, T)  — encoded delta history
        pcs:     LongTensor (B, T)  — encoded PC history (same length as deltas)
        targets: LongTensor (B, N)

        Returns:
          logits: (B, N, C)
          state:  (h, c)
          loss:   scalar CE over all steps if targets is provided
        """
        steps = int(steps)
        if steps <= 0:
            raise ValueError(f"steps must be > 0, got {steps}")
        if steps > self.max_steps:
            raise ValueError(
                f"steps ({steps}) exceeds max_steps ({self.max_steps}); "
                "increase max_steps when constructing the model"
            )
        if state is None:
            state = self.init_state(deltas.size(0), deltas.device)

        x_delta = self.delta_embed(deltas.long())       # (B, T, embed_dim)
        x_pc = self.pc_embed(pcs.long())                # (B, T, pc_embed_dim)
        x = torch.cat([x_delta, x_pc], dim=-1)         # (B, T, embed_dim + pc_embed_dim)

        out, state = self.lstm(x, state)
        last = out[:, -1, :]
        last = self.dropout(last)

        # (B, max_steps * C) -> (B, max_steps, C) -> (B, steps, C)
        logits_all = self.delta_head_multi(last)
        logits_all = logits_all.view(deltas.size(0), self.max_steps, self.num_delta_in)
        logits = logits_all[:, :steps, :]

        loss = None
        if targets is not None:
            targets = targets[:, :steps].contiguous().long()
            loss = self.loss_fn(logits.reshape(-1, self.num_delta_in), targets.reshape(-1))

        return logits, state, loss
