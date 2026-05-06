"""Dual-head LSTM for next-n prefetching (autoregressive evaluation).

Goal: keep this as small and reviewable as the original next_n_lstm.

Training:
  - windowed supervision on next-1 only.

Evaluation:
  - true autoregressive rollout for next-N, reporting loss/accuracy for:
    next-1 top1/topK and next-N top1/topK (for both delta + PC heads).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class EmbeddingLSTM(nn.Module):
    def __init__(
        self,
        num_pc: int,
        num_delta_in: int,
        delta_embed_dim: int,
        pc_embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pc_loss_weight: float,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.pc_loss_weight = float(pc_loss_weight)

        self.pc_embed = nn.Embedding(int(num_pc), int(pc_embed_dim))
        self.delta_embed = nn.Embedding(int(num_delta_in), int(delta_embed_dim))
        self.lstm = nn.LSTM(
            input_size=int(delta_embed_dim) + int(pc_embed_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=float(dropout))
        self.delta_head = nn.Linear(int(hidden_dim), int(num_delta_in))
        self.pc_head = nn.Linear(int(hidden_dim), int(num_pc))
        self.loss_fn = nn.CrossEntropyLoss()

    def init_state(self, batch_size: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(self.num_layers, int(batch_size), self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, int(batch_size), self.hidden_dim, device=device)
        return h0, c0

    def forward(
        self,
        pcs: torch.Tensor,
        deltas: torch.Tensor,
        *,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        delta_target: Optional[torch.Tensor] = None,
        pc_target: Optional[torch.Tensor] = None,
    ):
        """Predict next-1 (single-step) logits from an observed window.

        pcs/deltas: LongTensor (B, T)
        delta_target/pc_target: LongTensor (B,) for next-1 supervision.
        """
        if state is None:
            state = self.init_state(pcs.size(0), pcs.device)

        x = torch.cat((self.pc_embed(pcs.long()), self.delta_embed(deltas.long())), dim=-1)
        out, state = self.lstm(x, state)
        last = out[:, -1, :]
        last = self.dropout(last)
        delta_logits = self.delta_head(last)
        pc_logits = self.pc_head(last)

        loss = None
        if delta_target is not None and pc_target is not None:
            loss_delta = self.loss_fn(delta_logits, delta_target.view(-1).long())
            loss_pc = self.loss_fn(pc_logits, pc_target.view(-1).long())
            loss = loss_delta + self.pc_loss_weight * loss_pc

        return delta_logits, pc_logits, state, loss

    def rollout_logits(
        self,
        pcs: torch.Tensor,
        deltas: torch.Tensor,
        *,
        steps: int,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """Autoregressively generate logits for the next `steps` time steps.

        Returns:
          delta_logits_seq: (B, steps, num_delta)
          pc_logits_seq:    (B, steps, num_pc)
        """
        steps = int(steps)

        if state is None:
            state = self.init_state(pcs.size(0), pcs.device)

        # Prime the LSTM with the observed history.
        x_hist = torch.cat((self.pc_embed(pcs.long()), self.delta_embed(deltas.long())), dim=-1)
        out, state = self.lstm(x_hist, state)
        last = out[:, -1, :]

        delta_logits_steps = []
        pc_logits_steps = []

        for i in range(steps):
            h = self.dropout(last)
            dlog = self.delta_head(h)
            plog = self.pc_head(h)
            delta_logits_steps.append(dlog)
            pc_logits_steps.append(plog)

            if i != steps - 1:
                # Greedy decode to feed next step.
                prev_delta = torch.argmax(dlog, dim=-1)
                prev_pc = torch.argmax(plog, dim=-1)

                x_step = torch.cat(
                    (
                        self.pc_embed(prev_pc.view(-1, 1)),
                        self.delta_embed(prev_delta.view(-1, 1)),
                    ),
                    dim=-1,
                )
                out, state = self.lstm(x_step, state)
                last = out[:, -1, :]

        delta_logits_seq = torch.stack(delta_logits_steps, dim=1)
        pc_logits_seq = torch.stack(pc_logits_steps, dim=1)
        return delta_logits_seq, pc_logits_seq