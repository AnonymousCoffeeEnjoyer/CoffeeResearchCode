"""Attention-based model for next-n prefetching (autoregressive evaluation).

Architecture:
  - Separate embeddings for PC and delta tokens, concatenated per timestep.
  - MultiheadAttention over the sequence; take the last token hidden state.
  - Two single-step output heads: delta and PC (Linear(concat_dim → num_classes)).

Training:
  - Next-1 supervision on both heads (teacher-forced).

Evaluation:
  - True autoregressive rollout for next-N via a sliding window: the greedy-
    decoded delta and PC tokens from each step are appended to their windows
    (oldest token dropped) and a fresh forward pass is run for the next step.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class EmbeddingPattention(nn.Module):
    def __init__(
        self,
        num_pc: int,
        num_delta_in: int,
        delta_embed_dim: int,
        pc_embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        pc_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.num_delta_in = int(num_delta_in)
        self.num_pc = int(num_pc)

        concat_dim = int(delta_embed_dim) + int(pc_embed_dim)
        # num_heads must divide concat_dim
        if concat_dim % int(num_heads) != 0:
            raise ValueError(
                f"concat_dim ({concat_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.pc_embed = nn.Embedding(int(num_pc), int(pc_embed_dim))
        self.delta_embed = nn.Embedding(int(num_delta_in), int(delta_embed_dim))

        self.attn = nn.MultiheadAttention(
            embed_dim=concat_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.pc_loss_weight = float(pc_loss_weight)
        self.dropout = nn.Dropout(p=float(dropout))
        self.delta_head = nn.Linear(concat_dim, int(num_delta_in))
        self.pc_head = nn.Linear(concat_dim, int(num_pc))
        self.loss_fn = nn.CrossEntropyLoss()

    def _encode(self, pcs: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """Embed and concatenate pcs/deltas -> (B, T, concat_dim)."""
        return torch.cat(
            [self.pc_embed(pcs.long()), self.delta_embed(deltas.long())], dim=-1
        )

    def forward(
        self,
        pcs: torch.Tensor,
        deltas: torch.Tensor,
        *,
        delta_target: Optional[torch.Tensor] = None,
        pc_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Next-1 prediction from an observed window.

        pcs/deltas:   LongTensor (B, T)
        delta_target: LongTensor (B,) for next-1 supervision.
        pc_target:    LongTensor (B,) for next-1 supervision.

        Returns:
            delta_logits: (B, num_delta_in)
            pc_logits:    (B, num_pc)
            loss:         scalar combined CE or None
        """
        x = self._encode(pcs, deltas)                # (B, T, D)
        x, _ = self.attn(x, x, x)                   # (B, T, D)
        last = self.dropout(x[:, -1, :])             # (B, D)
        delta_logits = self.delta_head(last)         # (B, num_delta_in)
        pc_logits = self.pc_head(last)               # (B, num_pc)

        loss = None
        if delta_target is not None and pc_target is not None:
            loss_delta = self.loss_fn(delta_logits, delta_target.view(-1).long())
            loss_pc = self.loss_fn(pc_logits, pc_target.view(-1).long())
            loss = loss_delta + self.pc_loss_weight * loss_pc

        return delta_logits, pc_logits, loss

    def rollout_logits(
        self,
        pcs: torch.Tensor,
        deltas: torch.Tensor,
        *,
        steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively generate logits for `steps` future steps.

        Strategy: sliding window. After each predicted step, the greedy-decoded
        delta and PC tokens are appended to their respective windows (oldest
        token dropped) and a fresh attention forward pass is run.

        Returns:
            delta_logits_seq: (B, steps, num_delta_in)
            pc_logits_seq:    (B, steps, num_pc)
        """
        steps = int(steps)
        delta_window = deltas.long().clone()   # (B, T)
        pc_window = pcs.long().clone()         # (B, T)

        delta_logits_steps = []
        pc_logits_steps = []

        with torch.no_grad():
            for _ in range(steps):
                x = self._encode(pc_window, delta_window)
                x, _ = self.attn(x, x, x)
                last = x[:, -1, :]
                dlog = self.delta_head(last)            # (B, num_delta_in)
                plog = self.pc_head(last)               # (B, num_pc)
                delta_logits_steps.append(dlog)
                pc_logits_steps.append(plog)

                # Greedy decode and slide both windows forward.
                pred_delta = torch.argmax(dlog, dim=-1, keepdim=True)  # (B, 1)
                pred_pc = torch.argmax(plog, dim=-1, keepdim=True)     # (B, 1)
                delta_window = torch.cat([delta_window[:, 1:], pred_delta], dim=1)
                pc_window = torch.cat([pc_window[:, 1:], pred_pc], dim=1)

        delta_logits_seq = torch.stack(delta_logits_steps, dim=1)  # (B, steps, num_delta_in)
        pc_logits_seq = torch.stack(pc_logits_steps, dim=1)        # (B, steps, num_pc)
        return delta_logits_seq, pc_logits_seq
