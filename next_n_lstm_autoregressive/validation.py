from __future__ import annotations

from typing import Optional

import torch
try:
    from .config import config
    from .label_encoder import label_encoder_deltas
except ImportError:
    from config import config
    from label_encoder import label_encoder_deltas


def _delta_id_to_value_table(device: torch.device) -> Optional[torch.Tensor]:
    """Return a lookup table mapping encoded delta IDs -> numeric delta values.

    Requires `label_encoder_deltas` to be fitted (classes_ populated).
    """
    classes = getattr(label_encoder_deltas, "classes_", None)
    if classes is None:
        return None
    if len(classes) == 0:
        return None
    table = torch.as_tensor(classes, device=device)
    # Ensure integer dtype (deltas are numeric offsets).
    if table.dtype not in (torch.int32, torch.int64):
        table = table.to(torch.long)
    return table


def _cumsum_delta_hits(delta_logits: torch.Tensor, delta_targets: torch.Tensor):
    """Absolute (cumsum) correctness for delta predictions.

    We decode token IDs back to their numeric delta values, take cumulative sums
    across the horizon, and compare predicted vs ground-truth cumulative offsets.

    Returns (hits_next1, total_next1, hits_all, total_all).
    """
    if delta_logits is None or delta_targets is None or delta_logits.numel() == 0:
        return 0, 0, 0, 0

    b, h, _c = delta_logits.shape
    if b == 0 or h == 0:
        return 0, 0, 0, 0

    table = _delta_id_to_value_table(delta_logits.device)
    if table is None:
        return 0, 0, 0, 0

    pred_ids = torch.argmax(delta_logits, dim=-1).long()  # (B, H)
    true_ids = delta_targets[:, :h].long()

    pred_vals = table[pred_ids]  # (B, H)
    true_vals = table[true_ids]  # (B, H)

    pred_cum = torch.cumsum(pred_vals, dim=1)
    true_cum = torch.cumsum(true_vals, dim=1)

    matches = pred_cum.eq(true_cum)
    hits_all = int(matches.sum().item())
    hits_next1 = int(matches[:, 0].sum().item())
    total_all = int(matches.numel())
    total_next1 = int(matches.size(0))
    return hits_next1, total_next1, hits_all, total_all


def _topk_hits(logits: torch.Tensor, targets: torch.Tensor, k: int):
    """Return (hits_next1, total_next1, hits_all, total_all)."""
    if logits is None or targets is None:
        return 0, 0, 0, 0
    b, h, c = logits.shape
    if b == 0 or h == 0:
        return 0, 0, 0, 0

    labels = targets[:, :h].long()

    k_eff = min(max(1, int(k)), int(c))
    topk = torch.topk(logits, k=k_eff, dim=-1, sorted=True).indices
    matches = topk.eq(labels.unsqueeze(-1))

    hits_all = int(matches.any(dim=-1).sum().item())
    hits_next1 = int(matches[:, 0, :].any(dim=-1).sum().item())
    total_all = int(labels.numel())
    total_next1 = int(labels.size(0))
    return hits_next1, total_next1, hits_all, total_all


def _ce_next1(logits: torch.Tensor, targets: torch.Tensor, loss_fn):
    if logits is None or targets is None or logits.numel() == 0:
        return None
    return loss_fn(logits[:, 0, :], targets[:, 0].long())


def evaluate_model(network, data_iterator, computing_device="cpu"):
    """Evaluate with a true autoregressive rollout.

    Returns:
    losses: {next1_delta, next1_pc, next1_total}
      acc: dict with required next1/nextn top1/topK for delta + PC.
    """
    network.eval()
    horizon = int(config["next_n_predict"])
    top_k = int(config.get("top_k_pred", 1))

    # loss sums
    sum_d1 = 0.0
    sum_p1 = 0.0
    sum_t1 = 0.0
    count_next1 = 0

    # accuracy hit sums
    hits = {
        "delta_next1_top1": 0,
        "delta_next1_topk": 0,
        "delta_nextn_top1": 0,
        "delta_nextn_topk": 0,
        # Absolute (cumsum) delta accuracy using decoded numeric deltas.
        "delta_abs_next1_top1": 0,
        "delta_abs_nextn_top1": 0,
        "pc_next1_top1": 0,
        "pc_next1_topk": 0,
        "pc_nextn_top1": 0,
        "pc_nextn_topk": 0,
    }
    totals_next1 = 0
    totals_nextn = 0

    with torch.no_grad():
        for batch in data_iterator:
            pcs, deltas, delta_targets, pc_targets = [t.to(computing_device) for t in batch]
            bsz = int(pcs.size(0))
            if bsz == 0:
                continue

            delta_logits, pc_logits = network.rollout_logits(pcs, deltas, steps=horizon)

            # Loss: next-1 only (delta, pc, weighted total)
            ce_d1 = _ce_next1(delta_logits, delta_targets, network.loss_fn)
            ce_p1 = _ce_next1(pc_logits, pc_targets, network.loss_fn)

            if ce_d1 is not None and ce_p1 is not None:
                total_next1 = ce_d1 + network.pc_loss_weight * ce_p1
                sum_d1 += float(ce_d1.item()) * bsz
                sum_p1 += float(ce_p1.item()) * bsz
                sum_t1 += float(total_next1.item()) * bsz
                count_next1 += bsz

            # Accuracy: delta
            d1h1, d1t, dnh1, dnt = _topk_hits(delta_logits, delta_targets, k=1)
            d1hk, _, dnhk, _ = _topk_hits(delta_logits, delta_targets, k=top_k)
            hits["delta_next1_top1"] += d1h1
            hits["delta_next1_topk"] += d1hk
            hits["delta_nextn_top1"] += dnh1
            hits["delta_nextn_topk"] += dnhk

            # Accuracy: delta absolute (cumulative sum of decoded deltas)
            a1h1, _a1t, anh1, _ant = _cumsum_delta_hits(delta_logits, delta_targets)
            hits["delta_abs_next1_top1"] += a1h1
            hits["delta_abs_nextn_top1"] += anh1

            # Accuracy: pc
            p1h1, _p1t, pnh1, _pnt = _topk_hits(pc_logits, pc_targets, k=1)
            p1hk, _, pnhk, _ = _topk_hits(pc_logits, pc_targets, k=top_k)
            hits["pc_next1_top1"] += p1h1
            hits["pc_next1_topk"] += p1hk
            hits["pc_nextn_top1"] += pnh1
            hits["pc_nextn_topk"] += pnhk

            totals_next1 += d1t
            totals_nextn += dnt

    losses = {
        "next1_delta": (sum_d1 / count_next1) if count_next1 else 0.0,
        "next1_pc": (sum_p1 / count_next1) if count_next1 else 0.0,
        "next1_total": (sum_t1 / count_next1) if count_next1 else 0.0,
    }

    def _safe_div(a: int, b: int) -> float:
        return float(a) / float(b) if b else 0.0

    # Use a stricter key match than substring "next1" to avoid accidental matches.
    acc = {k: _safe_div(hits[k], totals_next1 if "_next1_" in k else totals_nextn) for k in hits}
    return losses, acc


def print_eval_summary(title: str, losses: dict, acc: dict, top_k: int) -> None:
    top_k = int(top_k)
    print(title)
    print(
        "Loss next1: "
        f"delta={float(losses.get('next1_delta', 0.0)):.6f} "
        f"pc={float(losses.get('next1_pc', 0.0)):.6f} "
        f"total={float(losses.get('next1_total', 0.0)):.6f}"
    )
    print(
        "Delta accuracy: "
        f"next1 top1={acc.get('delta_next1_top1', 0.0):.4f} "
        f"top{top_k}={acc.get('delta_next1_topk', 0.0):.4f} | "
        f"nextN top1={acc.get('delta_nextn_top1', 0.0):.4f} "
        f"top{top_k}={acc.get('delta_nextn_topk', 0.0):.4f}"
    )
    print(
        "Delta absolute accuracy: "
        f"next1 top1={acc.get('delta_abs_next1_top1', 0.0):.4f} | "
        f"nextN top1={acc.get('delta_abs_nextn_top1', 0.0):.4f}"
    )
    print(
        "PC acc:    "
        f"next1 top1={acc.get('pc_next1_top1', 0.0):.4f} "
        f"top{top_k}={acc.get('pc_next1_topk', 0.0):.4f} | "
        f"nextN top1={acc.get('pc_nextn_top1', 0.0):.4f} "
        f"top{top_k}={acc.get('pc_nextn_topk', 0.0):.4f}"
    )
