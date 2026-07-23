import torch
import torch.nn as nn


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss for one projection per sample.

    Samples with the same label are positives; samples with different labels
    are negatives. Batches must contain at least two samples for any class that
    should contribute to the loss.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, projections: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if projections.ndim != 2:
            raise ValueError(f"Expected projections with shape [B, D], got {tuple(projections.shape)}")
        if labels.ndim != 1:
            raise ValueError(f"Expected labels with shape [B], got {tuple(labels.shape)}")
        if projections.size(0) != labels.size(0):
            raise ValueError("Projection and label batch sizes do not match")

        device = projections.device
        labels = labels.view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).float().to(device)

        logits_mask = torch.ones_like(positive_mask) - torch.eye(positive_mask.size(0), device=device)
        positive_mask = positive_mask * logits_mask

        logits = torch.matmul(projections, projections.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

        positives_per_anchor = positive_mask.sum(dim=1)
        valid_anchors = positives_per_anchor > 0
        if not valid_anchors.any():
            raise ValueError("Supervised contrastive loss needs at least one positive pair in the batch")

        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positives_per_anchor.clamp_min(1.0)
        return -mean_log_prob_pos[valid_anchors].mean()
