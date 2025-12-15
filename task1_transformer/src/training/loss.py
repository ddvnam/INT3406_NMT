import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothedCrossEntropyLoss(nn.Module):
    """Label smoothed cross entropy loss"""
    def __init__(self, smoothing: float = 0.1, ignore_index: int = 0):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        vocab_size = logits.size(-1)
        
        # Log softmax
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Mask for padding
        mask = targets != self.ignore_index
        
        # Safe targets for gather
        targets_safe = targets.masked_fill(~mask, 0)
        
        # NLL loss for hard targets
        nll_loss = -log_probs.gather(dim=-1, index=targets_safe.unsqueeze(1)).squeeze(1)
        
        # Smoothing loss
        smooth_loss = -log_probs.sum(dim=-1)
        
        # Combine
        eps_i = self.smoothing / (vocab_size - 1)
        loss = (1.0 - self.smoothing - eps_i) * nll_loss + eps_i * smooth_loss
        
        # Apply mask
        loss = loss.masked_fill(~mask, 0.0)
        
        return loss.sum() / mask.sum().clamp(min=1)