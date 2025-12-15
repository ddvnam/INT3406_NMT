# src/training/scheduler.py
import math

class TransformerScheduler:
    """Learning rate scheduler for Transformer"""
    def __init__(self, optimizer, warmup_steps: int, lr_base: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.lr_base = lr_base
        self.step_num = 0
    
    def step(self):
        """Update learning rate"""
        self.step_num += 1
        
        if self.step_num <= self.warmup_steps:
            lr = self.lr_base * self.step_num / self.warmup_steps
        else:
            lr = self.lr_base * math.sqrt(self.warmup_steps) / math.sqrt(self.step_num)
        
        for group in self.optimizer.param_groups:
            group["lr"] = lr
    
    def get_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]["lr"]