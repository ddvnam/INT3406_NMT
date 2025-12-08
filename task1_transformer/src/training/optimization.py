import torch
import math
from typing import Dict, Any
from src.model.transformer import Transformer


class ScheduledOptim:
    """Optimizer với lịch trình learning rate theo công thức trong paper Transformer"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, init_lr: float, 
                 d_model: int, n_warmup_steps: int):
        """
        Args:
            optimizer: Optimizer gốc (Adam)
            init_lr: Learning rate khởi tạo
            d_model: Dimension của model
            n_warmup_steps: Số bước warmup
        """
        self._optimizer = optimizer
        self.init_lr = init_lr
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0
    
    def step_and_update_lr(self):
        """Thực hiện bước optimizer và cập nhật learning rate"""
        self._update_learning_rate()
        self._optimizer.step()
    
    def zero_grad(self):
        """Xóa gradients"""
        self._optimizer.zero_grad()
    
    def _get_lr_scale(self) -> float:
        """Tính hệ số scaling cho learning rate"""
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))
    
    def _update_learning_rate(self):
        """Cập nhật learning rate theo công thức Transformer"""
        self.n_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_current_lr(self) -> float:
        """Lấy learning rate hiện tại"""
        return self._optimizer.param_groups[0]['lr']
    
    def state_dict(self) -> Dict[str, Any]:
        """Lưu state của optimizer"""
        optimizer_state_dict = {
            'init_lr': self.init_lr,
            'd_model': self.d_model,
            'n_warmup_steps': self.n_warmup_steps,
            'n_steps': self.n_steps,
            '_optimizer': self._optimizer.state_dict(),
        }
        return optimizer_state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Tải state của optimizer"""
        self.init_lr = state_dict['init_lr']
        self.d_model = state_dict['d_model']
        self.n_warmup_steps = state_dict['n_warmup_steps']
        self.n_steps = state_dict['n_steps']
        self._optimizer.load_state_dict(state_dict['_optimizer'])

def create_optimizer(model: torch.nn.Module, d_model: int = 512, 
                    warmup_steps: int = 4000, init_lr: float = 0.2,
                    betas: tuple = (0.9, 0.98), eps: float = 1e-09) -> ScheduledOptim:
    """
    Hàm helper để tạo optimizer cho Transformer
    
    Args:
        model: Mô hình Transformer
        d_model: Dimension của model
        warmup_steps: Số bước warmup
        init_lr: Learning rate khởi tạo
        betas: Beta parameters cho Adam
        eps: Epsilon cho Adam
        
    Returns:
        ScheduledOptim: Optimizer với lịch trình learning rate
    """
    optimizer = torch.optim.Adam(model.parameters(), betas=betas, eps=eps)
    return ScheduledOptim(optimizer, init_lr, d_model, warmup_steps)


def get_optimizer_params(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Lấy thông tin parameters của optimizer
    
    Returns:
        Dict chứa thông tin về parameters
    """
    params_info = {
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'param_shapes': {name: list(p.shape) for name, p in model.named_parameters()}
    }
    return params_info


def test_optimizer():
    # test model with few src and trg vocab sizes
    model = Transformer(src_vocab_size=100, trg_vocab_size=100, d_model=512, N=2, heads=8, dropout=0.1)
    # get the total params of the model
    total_params = get_optimizer_params(model)
    print("Model total params:", total_params['total_params'])
    print("Model trainable params:", total_params['trainable_params'])
    optimizer = create_optimizer(model)
    for step in range(1, 10001):
        optimizer.step_and_update_lr()
        if step % 1000 == 0:
            print(f"Step {step}: Learning Rate = {optimizer.get_current_lr():.6f}")

if __name__ == "__main__":
    test_optimizer()