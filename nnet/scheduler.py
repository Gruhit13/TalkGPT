from torch import optim
import numpy as np

class TransformerLRScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: optim, d_model: int, warmup_steps: int):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.init_lr = np.power(self.d_model, -0.5)
        
        super(TransformerLRScheduler, self).__init__(self.optimizer)
    
    def get_lr(self):
        new_lr = np.min([
            np.power(self._step_count, -0.5),
            np.power(self.warmup_steps, -1.5) * self._step_count
        ])
        
        new_lr = self.init_lr * new_lr
        return [new_lr]