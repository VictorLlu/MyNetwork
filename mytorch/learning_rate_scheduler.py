import math

class LearningRateSchedular:
    def __init__(self, optim):
        self.optim = optim
    
    def get_lr(self):
        return self.optim.lr

class StepLR(LearningRateSchedular):
    def __init__(self, optim, lr_drop=0.5, epochs_drop=1):
        super().__init__(optim)
        self.epochs_drop = epochs_drop
        self.drop = lr_drop
    
    def step(self, epoch):
        lr = self.optim.base_lr * math.pow(self.drop, (epoch - 1) // self.epochs_drop)
        self.optim.lr = lr