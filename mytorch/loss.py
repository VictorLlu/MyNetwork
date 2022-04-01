from .nn import logsoftmax

def CrossEntropyLoss(input, target):
    x = logsoftmax(input, dim=1)
    prob = (x * target).sum(1)
    loss = -(prob / target.shape[0]).sum()
    return loss