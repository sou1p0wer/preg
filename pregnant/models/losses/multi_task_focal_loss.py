import torch
from torch import nn

# 来自：https://blog.csdn.net/zhaohongfei_358/article/details/129108869
class BinaryFocalLoss(nn.Module):
    """
    参考 https://github.com/lonePatient/TorchBlocks
    """

    def __init__(self, gamma=4.0, alpha=0.25, epsilon=1.e-9):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, input, target):
        """
        Args:
            input: model's output, shape of [batch_size]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        multi_hot_key = target
        logits = input
        # 如果模型没有做sigmoid的话，这里需要加上
        # logits = torch.sigmoid(logits)
        zero_hot_key = 1 - multi_hot_key
        loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
        loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()
        return loss.mean()


class FocalLoss(nn.Module):
    """
    参考 https://github.com/lonePatient/TorchBlocks
    """

    def __init__(self, gamma=4.0, alpha=1, epsilon=1.e-9, device=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha, device=device)
        else:
            self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, input, target):
        """
        Args:
            input: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
            alpha=[0.1, 0.2, 0.3, 0.15, 0.25]
        Returns:
            shape of [batch_size]
        """
        num_labels = input.size(-1)
        idx = target.view(-1, 1).long()
        one_hot_key = torch.zeros(idx.size(0), num_labels, dtype=torch.float32, device=idx.device)
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        one_hot_key[:, 0] = 0  # ignore 0 index.
        logits = torch.softmax(input, dim=-1)
        loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
        loss = loss.sum(1)
        return loss.mean()

def multi_task_Focal_loss(preds, labels, device):
    alphas = [
        [0.08, 0.94, 0.98],
        [0.13, 0.90, 0.97],
        [0.07, 0.94, 0.99, 1],
        0.05,
        0.04,
        0.01,
        0.01
    ]
    total_losses = []
    # 分别处理三种任务
    for index, pred in enumerate(preds):
        if pred.shape[1] == 3 or pred.shape[1] == 4:  # 3或4分类
            total_losses.append(FocalLoss(alpha=alphas[index], device=device)(pred, labels[:, index]))
        elif pred.shape[1] == 1: # 二分类
            total_losses.append(BinaryFocalLoss(alpha=alphas[index])(pred.squeeze(1), labels[:, index]))
    return sum(total_losses) 


def five_task_Focal_loss(preds, labels, device):
    alphas = [
        0.07,
        0.05,
        0.04,
        0.01,
        0.01
    ]
    total_losses = []
    # 分别处理三种任务
    for index, pred in enumerate(preds):
        if pred.shape[1] == 3 or pred.shape[1] == 4:  # 3或4分类
            total_losses.append(FocalLoss(alpha=alphas[index], device=device)(pred, labels[:, index]))
        elif pred.shape[1] == 1: # 二分类
            total_losses.append(BinaryFocalLoss(alpha=alphas[index])(pred.squeeze(1), labels[:, index]))
    return sum(total_losses) 