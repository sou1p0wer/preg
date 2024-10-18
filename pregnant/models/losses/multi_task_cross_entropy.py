import torch
import torch.nn.functional as F

def multi_task_CE_loss(preds, labels):
    total_losses = []
    # 分别处理三种任务
    for index, pred in enumerate(preds):
        if pred.shape[1] == 3 or pred.shape[1] == 4:  # 3或4分类
            total_losses.append(F.cross_entropy(pred, labels[:, index]))
        elif pred.shape[1] == 1: # 二分类
            total_losses.append(F.binary_cross_entropy_with_logits(pred.squeeze(1), labels[:, index].to(torch.float32)))
    return sum(total_losses) 