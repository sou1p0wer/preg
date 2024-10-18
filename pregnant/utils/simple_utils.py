import torch.nn as nn
import os
import matplotlib.pyplot as plt
from datetime import datetime

def print_float(num, reserve=4):
    # 返回保留四位小数，然后乘以100的值
    return ("{:." + str(reserve - 2) + "f}").format(round(num, reserve) * 100)


def reset_parameters_soft(module: nn.Module):
    r"""Call reset_parameters() only when it exists. Skip activation module."""
    if hasattr(module, "reset_parameters") and callable(
            module.reset_parameters):
        module.reset_parameters()

def create_file_if_not_exists(file_path):
    """
    如果文件路径不存在文件，则创建文件路径
    
    参数:
    - file_path: 文件路径
    """
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

# 日志
def metric_log(mean_auc, mean_mAP, target_cols, logger, epoch, split, train_loss=None):
    logger.info(f'{split} =========== {epoch} =========== ')
    if split == 'train':
        logger.info(f'Train Loss: {train_loss:.4f}')
    for index, col in enumerate(target_cols):
        logger.info(f"{col}: auc = {print_float(mean_auc[index])}, mAP = {print_float(mean_mAP[index], 6)}")
    mean_auc = sum(mean_auc) / len(mean_auc)
    mean_mAP = sum(mean_mAP) / len(mean_mAP)
    logger.info(f"{split}_mean_auc = {print_float(mean_auc)}, {split}_mean_mAP = {print_float(mean_mAP, 6)}")

# 画图
def plot_loss(loss_values, model_name, save_path=None):
    """
    绘制损失函数变化图并保存到指定路径
    
    参数:
    - loss_values: 训练过程中的损失函数值列表
    - save_path: 要保存图像的路径（可选）
    """
    epochs = range(1, len(loss_values) + 1)
    
    plt.plot(epochs, loss_values, 'b', label='Training Loss')
    plt.title('Loss Function - Training Progress')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 如果提供了保存路径，则保存图像
    if save_path:
        # 确保保存目录存在
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存图像到指定路径
        plt.savefig(os.path.join(save_path, f"{model_name}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}_loss.png"))
    else:
        plt.show()