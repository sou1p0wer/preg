import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F

from datetime import datetime
import numpy as np
from tqdm import tqdm
import argparse
import sys
sys.path.append('/vepfs-sha/xiezixun/high_risk_pregnant/pregnant')
import os, random
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from evaluation.metrics import compose_metric
from utils.simple_utils import metric_log
from datasets.pregnant import Pregnant
from models.backbones.dnn_base import DNNBase
from models.backbones.tabnet import TabNet
from models.backbones.ft_transformer import FTTransformer
from models.backbones.tabnet_my import TabTransNet
from models.backbones.resnet import ResNet
from models.backbones.mlp import MLP
from models.backbones.tabnet import TabNet

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="/vepfs-sha/xiezixun/high_risk_pregnant/pregnant/data/bishe_data")
# parser.add_argument('--ckpt_path', type=str, default="pregnant/outputs/checkpoints/mAUC_best_TabNet_my_12.pth")
parser.add_argument('--ckpt_path', type=str, default="/vepfs-sha/xiezixun/high_risk_pregnant/pregnant/outputs/saved_result/tabnet_my_transformer加在选择那/mAUC_best_TabNet_my_6.pth")
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--compile', action='store_true')
args = parser.parse_args()

# model parameter
batch_size=128 # 选择部分
num_layers=6
channels=128
gamma=1.2
model_name='TabNet_my'


seed_everything(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# Set up data

train_dataset = test_dataset = Pregnant(args.dataset, 'test')

test_loader = DataLoader(test_dataset, batch_size=batch_size)
device_ids = [0, 1]

# Set up model and optimizer
model = TabTransNet(
    device=device,
    num_cols=len(train_dataset.get_train_cols),
    cat_idxs=train_dataset.get_categorical_columns_idxs,
    cat_dims=train_dataset.get_categorical_dims_idxs,
    cat_emb_channels=3,
    num_layers=num_layers,
    split_attn_channels=channels,
    split_feat_channels=channels,
    gamma=gamma,
    is_visualization=True
).to(device_ids[0])

# model = nn.DataParallel(model, device_ids=device_ids)
model = torch.compile(model, dynamic=True) if args.compile else model 
state_dict = torch.load(args.ckpt_path)
state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
new_state_dict = {}
for key, value in state_dict.items():
    # 将 'multi_task_' 替换为 'multi_task_module.'
    new_key = key.replace('multi_task_', 'multi_task_module.')
    # 将修改后的键和值添加到新的 state_dict 中
    new_state_dict[new_key] = value
model.load_state_dict(new_state_dict)

@torch.no_grad()
def test(loader: DataLoader) -> float:
    model.eval()
    full_pred = {i: [] for i in range(len(test_dataset.get_target_cols))} # 收集所有的pred
    full_y = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred, visualization = model(x)
        for index, p in enumerate(pred): full_pred[index].append(p) 
        full_y.append(y)
    pred = [torch.cat(full_pred[k], dim=0) for k in full_pred]
    y = torch.cat(full_y, dim=0)
    mean_auc, mean_mAP = compose_metric(pred, y)
    return mean_auc, mean_mAP, visualization

test_mean_auc, test_mean_mAP, visualization = test(test_loader)

for i in range(len(visualization)):
    visualization[i] = torch.cat(visualization[i], dim=0)

visualization = torch.stack(list(visualization.values()), dim=0).detach().cpu()
visualization = visualization[[0, 1, 3, 4, 5]]
bs = 200

reshaped_tensor = visualization.view(visualization.shape[0], visualization.shape[1], -1, 3)
averaged_tensor = torch.mean(reshaped_tensor, dim=3)
input_tensor = torch.softmax(averaged_tensor, dim=2)    # 后续用于计算最高行
agg = torch.mean(input_tensor, dim=0)
probabilities = torch.cat([agg.unsqueeze(0), input_tensor], dim=0)

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.gca().tick_params(axis='both', which='both', direction='in')

fig, axes = plt.subplots(2, 3, figsize=(14, 13))  # 创建一个 2x3 的子图布局，用于绘制6个概率矩阵
plt.subplots_adjust(hspace=0.00000001)  # 调整子图的上下间距
for i, ax in enumerate(axes.flat):
    if i == 0:
        ax.imshow(probabilities[i, :bs], cmap='gray')
        ax.set_xlabel('Number of Columns')
        ax.set_ylabel('Batch Size')
        ax.set_title(f'M Mean')        
    else:
        ax.imshow(probabilities[i, :bs], cmap='gray')
        ax.set_xlabel('Number of Columns')
        ax.set_ylabel('Batch Size')
        ax.set_title(f'M[{i}]')

plt.savefig('可视化1.png')

# 打印综合最高的几列
# max_col = torch.argmax(probabilities[1], dim=1)
# unique_values, inverse_indices, counts = max_col.unique(return_inverse=True, return_counts=True)
# print(unique_values)
# print(inverse_indices)
# print(counts)
print(probabilities[1, 17])
print(probabilities[1, 17].unique())
"""

unique_values
tensor([ 38,  41,  43,  62,  64,  65,  66,  68,  71,  76,  77, 104, 129, 132,
        169])
counts
tensor([    2,  1476,     8,     4,     4,     4,     6,     2,    10,     2,
           16,    64,   856,     2, 29278])

"""