import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F

from datetime import datetime
import numpy as np
from tqdm import tqdm
import argparse

import os, random
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import sys
sys.path.append('/vepfs-sha/xiezixun/high_risk_pregnant/pregnant')
from evaluation.metrics import compose_metric
from utils.simple_utils import metric_log
from datasets.pregnant import Pregnant
from models.backbones.dnn_base import DNNBase
from models.backbones.tabnet import TabNet
from models.backbones.ft_transformer import FTTransformer
from models.backbones.tabnet_my import TabTransNet
from models.backbones.resnet import ResNet
from models.backbones.mlp import MLP
from utils.log import create_logger
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
batch_size=2048
num_layers=6
channels=128
gamma=1.2
model_name='TabNet_my'



seed_everything(args.seed)
logger = create_logger('/vepfs-sha/xiezixun/high_risk_pregnant/pregnant/outputs/debug')
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
    gamma=gamma
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
        pred = model(x)
        for index, p in enumerate(pred): full_pred[index].append(p) 
        full_y.append(y)
    pred = [torch.cat(full_pred[k], dim=0) for k in full_pred]
    y = torch.cat(full_y, dim=0)
    return pred, y

preds, y = test(test_loader)
y = y.to('cpu')
target_id = ['premature', 'low_BW', 'macrosomia', 'death', 'malformation']
for idx, pred in enumerate(preds):
    pred = pred.to('cpu')
    fpr, tpr, _ = roc_curve(y[:, idx], pred.squeeze())
    model_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='{}(AUC = {:.2f})'.format(target_id[idx], model_auc))


# 添加图例和标题
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.gca().tick_params(axis='both', which='both', direction='in')

plt.legend()
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# 显示图形
plt.savefig('auc.png')