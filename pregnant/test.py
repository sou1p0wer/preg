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
from pregnant.evaluation.metrics import compose_metric
from pregnant.utils.simple_utils import metric_log
from pregnant.datasets.pregnant import Pregnant
from pregnant.models.backbones.dnn_base import DNNBase
from pregnant.models.backbones.tabnet import TabNet
from pregnant.models.backbones.ft_transformer import FTTransformer
from pregnant.models.backbones.tabnet_my import TabTransNet
from pregnant.models.backbones.resnet import ResNet
from pregnant.models.backbones.mlp import MLP
from utils.log import create_logger
from pregnant.models.backbones.tabnet import TabNet

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="pregnant/data/bishe_data")
# parser.add_argument('--ckpt_path', type=str, default="pregnant/outputs/checkpoints/mAUC_best_TabNet_my_12.pth")
parser.add_argument('--ckpt_path', type=str, default="pregnant/outputs/saved_bishe/tabnet_my_transformer加在选择那/mAUC_best_TabNet_my_6.pth")
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--compile', action='store_true')
args = parser.parse_args()

# model parameter
batch_size=2048 # 选择部分
num_layers=6
channels=128
gamma=1.2
model_name='TabNet_my'


seed_everything(args.seed)
logger = create_logger()
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

model = nn.DataParallel(model, device_ids=device_ids)
model = torch.compile(model, dynamic=True) if args.compile else model 
model.load_state_dict(torch.load(args.ckpt_path))

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
    mean_auc, mean_mAP = compose_metric(pred, y)
    return mean_auc, mean_mAP

test_mean_auc, test_mean_mAP = test(test_loader)
metric_log(test_mean_auc, test_mean_mAP, test_dataset.get_target_cols, logger, epoch=0, split='test')
