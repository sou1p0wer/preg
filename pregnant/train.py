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
from pregnant.models.losses.multi_task_cross_entropy import multi_task_CE_loss
from pregnant.models.losses.multi_task_focal_loss import five_task_Focal_loss
from pregnant.evaluation.metrics import compose_metric
from pregnant.utils.simple_utils import print_float, metric_log, plot_loss
from pregnant.datasets.pregnant import Pregnant
from pregnant.models.backbones.dnn_base import DNNBase
from pregnant.models.backbones.tabnet import TabNet
from pregnant.models.backbones.tabnet_my import TabTransNet
from pregnant.models.backbones.ft_transformer import FTTransformer
from pregnant.models.backbones.resnet import ResNet
from pregnant.models.backbones.mlp import MLP
from utils.log import create_logger

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="pregnant/data/bishe_data")
parser.add_argument('--save_train_loss', type=str, default="pregnant/outputs/train_loss")
parser.add_argument('--save_chpt', type=str, default="pregnant/outputs/checkpoints")
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epochs', type=int, default=5)
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
train_dataset = Pregnant(args.dataset, 'train')
# train_dataset = Pregnant(args.dataset, 'toy')
val_dataset = Pregnant(args.dataset, 'val')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
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
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
lr_scheduler = ExponentialLR(optimizer, gamma=0.95)

def train(epoch: int) -> float:
    model.train()
    loss_accum = total_count = 0

    for x, y in tqdm(train_loader, desc=f'Epoch: {epoch}'):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = five_task_Focal_loss(pred, y, device=device)
        optimizer.zero_grad()
        loss.backward()
        loss_accum += float(loss) * len(y)
        total_count += len(y)
        optimizer.step()
    return loss_accum / total_count


@torch.no_grad()
def test(loader: DataLoader) -> float:
    model.eval()
    full_pred = {i: [] for i in range(len(train_dataset.get_target_cols))} # 收集所有的pred
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


best_val_mean_auc = 0
best_val_epoch = 0
train_losses = []
for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    train_losses.append(train_loss)
    train_mean_auc, train_mean_mAP = test(train_loader)
    metric_log(train_mean_auc, train_mean_mAP, train_dataset.get_target_cols, logger, epoch, split='train', train_loss=train_loss)
    val_mean_auc, val_mean_mAP = test(val_loader)
    metric_log(val_mean_auc, val_mean_mAP, train_dataset.get_target_cols, logger, epoch, split='val')
    
    bishe_m_auc = sum(val_mean_auc) / len(val_mean_auc)
    if best_val_mean_auc < bishe_m_auc:  # 毕设用auc作为评价指标
        best_val_mean_auc = bishe_m_auc
        best_val_epoch = epoch
        torch.save(model.state_dict(), os.path.join(args.save_chpt, f'mAUC_best_{model_name}_{num_layers}.pth'))
    lr_scheduler.step()

plot_loss(train_losses, model_name, args.save_train_loss)
logger.info(f'Best Val auc: {print_float(best_val_mean_auc)} at epoch: {best_val_epoch}')