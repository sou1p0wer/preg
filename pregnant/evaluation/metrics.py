from sklearn import metrics
import numpy as np

## AUC
# :y_true       # shape: (bs, )
# :y_pred       # shape: (bs, num_class) for multiclass; (bs, ) for binary
# def _metirc_auc(y_true, y_pred):
#     if y_pred.shape[1] == 1:  # 2分类修改shape
#         y_pred = y_pred.squeeze(1)
#     return metrics.roc_auc_score(y_true, y_pred, multi_class='ovo')
def _metirc_auc(y_true, y_pred):
    if y_pred.shape[1] == 1:  # 2分类修改shape
        y_pred = y_pred.squeeze(1)
    auc = metrics.roc_auc_score(y_true, y_pred, multi_class='ovo')
    return auc if auc > 0.5 else 1 - auc 


## mAP
# :y_true       # shape: (bs, )
# :y_pred       # shape: (bs, num_class) for multiclass; (bs, ) for binary
def _metric_mAP(y_true, y_pred):
    if y_pred.shape[1] == 1:  # 2分类修改shape
        y_pred = y_pred.squeeze(1)
    return metrics.average_precision_score(y_true, y_pred)

# 组装整体评价指标
def compose_metric(preds, labels):
    mean_auc = []
    mean_mAP = []
    labels = labels.to('cpu')
    for index, pred in enumerate(preds):
        pred = pred.to('cpu')
        mean_auc.append(_metirc_auc(labels[:, index], pred))
        mean_mAP.append(_metric_mAP(labels[:, index], pred))
    return mean_auc, mean_mAP

