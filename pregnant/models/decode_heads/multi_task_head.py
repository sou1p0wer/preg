import torch
import torch.nn as nn
from utils.simple_utils import reset_parameters_soft

class MultiTaskDecoderHead(nn.Module):
    def __init__(self, num_shared_features):
        super(MultiTaskDecoderHead, self).__init__()

        self.multi_task_module = nn.ModuleList()
        # 3分类任务
        for i in range(2):
            three_labels_module = nn.Sequential(
                nn.Linear(num_shared_features, num_shared_features),
                nn.ReLU(),
                nn.Linear(num_shared_features, 3),
                nn.Softmax(dim=1)
            )
            self.multi_task_module.append(three_labels_module)

        # 4分类任务
        four_labels_task = nn.Sequential(
            nn.Linear(num_shared_features, num_shared_features),
            nn.ReLU(),
            nn.Linear(num_shared_features, 4),
            nn.Softmax(dim=1)
        )
        self.multi_task_module.append(four_labels_task)

        # 2分类任务
        for i in range(4):
            binary_labels_task = nn.Sequential(
                nn.Linear(num_shared_features, num_shared_features),
                nn.ReLU(),
                nn.Linear(num_shared_features, 1),
                nn.Sigmoid()
            )
            self.multi_task_module.append(binary_labels_task)
        
        self.reset_parameters()

    def forward(self, shared_features):
        # shared_features： [bs, sl]
        out = []
        for task in self.multi_task_module:
            out.append(task(shared_features))

        return out  # shape: [num_task, bs, num_class]

    def reset_parameters(self) -> None:
        for task in self.multi_task_module:
            for m in task:
                reset_parameters_soft(m)