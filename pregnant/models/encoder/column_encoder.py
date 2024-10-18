import torch.nn as nn
import torch
import numpy as np

class ColumnEncoder(nn.Module):
    """
    for numerical features: stack
    for categorical features: embedding
    """
    def __init__(
            self,
            out_channels,
            cat_idxs,
            cat_dims
    ):
        super(ColumnEncoder, self).__init__()
        self.out_channels = out_channels
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims
        self.embeddings = nn.ModuleList()
        for cat_dim in cat_dims:
            self.embeddings.append(nn.Embedding(cat_dim + 1, self.out_channels))
        self.reset_parameters()

    def forward(self, x):
        # x: [batch_size, num_cols]
        bs, nun_cols = x.shape
        cols = []
        cat_feat_counter = 0
        for idx in range(x.shape[1]):
            if idx in self.cat_idxs:    # for categorical
                cols.append(
                    self.embeddings[cat_feat_counter](x[:, idx].long())
                )
                cat_feat_counter += 1
            else:   # for numerical
                cols.append(x[:, idx].unsqueeze(1).repeat(1, self.out_channels))

        return torch.cat(cols, dim=1).view(bs, nun_cols, -1)
    
    def reset_parameters(self) -> None:
        for em in self.embeddings:
            em.reset_parameters()