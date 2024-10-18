import torch.nn as nn
import torch
from models.decode_heads.multi_task_head import MultiTaskDecoderHead
from models.encoder.column_encoder import ColumnEncoder

class DNNBase(nn.Module):
    def __init__(self, input_size, share_size, outsize, num_cols, cat_idxs, cat_dims, cat_emb_dim=3):
        super(DNNBase, self).__init__()
        self.embedding = ColumnEncoder(cat_emb_dim, cat_idxs, cat_dims)
        self.sharedlayer = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, share_size),
            nn.ReLU(),
            nn.Dropout()
        )

        self.out_linear = nn.Linear(share_size, outsize)
        self.decoder = MultiTaskDecoderHead(outsize, num_cols)
   

    def forward(self, x):
        x = self.embedding(x)
        h_shared = self.sharedlayer(x)
        out = self.out_linear(h_shared)
        out = self.decoder(out)
        return out
    
if __name__ == '__main__':
    inputs = torch.randn((4096, 167, 20))
    model = DNNBase(20, 256, 128)
    x = model(inputs)
    print(x)