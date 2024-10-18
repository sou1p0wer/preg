from models.encoder.column_encoder import ColumnEncoder
from models.decode_heads.five_task_head import MultiTaskDecoderHead
from typing import Any

import torch
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Dropout,
    LayerNorm,
    Linear,
    Module,
    ReLU,
    Sequential,
)

class MLP(Module):
    r"""The light-weight MLP model that mean-pools column embeddings and
    applies MLP over it.

    Args:
        channels (int): The number of channels in the backbone layers.
        out_channels (int): The number of output channels in the decoder.
        num_layers (int): The number of layers in the backbone.
        col_stats(dict[str,Dict[:class:`torch_frame.data.stats.StatType`,Any]]):
             A dictionary that maps column name into stats.
             Available as :obj:`dataset.col_stats`.
        col_names_dict (dict[:class:`torch_frame.stype`, List[str]]): A
            dictionary that maps stype to a list of column names. The column
            names are sorted based on the ordering that appear in
            :obj:`tensor_frame.feat_dict`. Available as
            :obj:`tensor_frame.col_names_dict`.
        stype_encoder_dict
            (dict[:class:`torch_frame.stype`,
            :class:`torch_frame.nn.encoder.StypeEncoder`], optional):
            A dictionary mapping stypes into their stype encoders.
            (default: :obj:`None`, will call :obj:`EmbeddingEncoder()`
            for categorical feature and :obj:`LinearEncoder()` for
            numerical feature)
        normalization (str, optional): The type of normalization to use.
            :obj:`batch_norm`, :obj:`layer_norm`, or :obj:`None`.
            (default: :obj:`layer_norm`)
        dropout_prob (float): The dropout probability (default: `0.2`).
    """
    def __init__(
        self,
        channels: int,
        num_cols,
        num_layers: int,
        cat_idxs,
        cat_dims,
        cat_emb_channels: int = 3,
        normalization: str | None = "layer_norm",
        dropout_prob: float = 0.2,        
    ) -> None:
        super().__init__()

        # Map input tensor from (batch_size, num_cols) to (batch_size, num_cols, cat_emb_channels)
        self.feature_encoder = ColumnEncoder(
            out_channels=cat_emb_channels,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims
        )
        self.emb_col = Linear(num_cols, channels)
        self.mlp = Sequential()

        for _ in range(num_layers - 1):
            self.mlp.append(Linear(channels, channels))
            if normalization == "layer_norm":
                self.mlp.append(LayerNorm(channels))
            elif normalization == "batch_norm":
                self.mlp.append(BatchNorm1d(channels))
            self.mlp.append(ReLU())
            self.mlp.append(Dropout(p=dropout_prob))
        self.multi_decoder = MultiTaskDecoderHead(channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.feature_encoder.reset_parameters()
        self.emb_col.reset_parameters()
        for param in self.mlp:
            if hasattr(param, 'reset_parameters'):
                param.reset_parameters()
        self.multi_decoder.reset_parameters()

    def forward(self, x) -> Tensor:
        r"""Transforming :class:`TensorFrame` object into output prediction.

        Args:
            tf (TensorFrame): Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Output of shape [batch_size, out_channels].
        """
        x = self.feature_encoder(x)

        x = torch.mean(x, dim=-1)

        x = self.emb_col(x)

        out = self.mlp(x)

        out = self.multi_decoder(out)
        return out
