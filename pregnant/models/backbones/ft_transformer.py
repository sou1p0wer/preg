from models.encoder.column_encoder import ColumnEncoder
from models.decode_heads.five_task_head import MultiTaskDecoderHead
import torch.nn as nn
import torch
from torch import Tensor
from torch.nn import (
    LayerNorm,
    Parameter,
    TransformerEncoder,
    TransformerEncoderLayer,
    Linear,
    ReLU,
    Sequential
)

from typing import Any, Optional, Tuple

from torch import Tensor

class FTTransformer(nn.Module):
    r"""The FT-Transformer model introduced in the
    `"Revisiting Deep Learning Models for Tabular Data"
    <https://arxiv.org/abs/2106.11959>`_ paper.

    .. note::

        For an example of using FTTransformer, see `examples/revisiting.py
        <https://github.com/pyg-team/pytorch-frame/blob/master/examples/
        revisiting.py>`_.

    Args:
        channels (int): Hidden channel dimensionality
        out_channels (int): Output channels dimensionality
        num_layers (int): Number of layers.  (default: :obj:`3`)
        col_stats(dict[str,dict[:class:`torch_frame.data.stats.StatType`,Any]]):
             A dictionary that maps column name into stats.
             Available as :obj:`dataset.col_stats`.
        col_names_dict (dict[:obj:`torch_frame.stype`, list[str]]): A
            dictionary that maps stype to a list of column names. The column
            names are sorted based on the ordering that appear in
            :obj:`tensor_frame.feat_dict`. Available as
            :obj:`tensor_frame.col_names_dict`.
    """
    def __init__(
        self,
        channels: int,
        num_layers: int,
        cat_idxs,
        cat_dims,
        cat_emb_channels: int = 3,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer (got {num_layers})")

        # Map input tensor from (batch_size, num_cols) to (batch_size, num_cols, cat_emb_channels)
        self.feature_encoder = ColumnEncoder(
            out_channels=cat_emb_channels,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims
        )
        self.emb_linear = nn.Linear(cat_emb_channels, channels)
        self.backbone = FTTransformerConvs(channels=channels,
                                           num_layers=num_layers)
        self.multi_decoder = MultiTaskDecoderHead(channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.feature_encoder.reset_parameters()
        self.backbone.reset_parameters()
        self.multi_decoder.reset_parameters()

    def forward(self, x):
        # [batch_size, num_cols, cat_emb_channels]
        x = self.feature_encoder(x)
        x = self.emb_linear(x)  # [bs, n, dim]
        x, x_cls = self.backbone(x)
        out = self.multi_decoder(x_cls.squeeze())
        return out

class FTTransformerConvs(nn.Module):
    r"""The FT-Transformer backbone in the
    `"Revisiting Deep Learning Models for Tabular Data"
    <https://arxiv.org/abs/2106.11959>`_ paper.

    This module concatenates a learnable CLS token embedding :obj:`x_cls` to
    the input tensor :obj:`x` and applies a multi-layer Transformer on the
    concatenated tensor. After the Transformer layer, the output tensor is
    divided into two parts: (1) :obj:`x`, corresponding to the original input
    tensor, and (2) :obj:`x_cls`, corresponding to the CLS token tensor.

    Args:
        channels (int): Input/output channel dimensionality
        feedforward_channels (int, optional): Hidden channels used by
            feedforward network of the Transformer model. If :obj:`None`, it
            will be set to :obj:`channels` (default: :obj:`None`)
        num_layers (int): Number of transformer encoder layers. (default: 3)
        nhead (int): Number of heads in multi-head attention (default: 8)
        dropout (int): The dropout value (default: 0.1)
        activation (str): The activation function (default: :obj:`relu`)
    """
    def __init__(
        self,
        channels: int,
        feedforward_channels: Optional[int] = None,
        # Arguments for Transformer
        num_layers: int = 3,
        nhead: int = 8,
        dropout: float = 0.2,
        activation: str = 'relu',
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model=channels,
            nhead=nhead,
            dim_feedforward=feedforward_channels or channels,
            dropout=dropout,
            activation=activation,
            # Input and output tensors are provided as
            # [batch_size, seq_len, channels]
            batch_first=True,
        )
        encoder_norm = LayerNorm(channels)
        self.transformer = TransformerEncoder(encoder_layer=encoder_layer,
                                              num_layers=num_layers,
                                              norm=encoder_norm)
        self.cls_embedding = Parameter(torch.empty(channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.cls_embedding, std=0.01)
        for p in self.transformer.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        r"""CLS-token augmented Transformer convolution.

        Args:
            x (Tensor): Input tensor of shape [batch_size, num_cols, channels]

        Returns:
            (torch.Tensor, torch.Tensor): (Output tensor of shape
            [batch_size, num_cols, channels] corresponding to the input
            columns, Output tensor of shape [batch_size, channels],
            corresponding to the added CLS token column.)
        """
        B, _, _ = x.shape
        # [batch_size, num_cols, channels]
        x_cls = self.cls_embedding.repeat(B, 1, 1)
        # [batch_size, num_cols + 1, channels]
        x_concat = torch.cat([x_cls, x], dim=1)
        # [batch_size, num_cols + 1, channels]
        x_concat = self.transformer(x_concat)
        x_cls, x = x_concat[:, 0, :], x_concat[:, 1:, :]    # bs, 1, channels
        return x, x_cls