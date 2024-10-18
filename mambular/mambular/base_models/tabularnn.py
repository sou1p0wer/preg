import torch
import torch.nn as nn
from ..arch_utils.mlp_utils import MLP
from ..configs.tabularnn_config import DefaultTabulaRNNConfig
from .basemodel import BaseModel
from ..arch_utils.embedding_layer import EmbeddingLayer
from ..arch_utils.normalization_layers import (
    RMSNorm,
    LayerNorm,
    LearnableLayerScaling,
    BatchNorm,
    InstanceNorm,
    GroupNorm,
)


class TabulaRNN(BaseModel):
    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes=1,
        config: DefaultTabulaRNNConfig = DefaultTabulaRNNConfig(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)
        self.pooling_method = self.hparams.get("pooling_method", config.pooling_method)
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info

        norm_layer = self.hparams.get("norm", config.norm)
        if norm_layer == "RMSNorm":
            self.norm_f = RMSNorm(
                self.hparams.get("dim_feedforward", config.dim_feedforward)
            )
        elif norm_layer == "LayerNorm":
            self.norm_f = LayerNorm(
                self.hparams.get("dim_feedforward", config.dim_feedforward)
            )
        elif norm_layer == "BatchNorm":
            self.norm_f = BatchNorm(
                self.hparams.get("dim_feedforward", config.dim_feedforward)
            )
        elif norm_layer == "InstanceNorm":
            self.norm_f = InstanceNorm(
                self.hparams.get("dim_feedforward", config.dim_feedforward)
            )
        elif norm_layer == "GroupNorm":
            self.norm_f = GroupNorm(
                1, self.hparams.get("dim_feedforward", config.dim_feedforward)
            )
        elif norm_layer == "LearnableLayerScaling":
            self.norm_f = LearnableLayerScaling(
                self.hparams.get("dim_feedforward", config.dim_feedforward)
            )
        else:
            self.norm_f = None

        rnn_layer = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[config.model_type]
        self.rnn = rnn_layer(
            input_size=self.hparams.get("d_model", config.d_model),
            hidden_size=self.hparams.get("dim_feedforward", config.dim_feedforward),
            num_layers=self.hparams.get("n_layers", config.n_layers),
            bidirectional=self.hparams.get("bidirectional", config.bidirectional),
            batch_first=True,
            dropout=self.hparams.get("rnn_dropout", config.rnn_dropout),
            bias=self.hparams.get("bias", config.bias),
            nonlinearity=(
                self.hparams.get("rnn_activation", config.rnn_activation)
                if config.model_type == "RNN"
                else None
            ),
        )

        self.embedding_layer = EmbeddingLayer(
            num_feature_info=num_feature_info,
            cat_feature_info=cat_feature_info,
            d_model=self.hparams.get("d_model", config.d_model),
            embedding_activation=self.hparams.get(
                "embedding_activation", config.embedding_activation
            ),
            layer_norm_after_embedding=self.hparams.get(
                "layer_norm_after_embedding", config.layer_norm_after_embedding
            ),
            use_cls=False,
            cls_position=-1,
            cat_encoding=self.hparams.get("cat_encoding", config.cat_encoding),
        )

        head_activation = self.hparams.get("head_activation", config.head_activation)

        self.tabular_head = MLP(
            self.hparams.get("dim_feedforward", config.dim_feedforward),
            hidden_units_list=self.hparams.get(
                "head_layer_sizes", config.head_layer_sizes
            ),
            dropout_rate=self.hparams.get("head_dropout", config.head_dropout),
            use_skip_layers=self.hparams.get(
                "head_skip_layers", config.head_skip_layers
            ),
            activation_fn=head_activation,
            use_batch_norm=self.hparams.get(
                "head_use_batch_norm", config.head_use_batch_norm
            ),
            n_output_units=num_classes,
        )

        self.linear = nn.Linear(config.d_model, config.dim_feedforward)

    def forward(self, num_features, cat_features):
        """
        Defines the forward pass of the model.

        Parameters
        ----------
        num_features : Tensor
            Tensor containing the numerical features.
        cat_features : Tensor
            Tensor containing the categorical features.

        Returns
        -------
        Tensor
            The output predictions of the model.
        """

        x = self.embedding_layer(num_features, cat_features)
        # RNN forward pass
        out, _ = self.rnn(x)
        z = self.linear(torch.mean(x, dim=1))

        if self.pooling_method == "avg":
            x = torch.mean(out, dim=1)
        elif self.pooling_method == "max":
            x, _ = torch.max(out, dim=1)
        elif self.pooling_method == "sum":
            x = torch.sum(out, dim=1)
        elif self.pooling_method == "last":
            x = x[:, -1, :]
        else:
            raise ValueError(f"Invalid pooling method: {self.pooling_method}")
        x = x + z
        if self.norm_f is not None:
            x = self.norm_f(x)
        preds = self.tabular_head(x)

        return preds
