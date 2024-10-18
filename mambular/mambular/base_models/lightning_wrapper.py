import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from typing import Type


class TaskModel(pl.LightningModule):
    """
    PyTorch Lightning Module for training and evaluating a model.

    Parameters
    ----------
    model_class : Type[nn.Module]
        The model class to be instantiated and trained.
    config : dataclass
        Configuration dataclass containing model hyperparameters.
    loss_fn : callable
        Loss function to be used during training and evaluation.
    lr : float, optional
        Learning rate for the optimizer (default is 1e-3).
    num_classes : int, optional
        Number of classes for classification tasks (default is 1).
    lss : bool, optional
        Custom flag for additional loss configuration (default is False).
    **kwargs : dict
        Additional keyword arguments.
    """

    def __init__(
        self,
        model_class: Type[nn.Module],
        config,
        cat_feature_info,
        num_feature_info,
        num_classes=1,
        lss=False,
        family=None,
        loss_fct: callable = None,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lss = lss
        self.family = family
        self.loss_fct = loss_fct

        self.weight_0 = torch.tensor([11.0076])
        self.weight_1 = torch.tensor([14.1575])
        self.weight_2 = torch.tensor([26.8558])
        self.weight_3 = torch.tensor([279.5243])
        self.weight_4 = torch.tensor([78.1981])
        self.weight_5 = torch.tensor([1.1464])
        self.weight_6 = torch.tensor([127.4524])
        self.weight_7 = torch.tensor([0.3291, 0.3340, 0.3368])
        self.weight_8 = torch.tensor([0.3299, 0.3338, 0.3363])

        if lss:
            pass
        else:
            # if num_classes == 2:
            #     if not self.loss_fct:
            #         self.loss_fct = nn.BCEWithLogitsLoss()
            #     self.acc = torchmetrics.Accuracy(task="binary")
            #     self.auroc = torchmetrics.AUROC(task="binary")
            #     self.precision = torchmetrics.Precision(task="binary")
            #     self.num_classes = 1
            # elif num_classes > 2:
            #     if not self.loss_fct:
            #         self.loss_fct = nn.CrossEntropyLoss()
            #     self.acc = torchmetrics.Accuracy(
            #         # task="multiclass", num_classes=num_classes
            #         task="multiclass", num_classes=3
            #     )
            #     self.auroc = torchmetrics.AUROC(
            #         task="multiclass", num_classes=num_classes
            #     )
            #     self.precision = torchmetrics.Precision(
            #         task="multiclass", num_classes=num_classes
            #     )
            # else:
            #     self.loss_fct = nn.MSELoss()

            # self.bi_loss_fct = nn.BCEWithLogitsLoss()
            self.bi_acc = torchmetrics.Accuracy(task="binary")
            self.bi_auroc = torchmetrics.AUROC(task="binary")
            self.bi_precision = torchmetrics.Precision(task="binary")

            # self.multi_loss_fct = nn.CrossEntropyLoss()
            self.multi_acc = torchmetrics.Accuracy(task="multiclass", num_classes=3)
            self.multi_auroc = torchmetrics.AUROC(task="multiclass", num_classes=3)
            self.multi_precision = torchmetrics.Precision(task="multiclass", num_classes=3)

            self.reg_loss_fct = nn.MSELoss()

        self.save_hyperparameters(ignore=["model_class", "loss_fn"])

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)

        if family is None and num_classes == 2:
            output_dim = 1
        else:
            output_dim = num_classes

        self.base_model = model_class(
            config=config,
            num_feature_info=num_feature_info,
            cat_feature_info=cat_feature_info,
            num_classes=output_dim,
            **kwargs,
        )

    def forward(self, num_features, cat_features):
        """
        Forward pass through the model.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the model's forward method.
        **kwargs : dict
            Keyword arguments passed to the model's forward method.

        Returns
        -------
        Tensor
            Model output.
        """

        return self.base_model.forward(num_features, cat_features)

    def compute_loss(self, predictions, y_true, weight):
        """
        Compute the loss for the given predictions and true labels.

        Parameters
        ----------
        predictions : Tensor
            Model predictions.
        y_true : Tensor
            True labels.

        Returns
        -------
        Tensor
            Computed loss.
        """
        if self.lss:
            return self.family.compute_loss(predictions, y_true.squeeze(-1))
        else:
            if predictions.dim() == 1:
                y_true = y_true.float()
                weight = weight.to(y_true.device)
                loss_fct = nn.BCEWithLogitsLoss(pos_weight=weight)
                loss = loss_fct(predictions, y_true)
            else:
                y_true = y_true.long()
                weight = weight.to(y_true.device)
                loss_fct = nn.CrossEntropyLoss(weight=weight)
                loss = loss_fct(predictions,y_true)
            return loss

    def training_step(self, batch, batch_idx):
        """
        Training step for a single batch.

        Parameters
        ----------
        batch : tuple
            Batch of data containing numerical features, categorical features, and labels.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        Tensor
            Training loss.
        """

        cat_features, num_features, labels = batch
        preds = self(num_features=num_features, cat_features=cat_features)

        preds_0 = preds[:,0]
        preds_1 = preds[:,1]
        preds_2 = preds[:,2]
        preds_3 = preds[:,3]
        preds_4 = preds[:,4]
        preds_5 = preds[:,5]
        preds_6 = preds[:,6]
        preds_7 = preds[:,7:10]
        preds_8 = preds[:,10:]
        labels = torch.unbind(labels,dim=1)
        
        loss = self.compute_loss(preds_0, labels[0], weight=self.weight_0) + \
               self.compute_loss(preds_1, labels[1], weight=self.weight_1) + \
               self.compute_loss(preds_2, labels[2], weight=self.weight_2) + \
               self.compute_loss(preds_3, labels[3], weight=self.weight_3) + \
               self.compute_loss(preds_4, labels[4], weight=self.weight_4) + \
               self.compute_loss(preds_5, labels[5], weight=self.weight_5) + \
               self.compute_loss(preds_6, labels[6], weight=self.weight_6) + \
               self.compute_loss(preds_7, labels[7], weight=self.weight_7) + \
               self.compute_loss(preds_8, labels[8], weight=self.weight_8)

        self.log(
            "train_loss", 
            loss, 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True, 
            logger=True,
            sync_dist=True
        )

        # Log additional metrics
        if not self.lss:
            if self.num_classes > 1:
                # acc = self.acc(preds, labels)
                acc = (self.bi_auroc(preds_0,labels[0]) + 
                       self.bi_auroc(preds_1,labels[1]) + 
                       self.bi_auroc(preds_2,labels[2]) + 
                       self.bi_auroc(preds_3,labels[3]) + 
                       self.bi_auroc(preds_4,labels[4]) + 
                       self.bi_auroc(preds_5,labels[5]) + 
                       self.bi_auroc(preds_6,labels[6]) + 
                       self.multi_auroc(preds_7,labels[7]) + 
                       self.multi_auroc(preds_8,labels[8])
                       ) / 9
                self.log(
                    "train_auc",
                    acc,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True
                )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for a single batch.

        Parameters
        ----------
        batch : tuple
            Batch of data containing numerical features, categorical features, and labels.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        Tensor
            Validation loss.
        """

        cat_features, num_features, labels = batch
        preds = self(num_features=num_features, cat_features=cat_features)
        
        labels = labels.int()
        preds_0 = preds[:,0]
        preds_1 = preds[:,1]
        preds_2 = preds[:,2]
        preds_3 = preds[:,3]
        preds_4 = preds[:,4]
        preds_5 = preds[:,5]
        preds_6 = preds[:,6]
        preds_7 = preds[:,7:10]
        preds_8 = preds[:,10:]
        labels = torch.unbind(labels,dim=1)

        val_loss = self.compute_loss(preds_0, labels[0], weight=self.weight_0) + \
                   self.compute_loss(preds_1, labels[1], weight=self.weight_1) + \
                   self.compute_loss(preds_2, labels[2], weight=self.weight_2) + \
                   self.compute_loss(preds_3, labels[3], weight=self.weight_3) + \
                   self.compute_loss(preds_4, labels[4], weight=self.weight_4) + \
                   self.compute_loss(preds_5, labels[5], weight=self.weight_5) + \
                   self.compute_loss(preds_6, labels[6], weight=self.weight_6) + \
                   self.compute_loss(preds_7, labels[7], weight=self.weight_7) + \
                   self.compute_loss(preds_8, labels[8], weight=self.weight_8)

        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )

        # Log additional metrics
        if not self.lss:
            if self.num_classes > 1:
                # acc = self.acc(preds, labels)
                acc = (self.bi_auroc(preds_0,labels[0]) + 
                       self.bi_auroc(preds_1,labels[1]) + 
                       self.bi_auroc(preds_2,labels[2]) + 
                       self.bi_auroc(preds_3,labels[3]) + 
                       self.bi_auroc(preds_4,labels[4]) + 
                       self.bi_auroc(preds_5,labels[5]) + 
                       self.bi_auroc(preds_6,labels[6]) + 
                       self.multi_auroc(preds_7,labels[7]) +
                       self.multi_auroc(preds_8,labels[8])
                       ) / 9
                self.log(
                    "val_auc",
                    acc,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True
                )

        return val_loss

    def test_step(self, batch, batch_idx):
        """
        Test step for a single batch.

        Parameters
        ----------
        batch : tuple
            Batch of data containing numerical features, categorical features, and labels.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        Tensor
            Test loss.
        """
        cat_features, num_features, labels = batch
        preds = self(num_features=num_features, cat_features=cat_features)
        test_loss = self.compute_loss(preds, labels)

        self.log(
            "test_loss",
            test_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Log additional metrics
        if not self.lss:
            if self.num_classes > 1:
                acc = self.acc(preds, labels)
                self.log(
                    "test_acc",
                    acc,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

        return test_loss

    def configure_optimizers(self):
        """
        Sets up the model's optimizer and learning rate scheduler based on the configurations provided.

        Returns
        -------
        dict
            A dictionary containing the optimizer and lr_scheduler configurations.
        """
        optimizer = torch.optim.Adam(
            self.base_model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.lr_factor,
                patience=self.lr_patience,
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
