import lightning as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score , average_precision_score
import warnings
from ..base_models.lightning_wrapper import TaskModel
from ..data_utils.datamodule import MambularDataModule
from ..preprocessing import Preprocessor
import numpy as np
from lightning.pytorch.callbacks import ModelSummary
from sklearn.metrics import log_loss
from pytorch_lightning.loggers import TensorBoardLogger

class SklearnBaseClassifier(BaseEstimator):
    def __init__(self, model, config, **kwargs):
        preprocessor_arg_names = [
            "n_bins",
            "numerical_preprocessing",
            "use_decision_tree_bins",
            "binning_strategy",
            "task",
            "cat_cutoff",
            "treat_all_integers_as_numerical",
            "knots",
            "degree",
        ]

        self.config_kwargs = {
            k: v for k, v in kwargs.items() if k not in preprocessor_arg_names
        }
        self.config = config(**self.config_kwargs)

        preprocessor_kwargs = {
            k: v for k, v in kwargs.items() if k in preprocessor_arg_names
        }

        self.preprocessor = Preprocessor(**preprocessor_kwargs)
        self.task_model = None

        # Raise a warning if task is set to 'classification'
        if preprocessor_kwargs.get("task") == "regression":
            warnings.warn(
                "The task is set to 'regression'. The Classifier is designed for classification tasks.",
                UserWarning,
            )

        self.base_model = model
        self.built = False

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = {}
        params.update(self.config_kwargs)

        if deep:
            preprocessor_params = {
                "preprocessor__" + key: value
                for key, value in self.preprocessor.get_params().items()
            }
            params.update(preprocessor_params)

        return params

    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **parameters : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.
        """
        config_params = {
            k: v for k, v in parameters.items() if not k.startswith("preprocessor__")
        }
        preprocessor_params = {
            k.split("__")[1]: v
            for k, v in parameters.items()
            if k.startswith("preprocessor__")
        }

        if config_params:
            self.config_kwargs.update(config_params)
            if self.config is not None:
                for key, value in config_params.items():
                    setattr(self.config, key, value)
            else:
                self.config = self.config_class(**self.config_kwargs)

        if preprocessor_params:
            self.preprocessor.set_params(**preprocessor_params)

        return self

    def build_model(
        self,
        X,
        y,
        val_size: float = 0.2,
        X_val=None,
        y_val=None,
        random_state: int = 101,
        batch_size: int = 128,
        shuffle: bool = True,
        lr: float = 1e-4,
        lr_patience: int = 10,
        factor: float = 0.1,
        weight_decay: float = 1e-06,
        dataloader_kwargs={},
    ):
        """
        Builds the model using the provided training data.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            The target values (real numbers).
        val_size : float, default=0.2
            The proportion of the dataset to include in the validation split if `X_val` is None. Ignored if `X_val` is provided.
        X_val : DataFrame or array-like, shape (n_samples, n_features), optional
            The validation input samples. If provided, `X` and `y` are not split and this data is used for validation.
        y_val : array-like, shape (n_samples,) or (n_samples, n_targets), optional
            The validation target values. Required if `X_val` is provided.
        random_state : int, default=101
            Controls the shuffling applied to the data before applying the split.
        batch_size : int, default=64
            Number of samples per gradient update.
        shuffle : bool, default=True
            Whether to shuffle the training data before each epoch.
        lr : float, default=1e-3
            Learning rate for the optimizer.
        lr_patience : int, default=10
            Number of epochs with no improvement on the validation loss to wait before reducing the learning rate.
        factor : float, default=0.1
            Factor by which the learning rate will be reduced.
        weight_decay : float, default=0.025
            Weight decay (L2 penalty) coefficient.
        dataloader_kwargs: dict, default={}
            The kwargs for the pytorch dataloader class.



        Returns
        -------
        self : object
            The built classifier.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if isinstance(y, pd.Series):
            y = y.values
        if X_val:
            if not isinstance(X_val, pd.DataFrame):
                X_val = pd.DataFrame(X_val)
            if isinstance(y_val, pd.Series):
                y_val = y_val.values

        self.data_module = MambularDataModule(
            preprocessor=self.preprocessor,
            batch_size=batch_size,
            shuffle=shuffle,
            X_val=X_val,
            y_val=y_val,
            val_size=val_size,
            random_state=random_state,
            regression=False,
            **dataloader_kwargs
        )

        self.data_module.preprocess_data(
            X, y, X_val, y_val, val_size=val_size, random_state=random_state
        )

        # num_classes = len(np.unique(y))
        num_classes = 13

        self.task_model = TaskModel(
            model_class=self.base_model,
            num_classes=num_classes,
            config=self.config,
            cat_feature_info=self.data_module.cat_feature_info,
            num_feature_info=self.data_module.num_feature_info,
            lr=lr,
            lr_patience=lr_patience,
            lr_factor=factor,
            weight_decay=weight_decay,
        )

        self.built = True

        return self

    def get_number_of_params(self, requires_grad=True):
        """
        Calculate the number of parameters in the model.

        Parameters
        ----------
        requires_grad : bool, optional
            If True, only count the parameters that require gradients (trainable parameters).
            If False, count all parameters. Default is True.

        Returns
        -------
        int
            The total number of parameters in the model.

        Raises
        ------
        ValueError
            If the model has not been built prior to calling this method.
        """
        if not self.built:
            raise ValueError(
                "The model must be built before the number of parameters can be estimated"
            )
        else:
            if requires_grad:
                return sum(
                    p.numel() for p in self.task_model.parameters() if p.requires_grad
                )
            else:
                return sum(p.numel() for p in self.task_model.parameters())
    
    def load(
        self,
        X,
        y,
        val_size: float = 0.2,
        X_val=None,
        y_val=None,
        max_epochs: int = 100,
        random_state: int = 101,
        batch_size: int = 128,
        shuffle: bool = True,
        patience: int = 15,
        monitor: str = "val_loss",
        mode: str = "min",
        lr: float = 1e-4,
        lr_patience: int = 10,
        factor: float = 0.1,
        weight_decay: float = 1e-06,
        checkpoint_path="model_checkpoints",
        dataloader_kwargs={},
        rebuild=True,
        **trainer_kwargs
    ):
        if rebuild:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            if isinstance(y, pd.Series):
                y = y.values
            if X_val:
                if not isinstance(X_val, pd.DataFrame):
                    X_val = pd.DataFrame(X_val)
                if isinstance(y_val, pd.Series):
                    y_val = y_val.values

            self.data_module = MambularDataModule(
                preprocessor=self.preprocessor,
                batch_size=batch_size,
                shuffle=shuffle,
                X_val=X_val,
                y_val=y_val,
                val_size=val_size,
                random_state=random_state,
                regression=False,
                **dataloader_kwargs
            )

            self.data_module.preprocess_data(
                X, y, X_val, y_val, val_size=val_size, random_state=random_state
            )

            # num_classes = len(np.unique(y))
            num_classes = 13

            self.task_model = TaskModel(
                model_class=self.base_model,
                num_classes=num_classes,
                config=self.config,
                cat_feature_info=self.data_module.cat_feature_info,
                num_feature_info=self.data_module.num_feature_info,
                lr=lr,
                lr_patience=lr_patience,
                lr_factor=factor,
                weight_decay=weight_decay,
            )

        early_stop_callback = EarlyStopping(
            monitor=monitor, min_delta=0.00, patience=patience, verbose=False, mode=mode
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",  # Adjust according to your validation metric
            mode="min",
            save_top_k=1,
            dirpath=checkpoint_path,  # Specify the directory to save checkpoints
            filename="best_model",
        )

        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            checkpoint = torch.load(best_model_path)
            self.task_model.load_state_dict(checkpoint["state_dict"])

        return self
    
    def fit(
        self,
        X,
        y,
        val_size: float = 0.2,
        X_val=None,
        y_val=None,
        max_epochs: int = 100,
        random_state: int = 101,
        batch_size: int = 128,
        shuffle: bool = True,
        patience: int = 15,
        monitor: str = "val_loss",
        mode: str = "min",
        lr: float = 1e-4,
        lr_patience: int = 10,
        factor: float = 0.1,
        weight_decay: float = 1e-06,
        checkpoint_path="model_checkpoints",
        dataloader_kwargs={},
        rebuild=True,
        **trainer_kwargs
    ):
        """
        Trains the classification model using the provided training data. Optionally, a separate validation set can be used.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            The target values (real numbers).
        val_size : float, default=0.2
            The proportion of the dataset to include in the validation split if `X_val` is None. Ignored if `X_val` is provided.
        X_val : DataFrame or array-like, shape (n_samples, n_features), optional
            The validation input samples. If provided, `X` and `y` are not split and this data is used for validation.
        y_val : array-like, shape (n_samples,) or (n_samples, n_targets), optional
            The validation target values. Required if `X_val` is provided.
        max_epochs : int, default=100
            Maximum number of epochs for training.
        random_state : int, default=101
            Controls the shuffling applied to the data before applying the split.
        batch_size : int, default=64
            Number of samples per gradient update.
        shuffle : bool, default=True
            Whether to shuffle the training data before each epoch.
        patience : int, default=10
            Number of epochs with no improvement on the validation loss to wait before early stopping.
        monitor : str, default="val_loss"
            The metric to monitor for early stopping.
        mode : str, default="min"
            Whether the monitored metric should be minimized (`min`) or maximized (`max`).
        lr : float, default=1e-3
            Learning rate for the optimizer.
        lr_patience : int, default=10
            Number of epochs with no improvement on the validation loss to wait before reducing the learning rate.
        factor : float, default=0.1
            Factor by which the learning rate will be reduced.
        weight_decay : float, default=0.025
            Weight decay (L2 penalty) coefficient.
        checkpoint_path : str, default="model_checkpoints"
            Path where the checkpoints are being saved.
        dataloader_kwargs: dict, default={}
            The kwargs for the pytorch dataloader class.
        rebuild: bool, default=True
            Whether to rebuild the model when it already was built.
        **trainer_kwargs : Additional keyword arguments for PyTorch Lightning's Trainer class.


        Returns
        -------
        self : object
            The fitted classifier.
        """
        if rebuild:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            if isinstance(y, pd.Series):
                y = y.values
            if X_val:
                if not isinstance(X_val, pd.DataFrame):
                    X_val = pd.DataFrame(X_val)
                if isinstance(y_val, pd.Series):
                    y_val = y_val.values

            self.data_module = MambularDataModule(
                preprocessor=self.preprocessor,
                batch_size=batch_size,
                shuffle=shuffle,
                X_val=X_val,
                y_val=y_val,
                val_size=val_size,
                random_state=random_state,
                regression=False,
                **dataloader_kwargs
            )

            self.data_module.preprocess_data(
                X, y, X_val, y_val, val_size=val_size, random_state=random_state
            )

            # num_classes = len(np.unique(y))
            num_classes = 13

            self.task_model = TaskModel(
                model_class=self.base_model,
                num_classes=num_classes,
                config=self.config,
                cat_feature_info=self.data_module.cat_feature_info,
                num_feature_info=self.data_module.num_feature_info,
                lr=lr,
                lr_patience=lr_patience,
                lr_factor=factor,
                weight_decay=weight_decay,
            )

        early_stop_callback = EarlyStopping(
            monitor=monitor, min_delta=0.00, patience=patience, verbose=False, mode=mode
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",  # Adjust according to your validation metric
            mode="min",
            save_top_k=1,
            dirpath=checkpoint_path,  # Specify the directory to save checkpoints
            filename="best_model",
        )

        logger = TensorBoardLogger('tb_logs',name='pregnant')
        # Initialize the trainer and train the model
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[
                early_stop_callback,
                checkpoint_callback,
                ModelSummary(max_depth=2),
            ],
            logger=logger,
            **trainer_kwargs
        )
        self.trainer.fit(self.task_model, self.data_module)

        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            checkpoint = torch.load(best_model_path)
            self.task_model.load_state_dict(checkpoint["state_dict"])

        return self

    def predict(self, X):
        """
        Predicts target values for the given input samples.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            The input samples for which to predict target values.


        Returns
        -------
        predictions : ndarray, shape (n_samples,) or (n_samples, n_outputs)
            The predicted target values.
        """
        # Ensure model and data module are initialized
        if self.task_model is None or self.data_module is None:
            raise ValueError("The model or data module has not been fitted yet.")

        # Preprocess the data using the data module
        cat_tensors, num_tensors = self.data_module.preprocess_test_data(X)

        # Move tensors to appropriate device
        device = next(self.task_model.parameters()).device
        if isinstance(cat_tensors, list):
            cat_tensors = [tensor.to(device) for tensor in cat_tensors]
        else:
            cat_tensors = cat_tensors.to(device)

        if isinstance(num_tensors, list):
            num_tensors = [tensor.to(device) for tensor in num_tensors]
        else:
            num_tensors = num_tensors.to(device)

        # Set model to evaluation mode
        self.task_model.eval()

        # Perform inference
        with torch.no_grad():
            logits = self.task_model(num_features=num_tensors, cat_features=cat_tensors)

            # Check the shape of the logits to determine binary or multi-class classification
            if logits.shape[1] == 1:
                # Binary classification
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities > 0.5).long().squeeze()
            else:
                # Multi-class classification
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)

        # Convert predictions to NumPy array and return
        return predictions.cpu().numpy()

    def predict_proba(self, X):
        """
        Predict class probabilities for the given input samples.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            The input samples for which to predict class probabilities.


        Notes
        -----
        The method preprocesses the input data using the same preprocessor used during training,
        sets the model to evaluation mode, and then performs inference to predict the class probabilities.
        Softmax is applied to the logits to obtain probabilities, which are then converted from a PyTorch tensor
        to a NumPy array before being returned.


        Examples
        --------
        >>> from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score
        >>> # Define the metrics you want to evaluate
        >>> metrics = {
        ...     'Accuracy': (accuracy_score, False),
        ...     'Precision': (precision_score, False),
        ...     'F1 Score': (f1_score, False),
        ...     'AUC Score': (roc_auc_score, True)
        ... }
        >>> # Assuming 'X_test' and 'y_test' are your test dataset and labels
        >>> # Evaluate using the specified metrics
        >>> results = classifier.evaluate(X_test, y_test, metrics=metrics)


        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities for each input sample.

        """
        # Preprocess the data
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        device = next(self.task_model.parameters()).device
        cat_tensors, num_tensors = self.data_module.preprocess_test_data(X)
        if isinstance(cat_tensors, list):
            cat_tensors = [tensor.to(device) for tensor in cat_tensors]
        else:
            cat_tensors = cat_tensors.to(device)

        if isinstance(num_tensors, list):
            num_tensors = [tensor.to(device) for tensor in num_tensors]
        else:
            num_tensors = num_tensors.to(device)

        # Set the model to evaluation mode
        self.task_model.eval()

        # Perform inference
        with torch.no_grad():
            logits = self.task_model(num_features=num_tensors, cat_features=cat_tensors)
            logits_0 = logits[:,0]
            logits_1 = logits[:,1]
            logits_2 = logits[:,2]
            logits_3 = logits[:,3]
            logits_4 = logits[:,4]
            logits_5 = logits[:,5]
            logits_6 = logits[:,6]
            logits_7 = logits[:,7:10]
            logits_8 = logits[:,10:]
            probabilities_0 = torch.sigmoid(logits_0)
            probabilities_1 = torch.sigmoid(logits_1)
            probabilities_2 = torch.sigmoid(logits_2)
            probabilities_3 = torch.sigmoid(logits_3)
            probabilities_4 = torch.sigmoid(logits_4)
            probabilities_5 = torch.sigmoid(logits_5)
            probabilities_6 = torch.sigmoid(logits_6)
            probabilities_7 = torch.softmax(logits_7,dim=1)
            probabilities_8 = torch.softmax(logits_8,dim=1)
            probabilities = [probabilities_0,probabilities_1,probabilities_2,probabilities_3,probabilities_4,probabilities_5,probabilities_6,probabilities_7,probabilities_8]
            # if logits.shape[1] > 1:
            #     probabilities = torch.softmax(logits, dim=1)
            # else:
            #     probabilities = torch.sigmoid(logits)

        # Convert probabilities to NumPy array and return
        return [pro.cpu().numpy() for pro in probabilities]

    def evaluate(self, X, y_true, metrics=None):
        """
        Evaluate the model on the given data using specified metrics.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            The input samples to predict.
        y_true : array-like of shape (n_samples,)
            The true class labels against which to evaluate the predictions.
        metrics : dict
            A dictionary where keys are metric names and values are tuples containing the metric function
            and a boolean indicating whether the metric requires probability scores (True) or class labels (False).


        Returns
        -------
        scores : dict
            A dictionary with metric names as keys and their corresponding scores as values.


        Notes
        -----
        This method uses either the `predict` or `predict_proba` method depending on the metric requirements.
        """
        # Ensure input is in the correct format
        if metrics is None:
            metrics = {'AUC Score': (roc_auc_score, True)}

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Initialize dictionary to store results
        # scores = {}
        scores = []

        # Generate class probabilities if any metric requires them
        if any(use_proba for _, use_proba in metrics.values()):
            probabilities = self.predict_proba(X)

        # Generate class labels if any metric requires them
        if any(not use_proba for _, use_proba in metrics.values()):
            predictions = self.predict(X)

        #自己加的代码
        y_true = [y_true[:, i] for i in range(9)]
        
        # Compute each metric
        for metric_name, (metric_func, use_proba) in metrics.items():
            if use_proba:
                for i in range(7):
                    scores.append(metric_func(y_true[i], probabilities[i]))
                for i in range(7,9):
                    scores.append(metric_func(y_true[i], probabilities[i], multi_class='ovo'))
            else:
                scores[metric_name] = metric_func(y_true, predictions)

        return scores

    def score(self, X, y, metric=(log_loss, True)):
        """
        Calculate the score of the model using the specified metric.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            The input samples to predict.
        y : array-like of shape (n_samples,)
            The true class labels against which to evaluate the predictions.
        metric : tuple, default=(log_loss, True)
            A tuple containing the metric function and a boolean indicating whether the metric requires probability scores (True) or class labels (False).

        Returns
        -------
        score : float
            The score calculated using the specified metric.
        """
        metric_func, use_proba = metric

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if use_proba:
            probabilities = self.predict_proba(X)
            return metric_func(y, probabilities)
        else:
            predictions = self.predict(X)
            return metric_func(y, predictions)
