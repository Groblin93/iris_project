import dvc.api
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import argmax
from torch.nn import CrossEntropyLoss, Linear, ReLU
from torch.optim import AdamW


class lightning_nn_classification_model(pl.LightningModule):
    def __init__(self, cfg):
        super(lightning_nn_classification_model, self).__init__()
        self.cfg = cfg
        self.input_layer = Linear(cfg.model.input_dim, cfg.model.lays[0])
        self.hidden_layer1 = Linear(cfg.model.lays[0], cfg.model.lays[1])
        self.output_layer = Linear(cfg.model.lays[1], cfg.model.output_dim)
        self.relu = ReLU()

    def forward(self, X):
        out = self.relu(self.input_layer(X))
        out = self.relu(self.hidden_layer1(out))
        out = self.output_layer(out)
        return out

    def criterion(self, y_pred, y):
        loss = CrossEntropyLoss()
        return loss(y_pred, y)

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.cfg.training.rate)
        return opt

    def training_step(self, batch):
        X, y = batch
        X = X.to(torch.float32)
        y = y.long()
        y_pred = self.forward(X)
        loss = self.criterion(y_pred, y)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        X, y = batch
        X = X.to(torch.float32)
        y = y.long()
        y_pred = self.forward(X)
        loss = self.criterion(y_pred, y)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        X, y = batch
        X = X.to(torch.float32)
        y = y.long()
        y_pred = self.forward(X)
        test_loss = self.criterion(y_pred, y)
        labels_hat = argmax(y_pred, dim=1)
        test_acc = sum(y == labels_hat).item() / (len(y) * 1.0)
        self.log_dict({"test_loss": test_loss, "test_acc": test_acc})


class data_set:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        X = self.X[index]
        # y = self.y.iloc[index]
        y = self.y[index]
        return (X, y)

    def __len__(self):
        count = self.X.shape[0]
        return count


class data_module(pl.LightningDataModule):
    def __init__(
        self,
        path,
        name,
        X_name,
        y_name,
        repo,
        col_names,
        nums,
        test_size,
        val_size,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.path = path
        self.name = (name,)
        self.X_name = (X_name,)
        self.y_name = (y_name,)
        self.repo = (repo,)
        self.col_names = (col_names,)
        self.nums = nums
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        with dvc.api.open(self.path + self.name[0], repo=self.repo[0]) as f:
            df = pd.read_csv(f, names=self.col_names[0])
            df["Species"] = df["Species"].map(self.nums)
            scaler = StandardScaler()
            X = scaler.fit_transform(df.drop(["Species"], axis=1))
            y = df["Species"].values
            train_X, test_X, train_y, test_y = train_test_split(
                X, y, test_size=self.test_size
            )
            train_X, val_X, train_y, val_y = train_test_split(
                train_X, train_y, test_size=self.val_size
            )
            np.savetxt(self.path + self.X_name[0], test_X, delimiter=",")
            np.savetxt(self.path + self.y_name[0], test_y, delimiter=",")

            self.train_dataset = data_set(train_X, train_y)
            self.val_dataset = data_set(val_X, val_y)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
