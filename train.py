import pickle

import dvc.api
import hydra
import numpy as np
import pandas as pd
import torch
from iris_project.my_types import nn_classification_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_network(
    model, optimizer, criterion, X_train, y_train, num_epochs, train_losses
):
    for epoch in range(num_epochs):
        # clear out the gradients from the last step loss.backward()
        optimizer.zero_grad()

        # forward feed
        output_train = model(X_train)

        # calculate the loss
        loss_train = criterion(output_train, y_train)

        # backward propagation: calculate gradients
        loss_train.backward()

        # update the weights
        optimizer.step()

        train_losses[epoch] = loss_train.item()


def read_data(data):
    with dvc.api.open(data.path + data.name, repo=data.repo) as f:
        df = pd.read_csv(f, names=data.col_names)
        df["Species"] = df["Species"].map(data.nums)
        X = df.drop(["Species"], axis=1).values
        y = df["Species"].values

    return X, y


def tts(X, y, data):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=data.test_size, random_state=data.seed
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    np.savetxt(data.path + data.X_name, X_test, delimiter=",")
    np.savetxt(data.path + data.y_name, y_test, delimiter=",")
    return X_train, y_train


@hydra.main(config_path="configs", config_name="config", version_base="1.3.2")
def main(cfg):
    X, y = read_data(cfg.data)
    X_train, y_train = tts(X, y, cfg.data)

    model = nn_classification_model(cfg.model)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.rate)
    train_losses = np.zeros(cfg.training.num_epochs)

    train_network(
        model,
        optimizer,
        criterion,
        X_train,
        y_train,
        cfg.training.num_epochs,
        train_losses,
    )
    pickle.dump(model, open(cfg.model.path + cfg.model.name, "wb"))


if __name__ == "__main__":
    main()
