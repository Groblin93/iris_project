import pickle

import dvc.api
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


def read_data():
    with dvc.api.open(
        "data/Iris.csv", repo="https://github.com/Groblin93/iris_project"
    ) as f:
        col_names = [
            "Sepal_Length",
            "Sepal_Width",
            "Petal_Length",
            "Petal_Width",
            "Species",
        ]
        df = pd.read_csv(f, names=col_names)
        df["Species"] = df["Species"].map(
            {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
        )
        X = df.drop(["Species"], axis=1).values
        y = df["Species"].values

    return X, y


def tts(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    np.savetxt("data/X_test.csv", X_test, delimiter=",")
    np.savetxt("data/y_test.csv", y_test, delimiter=",")
    return X_train, y_train


def main():
    X, y = read_data()
    X_train, y_train = tts(X, y)

    input_dim = 4
    output_dim = 3
    learning_rate = 0.1
    num_epochs = 1000

    model = nn_classification_model(input_dim, output_dim)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = np.zeros(num_epochs)

    train_network(model, optimizer, criterion, X_train, y_train, num_epochs, train_losses)
    pickle.dump(model, open("models/Iris_model.sav", "wb"))


if __name__ == "__main__":
    main()
