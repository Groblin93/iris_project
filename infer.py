import pickle

import numpy as np
import torch


def write_pred(model):
    X_test = np.loadtxt(open("data/X_test.csv", "rb"), delimiter=",")
    y_test = np.loadtxt(open("data/y_test.csv", "rb"), delimiter=",")
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    y_pred = []
    final_pred = []
    with torch.no_grad():
        y_pred = model(X_test)
    for i in range(len(y_pred)):
        final_pred.append(np.argmax(y_pred[i]))
    final_pred = np.array(final_pred)
    np.savetxt("data/y_pred.csv", final_pred, delimiter=",")


def get_accuracy_multiclass(pred_arr, original_arr):
    if len(pred_arr) != len(original_arr):
        return False
    count = 0
    for i in range(len(original_arr)):
        if pred_arr[i] == original_arr[i]:
            count += 1
    return count / len(pred_arr)


def main():
    Iris_model = pickle.load(open("models/Iris_model.sav", "rb"))
    write_pred(Iris_model)

    y_pred = np.loadtxt(open("data/y_pred.csv", "rb"), delimiter=",")
    y_test = np.loadtxt(open("data/y_test.csv", "rb"), delimiter=",")
    test_acc = get_accuracy_multiclass(y_pred, y_test)
    print(f"Test Accuracy: {round(test_acc*100,3)}")


if __name__ == "__main__":
    main()
