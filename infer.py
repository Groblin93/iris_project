import pickle

import hydra
import numpy as np
import torch


def write_pred(model, data):
    X_test = np.loadtxt(open(data.path + data.X_name, "rb"), delimiter=",")
    y_test = np.loadtxt(open(data.path + data.y_name, "rb"), delimiter=",")
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    y_pred = []
    final_pred = []
    with torch.no_grad():
        y_pred = model(X_test)
    for i in range(len(y_pred)):
        final_pred.append(np.argmax(y_pred[i]))
    final_pred = np.array(final_pred)
    np.savetxt(data.path + data.y_pred_name, final_pred, delimiter=",")


def get_accuracy_multiclass(pred_arr, original_arr):
    if len(pred_arr) != len(original_arr):
        return False
    count = 0
    for i in range(len(original_arr)):
        if pred_arr[i] == original_arr[i]:
            count += 1
    return count / len(pred_arr)


@hydra.main(config_path="configs", config_name="config", version_base="1.3.2")
def main(cfg):
    Iris_model = pickle.load(open(cfg.model.path + cfg.model.name, "rb"))
    write_pred(Iris_model, cfg.data)

    y_pred = np.loadtxt(open(cfg.data.path + cfg.data.y_pred_name, "rb"), delimiter=",")
    y_test = np.loadtxt(open(cfg.data.path + cfg.data.y_name, "rb"), delimiter=",")
    test_acc = get_accuracy_multiclass(y_pred, y_test)
    print(f"Test Accuracy: {round(test_acc*100,cfg.metric.round)}")


if __name__ == "__main__":
    main()
