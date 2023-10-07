import pickle
import numpy as np
import torch

class NeuralNetworkClassificationModel(torch.nn.Module):
    def __init__(self,input_dim,output_dim):
        super(NeuralNetworkClassificationModel,self).__init__()
        self.input_layer    = torch.nn.Linear(input_dim,128)
        self.hidden_layer1  = torch.nn.Linear(128,64)
        self.output_layer   = torch.nn.Linear(64,output_dim)
        self.relu = torch.nn.ReLU()
    
    
    def forward(self,x):
        out =  self.relu(self.input_layer(x))
        out =  self.relu(self.hidden_layer1(out))
        out =  self.output_layer(out)
        return out


def write_pred(model):
    X_test = np.loadtxt(open("data/X_test.csv", "rb"), delimiter=",", skiprows=0)
    y_test = np.loadtxt(open("data/y_test.csv", "rb"), delimiter=",", skiprows=0)
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

def get_accuracy_multiclass(pred_arr,original_arr):
    if len(pred_arr)!=len(original_arr):
        return False    
    count = 0
    for i in range(len(original_arr)):
        if pred_arr[i] == original_arr[i]:
            count+=1
    return count/len(pred_arr)


def main():
    Iris_model = pickle.load(open('models/Iris_model.sav', 'rb'))
    write_pred(Iris_model)

    y_pred = np.loadtxt(open("data/y_pred.csv", "rb"), delimiter=",", skiprows=0)
    y_test = np.loadtxt(open("data/y_test.csv", "rb"), delimiter=",", skiprows=0)
    test_acc  = get_accuracy_multiclass(y_pred,y_test)
    print(f"Test Accuracy: {round(test_acc*100,3)}") 

if __name__ == '__main__':
    main()