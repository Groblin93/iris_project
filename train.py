import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import pickle

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


def train_network(model,optimizer,criterion,X_train,y_train,num_epochs,train_losses):
    for epoch in range(num_epochs):
        #clear out the gradients from the last step loss.backward()
        optimizer.zero_grad()
        
        #forward feed
        output_train = model(X_train)

        #calculate the loss
        loss_train = criterion(output_train, y_train)
        
        #backward propagation: calculate gradients
        loss_train.backward()

        #update the weights
        optimizer.step()

        train_losses[epoch] = loss_train.item()


def read_data(data_name):
    df = pd.read_csv('data/'+data_name)
    df['Species'] = df['Species'].map({'Iris-setosa':0,
                                       'Iris-versicolor':1,
                                       'Iris-virginica':2})
    df.drop(['Id'],axis=1,inplace=True)
    X = df.drop(["Species"],axis=1).values
    y = df["Species"].values
    return X, y


def tts(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    np.savetxt("data/X_test.csv", X_test, delimiter=",")
    np.savetxt("data/y_test.csv", y_test, delimiter=",")
    return X_train, y_train

def main():
    data_name = 'Iris.csv'
    X, y = read_data(data_name)
    X_train, y_train = tts(X, y)
    
    input_dim  = 4 
    output_dim = 3
    model = NeuralNetworkClassificationModel(input_dim,output_dim)
    learning_rate = 0.01
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    num_epochs = 1000
    train_losses = np.zeros(num_epochs)
    
    train_network(model,optimizer,criterion,X_train,y_train,num_epochs,train_losses)
    pickle.dump(model, open('models/Iris_model.sav','wb'))


if __name__ == '__main__':
    main()