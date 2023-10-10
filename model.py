import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np

# %%
### model 
class LinearRegressionModel(nn.Module):
    def __init__(self,input_size, hidden_size1, hidden_size2, output_size):
        super(LinearRegressionModel,self).__init__()

        self.hidden1 = nn.Linear(input_size, hidden_size1)
        self.hidden2 = nn.Linear(hidden_size1,hidden_size2)

        self.output = nn.Linear(hidden_size2, output_size)
                
    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x 

# %%
### def train model 

def train_model(model:nn, train_loader, test_loader, lr=1e-3, n_epoch=10,save=False, device= 'cpu'):
    optimr = Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    if device =='cuda':
        model.cuda()
        loss_fn.cuda()

    # training
    for e in range(n_epoch):
        print(f'---------第{e+1}輪訓練開始-----------')
        model.train()

        for data in train_loader:
            inputs, labels = data
            outputs = model(inputs)

            if device =='cuda':
                inputs.cuda()
                labels.cuda()
                outputs.cuda()

            loss = loss_fn(outputs, labels)

            # apply optim 
            optimr.zero_grad()
            loss.backward()
            optimr.step()

        print(f'round{e+1} train rmse = {loss.item()}')
        
        # testing 
        model.eval()
        total_test_loss = 0

        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                outputs = model(inputs)

                if device=='cuda':
                    inputs.cuda()
                    labels.cude()
                    outputs.cuda()

                loss = loss_fn(outputs, labels)
                total_test_loss += loss.item()
                
        print(f'round{e+1} test rmse ={total_test_loss}\n')

        if save :
            torch.save(model, f'model_{e+1}_1007.pth')
        else:
            pass

        
