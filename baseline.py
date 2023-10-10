# %%
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
from torch.nn.functional import normalize

import model

# device 
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# dtype
dtype = torch.float32

# %% 
### find if nan

PATH = 'C:/Users/Wilson/Desktop/t-brain_dataset'
df = pd.read_csv(f'{PATH}/training_data.csv')
num_rows, num_columns = df.shape

numeric_df = df.select_dtypes(include='number')
numeric_df.drop(['單價'],axis=1,inplace=True)

features = numeric_df.values
labels = df['單價'].values

# %%
### preprocess 
features_tersor = torch.tensor(features, dtype=dtype)
normalized_features = normalize(features_tersor, dim=0)

labels_tensor = torch.tensor(labels, dtype=torch.float32)
# %% 
### split data
train_ratio = 0.8
test_ratio = 0.2
total_size = len(df)
batch_size = 64
shuffle = True

train_size = int(train_ratio * total_size)
test_size = total_size - train_size

# random_split 
train_dataset, test_dataset = random_split(
    TensorDataset(normalized_features, labels_tensor),
    [train_size, test_size]
)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
### model = LRM
input_size = features_tersor.shape[1]
hidden_size1 = 25
hidden_size2 = 25
output_size = 1

lms_model = model.LinearRegressionModel(
        input_size=input_size,
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2,
        output_size=output_size
)

# %%
### train and test 
lr = 1e-2
epoch = 10

model.train_model(
    model=lms_model,
    train_loader=train_loader,
    test_loader=test_loader,
    lr=lr,
    n_epoch=epoch,
    save=False,
    device=device
)
# %%
