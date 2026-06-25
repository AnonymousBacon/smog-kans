import torch
from kan import *

device = torch.device('cpu')
model = KAN(width = [2, 5, 1], grid = 5, k = 3, seed = 42, device = device)
f = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)
dataset = create_dataset(f, n_var=2, device=device)

print("train inputs shape:", dataset['train_input'].shape)
print("train labels shape:", dataset['train_label'].shape)

predictions = model(dataset['train_input'])
loss = torch.mean((predictions - dataset['train_label']) ** 2)

print('loss before training:',loss.item())
print(torch.var(dataset['train_label']).item())

model.fit(dataset, opt="LBFGS", steps=50)