"""
torch_basic_supervised.py

Author: Seongyoon Kim (seongyoonk25@gmail.com)
Date: 2023-05-16

This script provides a basic framework for supervised deep learning using PyTorch.
It includes the following steps:

1. Import required libraries.
2. Read input data from CSV files.
3. Calculate mean and standard deviation for input and output data.
4. Define the model architecture.
5. Allocate CUDA if available.
6. Configure hyperparameters and set up data loaders.
7. Create and train the model.
8. Plot and evaluate the training and testing results.

Please make sure to install the required libraries and prepare the input CSV files
before running this script.
"""


# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
# import torch_optimizer as optim  # for extended optimizers

# Read data
data_input_train = pd.read_csv('./training_input.csv', header=None).values
data_input_test = pd.read_csv('./test_input.csv', header=None).values
data_output_train = pd.read_csv('./training_input.csv', header=None).values
data_output_test = pd.read_csv('./test_input.csv', header=None).values

# Get mean and std
input_mean = torch.FloatTensor(data_input_train.mean(0))
input_std = torch.FloatTensor(data_input_train.std(0))
output_mean = torch.FloatTensor(data_output_train.mean(0))
output_std = torch.FloatTensor(data_output_train.std(0))


# Construct model
class Model(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Any network
        self.embed = nn.Linear(self.input_size, self.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size,
                                                   nhead=4,
                                                   dim_feedforward=self.hidden_size,
                                                   activation='gelu',
                                                   batch_first=True)
        self.model = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.out = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, x):

        batch_size = x.size(0)

        out = self.embed(x).view(batch_size, 1, self.hidden_size)
        out = self.model(out).view(batch_size, self.hidden_size)
        out = self.out(out).view(batch_size, self.output_size)

        return out


# Allocate CUDA
cuda = torch.cuda.is_available()
# cuda = False
if cuda:
    device = 'cuda:2'
else:
    device = 'cpu'

print(cuda, device)

# Configure
num_epoch = int(1e6)
lr = 1e-2
weight_decay = 1e-6
batch_size = 32
criterion = nn.MSELoss()
hidden_size = 128

# Get dataloader
X_train = torch.FloatTensor(data_input_train)
y_train = torch.FloatTensor(data_output_train)
train_data = []
for i in range(len(X_train)):
    train_data.append([X_train[i], y_train[i]])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

X_test = torch.FloatTensor(data_input_test)
y_test = torch.FloatTensor(data_output_test)
test_data = []
for i in range(len(X_test)):
    test_data.append([X_test[i], y_test[i]])
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Make model
model = Model(data_input_train.shape[1], hidden_size, data_output_train.shape[1]).to(device)

# Set optimizer
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.99)

# Set model save path
path = f'./models/test_path_{hidden_size}hidden'

# Train!
loss_array_train = []
loss_array_test = []
patience = 0
patience_thresh = 1000
min_loss = np.inf
time_init = time.time()
for e in range(num_epoch):

    model.train()  # training mode

    loss_array_tmp = []

    for X_batch, Y_batch in train_loader:

        out = model(((X_batch - input_mean) / input_std).to(device))
        loss = criterion(out, ((Y_batch - output_mean) / output_std).to(device))
        loss_array_tmp.append(loss.item())
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)

        optimizer.step()

    loss_array_train.append(np.mean(loss_array_tmp))

    model.eval()  # evaluation mode

    loss_array_tmp = []

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:

            out = model(((X_batch - input_mean) / input_std).to(device))
            loss = criterion(out, ((Y_batch - output_mean) / output_std).to(device))
            loss_array_tmp.append(loss.item())

        loss_array_test.append(np.mean(loss_array_tmp))

    if e % 100 == 0:
        print('Epoch: {}, Train loss: {:.4e}, Test  loss: {:4e}, Total time:{:.1f}s'.format(e, loss_array_train[-1], loss_array_test[-1], time.time() - time_init))

    # update the minimum loss
    if min_loss > loss_array_train[-1]:
        patience = 0
        min_loss = loss_array_train[-1]
        torch.save(model.state_dict(), path)
    else:
        patience += 1

    # early stop when patience become larger than patience_thresh
    if patience > patience_thresh:
        break

plt.plot(loss_array_train, label='train loss')
plt.plot(loss_array_test, label='test loss')
plt.axvline(x=patience_thresh, linestyle='--', color='grey')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.yscale('log')
plt.legend()
plt.show()

# Evaluate!
model.load_state_dict(torch.load(path, map_location=device))
model.eval()

with torch.no_grad():
    out_array = []
    true_array = []
    for X_batch, Y_batch in train_loader:
        out = model(((X_batch - input_mean) / input_std).to(device))
        out_array.append((out.detach().cpu() * output_std + output_mean).numpy())
        true_array.append(Y_batch.cpu().numpy())

out_array = np.concatenate(out_array)
true_array = np.concatenate(true_array)

plt.figure(figsize=(6, 5))
plt.plot([np.min([true_array, out_array]), np.max([true_array, out_array])], 
         [np.min([true_array, out_array]), np.max([true_array, out_array])], 'k--')
for i in range(data_output_train.shape[1]):
    plt.plot(true_array[:, i], out_array[:, i], '.')
plt.xlabel('Observed', fontsize=24)
plt.ylabel('Predicted', fontsize=24)
plt.title('Train', fontsize=28)
plt.tight_layout()
plt.show()

with torch.no_grad():
    out_array = []
    for X_batch in test_loader:
        out = model(((X_batch - input_mean) / input_std).to(device))
        out_array.append((out.detach().cpu() * output_std + output_mean).numpy())

out_array = np.concatenate(out_array)
