"""
anti-pgd.py

Author: Seongyoon Kim (seongyoonk25@gmail.com)
Date: 2023-05-16

This script provides a basic implementation of the Anti-correlated Perturbed Gradient
Descent (Anti-PGD) method for improving generalization in deep learning models. It includes:

1. Definition of a sample model architecture.
2. Implementation of the AntiPGD class for applying anti-PGD updates to model parameters.
3. Example usage of the AntiPGD with a model, an optimizer, and a dataset.

Please make sure to install the required libraries and prepare the input dataset
before running this script.
"""


import torch
import torch.nn as nn
import torch.optim as optim


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()

        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # any hidden size
        self.output_size = output_size  # target size

        self.out = nn.Sequential(  # any model
            nn.Linear(self.input_size, self.hidden_size),
            ...,
            nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, x):
        out = self.out(x)
        return out


class AntiPGD:
    """
    Anticorrelated perturbed gradient descent (Anti-PGD) implementation,
    based on
    Orvieto, A., Kersting, H., Proske, F., Bach, F., & Lucchi, A. (2022).
    Anticorrelated noise injection for improved generalization.
    arXiv preprint arXiv:2202.02831.
    (https://arxiv.org/abs/2202.02831)
    """
    def __init__(self, model):
        super(AntiPGD, self).__init__()

        self.model = model

        self.previous_perturbation = []
        for p in self.model.parameters():
            if p.requires_grad:
                self.previous_perturbation.append(torch.randn_like(p))

    def step(self, lr, sigma=.5):
        self.current_perturbation = []
        i = 0
        for p in model.parameters():
            if p.requires_grad:
                self.current_perturbation.append(torch.randn_like(p))
                p.grad += sigma*lr*(self.current_perturbation[i] -
                                    self.previous_perturbation[i])
                i += 1

        self.previous_perturbation = self.current_perturbation


lr = ...
weight_decay = ...
model = Model(...)
optimizer = optim.SGD(model.parameters(),
                      lr=lr,
                      weight_decay=weight_decay,
                      momentum=0.99)
criterion = nn.MSELoss()
trainloader = ...
antipgd = AntiPGD(model)
for X_batch, Y_batch in trainloader:

    out = model(X_batch)
    loss = criterion(out, Y_batch)
    model.zero_grad()
    loss.backward()

    antipgd.step(lr)
    optimizer.step()
