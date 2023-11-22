"""
sharpness_aware_minimization.py

Author: Seongyoon Kim (seongyoonk25@gmail.com)
Date: 2023-05-16

This script provides a basic implementation of the Sharpness-Aware Minimization (SAM)
optimizer for efficiently improving generalization in deep learning models. It includes:

1. Definition of a sample model architecture.
2. Implementation of the SAM optimizer class.
3. Example usage of the SAM optimizer with a model and a dataset.

The implementation is based on the following papers:

- Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2020).
  Sharpness-aware minimization for efficiently improving generalization.
  arXiv preprint arXiv:2010.01412.
  (https://arxiv.org/abs/2010.01412)

- Kwon, J., Kim, J., Park, H., & Choi, I. K. (2021, July).
  Asam: Adaptive sharpness-aware minimization for scale-invariant learning of deep neural networks.
  In International Conference on Machine Learning (pp. 5905-5914). PMLR.
  (https://proceedings.mlr.press/v139/kwon21b.html)

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


class SAM(torch.optim.Optimizer):
    """
    Sharpness-aware minimization (SAM) implementation by https://github.com/davda54/sam,
    based on
    Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2020).
    Sharpness-aware minimization for efficiently improving generalization.
    arXiv preprint arXiv:2010.01412.
    (https://arxiv.org/abs/2010.01412)
    and
    Kwon, J., Kim, J., Park, H., & Choi, I. K. (2021, July). 
    Asam: Adaptive sharpness-aware minimization for scale-invariant learning of deep neural networks. 
    In International Conference on Machine Learning (pp. 5905-5914). PMLR.
    (https://proceedings.mlr.press/v139/kwon21b.html)
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


lr = ...
weight_decay = ...
model = Model(...)
base_optimizer = optim.SGD  # define an optimizer for the "sharpness-aware" update
optimizer = SAM(model.parameters(), base_optimizer, rho=0.05, lr=lr, momentum=0.9)  # rho=2 for ASAM
criterion = nn.MSELoss()
trainloader = ...
for X_batch, Y_batch in trainloader:

    out = model(X_batch)
    loss = criterion(out, Y_batch)
    model.zero_grad()
    loss.backward()
    optimizer.first_step(zero_grad=True)

    out = model(X_batch)
    criterion(out, Y_batch).backward()  # make sure to do a full forward pass
    optimizer.second_step(zero_grad=True)
