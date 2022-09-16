import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_quantiles):
        super(Model, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_quantiles = num_quantiles

        self.out = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            ...,
            nn.Linear(self.hidden_size, self.output_size*self.num_quantiles)
        )

    def forward(self, x):

        batch_size = x.size(0)

        out = self.out(x).view(batch_size, self.output_size, self.num_quantiles)

        return out


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        losses = torch.cat(losses, dim=-1)
        loss = torch.mean(torch.sum(losses, dim=-1))
        return loss

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        if y_pred.ndim == 3:
            idx = self.quantiles.index(0.5)
            y_pred = y_pred[..., idx]
        return y_pred


model = Model(...)
quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
criterion = QuantileLoss(quantiles)
trainloader = ...
for X_batch, Y_batch in trainloader:

    out = model(X_batch)
    loss = criterion(out, Y_batch)

    out_numpy = criterion.to_prediction(out.detach().cpu()).numpy()
