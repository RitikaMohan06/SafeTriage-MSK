"""
calibration.py
===============
Temperature scaling: learns a single scalar T that rescales logits
(logit / T) so that softmax probabilities better match true frequencies.

Why we need this BEFORE conformal prediction: conformal prediction gives
correct coverage regardless of how well-calibrated the underlying scores
are, BUT the resulting prediction sets are smaller (more useful / less
"everything is uncertain") when the scores are well-calibrated to begin
with. Temperature scaling is the cheapest way to improve that efficiency —
it adds one parameter and cannot change the model's ranking of cases
(so accuracy is unaffected), only the confidence values.
"""
import torch
import torch.nn as nn
import torch.optim as optim


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits, labels, lr=0.01, max_iter=50):
        """
        logits, labels: 1D tensors from the TUNE split (not train, not
        calibration, not test — a fourth, independent slice of data would
        be ideal, but for a course project reusing the tune split for this
        single scalar is an accepted simplification; state this in your
        report).
        """
        logits = logits.detach()
        labels = labels.float().detach()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits).squeeze(), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self.temperature.item()
