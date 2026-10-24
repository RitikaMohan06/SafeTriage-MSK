"""
model.py
========
A pretrained DenseNet121 (ImageNet weights) fine-tuned for binary
classification, applied per-view then averaged to a study-level score.

Why DenseNet121: it's what the original MURA paper used, so it's a fair,
comparable baseline architecture — your novelty is in the calibration and
deferral layer on top, not in inventing a new CNN.

Why average views at the SCORE level (not average the raw images, and not
just pick one view): each view is evidence of the same underlying condition.
Averaging probabilities is simple, differentiable, and is what the original
MURA baseline does too, which makes your comparison to it fair.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class StudyLevelDenseNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = backbone.classifier.in_features
        backbone.classifier = nn.Linear(in_features, 1)  # single logit
        self.backbone = backbone

    def forward(self, x):
        """
        x: [batch, n_views, 3, H, W]
        Returns: study_logit [batch], per_view_logits [batch, n_views]
                 (per-view logits are kept for Grad-CAM / inspection)
        """
        b, v, c, h, w = x.shape
        x = x.view(b * v, c, h, w)
        logits = self.backbone(x).view(b, v)  # [batch, n_views]
        study_logit = logits.mean(dim=1)  # simple average aggregation
        return study_logit, logits
