"""
gradcam.py
==========
Grad-CAM highlights which pixels of an X-ray drove the model's prediction.
We use it specifically to compare DEFERRED vs. CONFIDENT cases: the
hypothesis worth showing in your report is that deferred cases often have
diffuse/ambiguous attention maps (no clear focal abnormality), which is
a nice qualitative sanity check that the uncertainty model "knows what
it doesn't know" for a sensible reason, not an arbitrary one.
"""
import torch
import torch.nn.functional as F
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, single_view_image_tensor):
        """
        single_view_image_tensor: [1, 3, H, W] — one view, batch size 1.
        Returns a [H, W] numpy heatmap normalized to [0, 1].
        """
        self.model.eval()
        logit = self.model.backbone(single_view_image_tensor)
        self.model.zero_grad()
        logit.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # global avg pool of grads
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=single_view_image_tensor.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def get_target_layer(model):
    """DenseNet121's last conv block — standard Grad-CAM target for this arch."""
    return model.backbone.features.norm5
