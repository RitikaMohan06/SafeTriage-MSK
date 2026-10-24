"""Model loading + prediction + Grad-CAM overlay. torch is imported lazily so the
dashboard still works when no trained weights are present."""
import base64, io, os
import numpy as np
from PIL import Image
from src.serving import Calibration, decide

WEIGHTS = os.environ.get("WEIGHTS", "outputs/best_model.pt")
CALIB = os.environ.get("CALIB", "outputs/calibration.json")


class Engine:
    def __init__(self):
        self.ready, self.error = False, None
        try:
            import torch
            from src.model import StudyLevelDenseNet
            from src.dataset import get_transforms
            from src.gradcam import GradCAM, get_target_layer
            self.torch = torch
            self.cal = Calibration(CALIB)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = StudyLevelDenseNet(pretrained=False)
            self.model.load_state_dict(torch.load(WEIGHTS, map_location=self.device))
            self.model.to(self.device).eval()
            self.tf = get_transforms(False)
            self.cam = GradCAM(self.model, get_target_layer(self.model))
            self.ready = True
        except Exception as e:  # missing torch / weights / calibration
            self.error = f"{type(e).__name__}: {e}"

    def _overlay(self, img, tensor):
        import matplotlib; matplotlib.use("Agg")
        from matplotlib import cm
        # Grad-CAM requires autograd, but the model is already in eval mode and
        # the view tensor only needs a lightweight temporary gradient pass.
        x = tensor.unsqueeze(0).clone().detach().requires_grad_(True)
        heat = self.cam.generate(x)
        base = np.asarray(img.resize((heat.shape[1], heat.shape[0])).convert("RGB"), dtype=np.float32) / 255
        col = cm.jet(heat)[..., :3]
        out = (0.55 * base + 0.45 * col).clip(0, 1)
        buf = io.BytesIO(); Image.fromarray((out * 255).astype(np.uint8)).save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def predict(self, images, body_part):
        images = images[:3]
        tensors = [self.tf(i.convert("RGB")) for i in images]
        x = self.torch.stack(tensors + [tensors[-1]] * (3 - len(tensors))).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            logit, per_view = self.model(x)
        p = self.cal.probability(logit.item())
        lo, hi = self.cal.thresholds(body_part)
        overlays = [self._overlay(im, t) for im, t in zip(images, tensors)]
        return dict(probability=p, decision=decide(p, lo, hi), tau_lo=lo, tau_hi=hi,
                    body_part=body_part, n_views=len(images), gradcam=overlays,
                    per_view_probability=[self.cal.probability(v) for v in per_view[0][:len(images)].tolist()])
