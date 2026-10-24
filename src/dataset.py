"""
dataset.py
==========
Loads MURA, and — this is the important part — splits it BY PATIENT and
groups images BY STUDY, because:

  - MURA studies can have 1-3 views (images) of the same body part visit.
    A model should see ALL views of a study to make one decision, not treat
    each view as an independent example (that would let the model "cheat"
    by learning to recognize easy views).
  - Splitting by patient (not by image, not even by study) prevents the same
    person's bone anatomy from appearing in both train and test, which would
    inflate your reported accuracy in a way that would not replicate on new
    patients.
"""
import os
import re
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

BODY_PARTS = ["ELBOW", "FINGER", "FOREARM", "HAND", "HUMERUS", "SHOULDER", "WRIST"]


def parse_mura_csv(csv_path, mura_root):
    """
    MURA ships a *_image_paths.csv listing every image path, e.g.:
        MURA-v1.1/train/XR_WRIST/patient00001/study1_positive/image1.png

    We parse out: patient_id, body_part, study_id, label (0=normal,1=abnormal),
    and group rows into one row per STUDY (with a list of its image paths).
    This is the "study-level aggregation" referenced in the README.
    """
    df = pd.read_csv(csv_path, header=None, names=["path"])
    records = []
    for p in df["path"]:
        full_path = os.path.join(mura_root, p.replace("MURA-v1.1/", ""))
        m = re.search(
            r"XR_(\w+)/(patient\d+)/(study\d+_(positive|negative))/", p
        )
        if not m:
            continue
        body_part, patient_id, study_folder, label_str = m.groups()
        study_key = f"{patient_id}_{study_folder}"
        label = 1 if label_str == "positive" else 0
        records.append(
            dict(
                study_key=study_key,
                patient_id=patient_id,
                body_part=body_part,
                label=label,
                image_path=full_path,
            )
        )
    img_df = pd.DataFrame(records)

    # Collapse to one row per study, with a list of that study's image paths.
    study_df = (
        img_df.groupby(["study_key", "patient_id", "body_part", "label"])[
            "image_path"
        ]
        .apply(list)
        .reset_index()
        .rename(columns={"image_path": "image_paths"})
    )
    return study_df


def patient_level_split(study_df, tune_frac=0.15, calib_frac=0.15, seed=42):
    """
    Split the TRAIN portion of MURA into train / tune / calibration by
    PATIENT ID (never by study or image) so no patient's data leaks across
    splits.

      - train: fits the model weights
      - tune: used for early stopping + temperature scaling
      - calibration: used ONLY to compute conformal thresholds (must be
        data the model has never influenced its own weights with)

    The official MURA valid/ folder is loaded separately and used only once,
    at the very end, as the test set.
    """
    patients = study_df["patient_id"].unique()
    rng = pd.Series(patients).sample(frac=1, random_state=seed).values

    n = len(rng)
    n_tune = int(n * tune_frac)
    n_calib = int(n * calib_frac)

    tune_patients = set(rng[:n_tune])
    calib_patients = set(rng[n_tune : n_tune + n_calib])
    train_patients = set(rng[n_tune + n_calib :])

    train_df = study_df[study_df.patient_id.isin(train_patients)].reset_index(drop=True)
    tune_df = study_df[study_df.patient_id.isin(tune_patients)].reset_index(drop=True)
    calib_df = study_df[study_df.patient_id.isin(calib_patients)].reset_index(drop=True)
    return train_df, tune_df, calib_df


def get_transforms(train: bool):
    """
    Standard chest/bone X-ray augmentation. Mild geometric jitter only —
    we do NOT flip left/right for some body parts where laterality could
    matter clinically; keep it simple and document this choice in your
    report as a limitation/design decision.
    """
    base = [
        T.Resize((320, 320)),
        T.CenterCrop(288),
    ]
    if train:
        base += [
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    base += [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(base)


class MuraStudyDataset(Dataset):
    """
    Each __getitem__ returns ONE STUDY:
      - a stack of its view images (we average their model scores later,
        this class just returns the raw image tensors)
      - the study label (0/1)
      - the body part (needed for body-part-conditional calibration)

    Studies have a variable number of views (1-3). We return a fixed-size
    stack by repeating the last view if a study has fewer than `max_views`,
    which is simpler than padding+masking for a course project.
    """

    def __init__(self, study_df, train=True, max_views=3):
        self.df = study_df.reset_index(drop=True)
        self.transform = get_transforms(train)
        self.max_views = max_views

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        paths = row["image_paths"][: self.max_views]
        while len(paths) < self.max_views:
            paths.append(paths[-1])

        imgs = []
        for p in paths:
            try:
                with Image.open(p) as img:
                    imgs.append(self.transform(img.convert("RGB")))
            except Exception:
                continue
        if not imgs:
            raise ValueError(f"No valid image data for study {row['study_key']}")
        import torch

        imgs = torch.stack(imgs[: self.max_views])  # [max_views, 3, H, W]
        return imgs, row["label"], row["body_part"], row["study_key"]
