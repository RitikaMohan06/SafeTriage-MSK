"""FracAtlas loader -> same study-level schema as parse_mura_csv.
Expected layout (Figshare release):
  FracAtlas/images/Fractured/*.jpg, FracAtlas/images/Non_fractured/*.jpg, FracAtlas/dataset.csv
dataset.csv columns used: image_id, hand, leg, hip, shoulder, mixed, fractured.
No patient IDs exist, so each image = one study (random split; state as a limitation)."""
import os
import pandas as pd
from PIL import Image

PARTS = ["hand", "leg", "hip", "shoulder"]


def _is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.load()
            img.convert("RGB")
        return True
    except Exception:
        return False


def parse_fracatlas(root):
    csv = next((os.path.join(d, "dataset.csv") for d in (root, os.path.join(root, "FracAtlas"))
                if os.path.exists(os.path.join(d, "dataset.csv"))), None)
    if csv is None:
        raise FileNotFoundError("dataset.csv not found under " + root)
    base = os.path.dirname(csv)
    df = pd.read_csv(csv)
    df.columns = [c.strip().lower() for c in df.columns]
    index = {}
    for sub, lab in (("Fractured", 1), ("Non_fractured", 0)):
        d = os.path.join(base, "images", sub)
        for f in os.listdir(d) if os.path.isdir(d) else []:
            path = os.path.join(d, f)
            if _is_valid_image(path):
                index[f] = (path, lab)
    rows, skipped = [], 0
    for _, r in df.iterrows():
        fid = str(r["image_id"])
        hit = index.get(fid)
        flags = [p for p in PARTS if int(r.get(p, 0)) == 1]
        if hit is None or len(flags) != 1 or int(r.get("mixed", 0)) == 1:
            skipped += 1  # unknown file, multi-region or unlabeled-region images
            continue
        rows.append(dict(study_key=fid, patient_id=fid, body_part=flags[0].upper(),
                         label=hit[1], image_paths=[hit[0]]))
    print(f"FracAtlas: kept {len(rows)}, skipped {skipped} (mixed/unknown region or corrupt image)")
    return pd.DataFrame(rows)


def random_test_split(df, test_frac=0.15, seed=42):
    test = df.sample(frac=test_frac, random_state=seed)
    return df.drop(test.index).reset_index(drop=True), test.reset_index(drop=True)
