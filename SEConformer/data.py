import os
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from torchvision import transforms as T


BREAKHIS_LABEL_MAP = {
    "adenosis": 0,
    "fibroadenoma": 1,
    "phyllodes_tumor": 2,
    "tubular_adenoma": 3,
    "ductal_carcinoma": 4,
    "lobular_carcinoma": 5,
    "mucinous_carcinoma": 6,
    "papillary_carcinoma": 7,
}


BREAKHIS_ALIASES = {
    "adenosis": ["adenosis", "a"],
    "fibroadenoma": ["fibroadenoma", "f"],
    "phyllodes_tumor": ["phyllodes_tumor", "pt", "phyllodes"],
    "tubular_adenoma": ["tubular_adenoma", "ta"],
    "ductal_carcinoma": ["ductal_carcinoma", "dc"],
    "lobular_carcinoma": ["lobular_carcinoma", "lc"],
    "mucinous_carcinoma": ["mucinous_carcinoma", "mc"],
    "papillary_carcinoma": ["papillary_carcinoma", "pc"],
}


def build_train_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(15),
        T.RandomAffine(0, translate=(0.05, 0.05), shear=10),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])


def build_eval_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])


def load_rgb_pil(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def load_dicom_pil(dicom_path: str) -> Image.Image:
    try:
        import pydicom
    except Exception as exc:
        raise ImportError(
            "pydicom nao esta instalado. Instale com: pip install pydicom"
        ) from exc

    ds = pydicom.dcmread(dicom_path)
    arr = ds.pixel_array.astype(np.float32)

    slope = getattr(ds, "RescaleSlope", 1.0)
    intercept = getattr(ds, "RescaleIntercept", 0.0)
    arr = arr * float(slope) + float(intercept)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr

    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    arr = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(arr)


class ImageClassificationDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_loader: Callable[[str], Image.Image], train=True, img_size=224):
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_loader = image_loader
        self.transforms = build_train_transform(img_size) if train else build_eval_transform(img_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self.image_loader(row["image_path"])
        image = self.transforms(image)
        label = int(row["label"])
        return image, torch.tensor(label, dtype=torch.long)


def _parse_breakhis_label(root_lower: str):
    if "benign" in root_lower and "malignant" not in root_lower:
        return 0, "benign"
    if "malignant" in root_lower:
        return 1, "malignant"

    for class_name, aliases in BREAKHIS_ALIASES.items():
        if any(alias in root_lower for alias in aliases):
            return BREAKHIS_LABEL_MAP[class_name], class_name
    return None, None


def build_breakhis_dataframe(base_path: str, mode="binary"):
    image_paths = []
    labels = []
    class_names = []

    for root, _dirs, files in os.walk(base_path):
        for file_name in files:
            if not file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                continue

            full_path = os.path.join(root, file_name)
            root_lower = root.lower().replace("\\", "/")

            if mode == "binary":
                if "benign" in root_lower:
                    label = 0
                    class_name = "benign"
                elif "malignant" in root_lower:
                    label = 1
                    class_name = "malignant"
                else:
                    continue
            else:
                label, class_name = _parse_breakhis_label(root_lower)
                if class_name in {"benign", "malignant"} or label is None:
                    continue

            image_paths.append(full_path)
            labels.append(label)
            class_names.append(class_name)

    return pd.DataFrame({
        "image_path": image_paths,
        "label": labels,
        "class_name": class_names,
    })


def build_breakhis_csv(base_path: str, out_csv="breakhis_folds.csv", mode="binary", n_splits=5, random_state=42):
    df = build_breakhis_dataframe(base_path, mode=mode)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    df["fold"] = -1

    for fold, (_train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
        df.loc[val_idx, "fold"] = fold

    df.to_csv(out_csv, index=False)
    print("CSV criado com sucesso!", out_csv)
    return df


def _parse_birads(value):
    try:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        parts = text.replace(" ", "").split("-")
        numbers = [int(part) for part in parts if part.isdigit()]
        if numbers:
            return max(numbers)
        for char in text:
            if char.isdigit():
                return int(char)
    except Exception:
        return None
    return None


def build_inbreast_csv(
    inbreast_csv_path: str,
    dicom_dir: str,
    out_csv="inbreast_folds.csv",
    mode="binary",
    birads_threshold=4,
    n_splits=5,
    random_state=42,
):
    df = pd.read_csv(inbreast_csv_path, sep=";")
    dicom_dir = os.path.abspath(dicom_dir)

    dicom_map = {}
    for file_name in os.listdir(dicom_dir):
        if file_name.lower().endswith(".dcm"):
            prefix = file_name.split("_")[0]
            dicom_map[prefix] = os.path.join(dicom_dir, file_name)

    rows = []
    for _, row in df.iterrows():
        file_id = str(row["File Name"])
        birads = _parse_birads(row.get("Bi-Rads"))
        if birads is None or file_id not in dicom_map:
            continue

        if mode == "multiclass":
            label = birads - 1
            if label < 0 or label > 5:
                continue
            class_name = f"birads_{birads}"
        else:
            label = 1 if birads >= birads_threshold else 0
            class_name = "malignant" if label == 1 else "benign"

        rows.append({
            "image_path": dicom_map[file_id],
            "label": label,
            "birads": birads,
            "class_name": class_name,
        })

    out_df = pd.DataFrame(rows)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    out_df["fold"] = -1

    for fold, (_train_idx, val_idx) in enumerate(skf.split(out_df, out_df["label"])):
        out_df.loc[val_idx, "fold"] = fold

    out_df.to_csv(out_csv, index=False)
    print("CSV criado com sucesso!", out_csv)
    return out_df


def split_dataframe_holdout(dataframe: pd.DataFrame, val_fraction=0.2, seed=42):
    shuffled = dataframe.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_size = max(1, int(len(shuffled) * val_fraction))
    val_df = shuffled.iloc[:val_size].copy()
    train_df = shuffled.iloc[val_size:].copy()
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
