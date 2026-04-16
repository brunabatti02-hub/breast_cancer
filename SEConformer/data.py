import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from torchvision import transforms as T
import torch
import numpy as np


class BreastCancerDataset(Dataset):
    def __init__(self, csv_file, fold, train=True, img_size=128):
        self.df = pd.read_csv(csv_file)

        if train:
            self.df = self.df[self.df["fold"] != fold]
        else:
            self.df = self.df[self.df["fold"] == fold]

        self.df = self.df.reset_index(drop=True)

        if train:
            self.transforms = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(p=0.2),
                T.RandomRotation(15),
                T.RandomAffine(0, translate=(0.05, 0.05), shear=10),
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "image_path"]
        label = self.df.loc[idx, "label"]

        image = Image.open(img_path).convert("RGB")
        image = self.transforms(image)

        return image, torch.tensor(label, dtype=torch.long)


def build_folds_csv(base_path, out_csv="Folds.csv", n_splits=5, random_state=42):
    image_paths = []
    labels = []

    for root, _dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".png"):
                full_path = os.path.join(root, file)

                root_lower = root.lower()
                if "benign" in root_lower:
                    label = 0
                elif "malignant" in root_lower:
                    label = 1
                else:
                    continue

                image_paths.append(full_path)
                labels.append(label)

    df = pd.DataFrame({
        "image_path": image_paths,
        "label": labels,
    })

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    df["fold"] = -1

    for fold, (_train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
        df.loc[val_idx, "fold"] = fold

    df.to_csv(out_csv, index=False)
    print("CSV criado com sucesso!", out_csv)
    return df


def _parse_birads(value):
    try:
        s = str(value)
        parts = s.replace(" ", "").split("-")
        nums = [int(p) for p in parts if p.isdigit()]
        if nums:
            return max(nums)
        # fallback: pick first digit in string
        for ch in s:
            if ch.isdigit():
                return int(ch)
    except Exception:
        pass
    return None


def build_inbreast_csv(inbreast_csv_path, dicom_dir, out_csv="INbreast_Folds.csv",
                       n_splits=5, random_state=42, birads_threshold=4, mode="binary"):
    df = pd.read_csv(inbreast_csv_path, sep=";")

    # Map DICOM files by numeric prefix
    dicom_dir = os.path.abspath(dicom_dir)
    dicom_files = [f for f in os.listdir(dicom_dir) if f.lower().endswith(".dcm")]
    dicom_map = {}
    for fname in dicom_files:
        prefix = fname.split("_")[0]
        dicom_map[prefix] = os.path.join(dicom_dir, fname)

    image_paths = []
    labels = []
    birads_list = []

    for _, row in df.iterrows():
        file_id = str(row["File Name"])
        birads = _parse_birads(row.get("Bi-Rads", None))
        if birads is None:
            continue

        dcm_path = dicom_map.get(file_id)
        if dcm_path is None:
            continue

        if mode == "multiclass":
            # map BI-RADS 1..6 to 0..5
            label = int(birads) - 1
            if label < 0 or label > 5:
                continue
        else:
            label = 1 if birads >= birads_threshold else 0
        image_paths.append(dcm_path)
        labels.append(label)
        birads_list.append(birads)

    out_df = pd.DataFrame({
        "image_path": image_paths,
        "label": labels,
        "birads": birads_list,
    })

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    out_df["fold"] = -1
    for fold, (_train_idx, val_idx) in enumerate(skf.split(out_df, out_df["label"])):
        out_df.loc[val_idx, "fold"] = fold

    out_df.to_csv(out_csv, index=False)
    print("CSV criado com sucesso!", out_csv)
    return out_df


def _load_dicom_as_pil(dicom_path):
    try:
        import pydicom
    except Exception as e:
        raise ImportError(
            "pydicom não está instalado. Instale com: pip install pydicom"
        ) from e

    ds = pydicom.dcmread(dicom_path)
    arr = ds.pixel_array.astype(np.float32)

    # apply rescale if present
    slope = getattr(ds, "RescaleSlope", 1.0)
    intercept = getattr(ds, "RescaleIntercept", 0.0)
    arr = arr * float(slope) + float(intercept)

    # invert if MONOCHROME1
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr

    # normalize to 0-255
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

    # convert to 3-channel
    arr = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(arr)


class InbreastDataset(Dataset):
    def __init__(self, csv_file, fold, train=True, img_size=224):
        self.df = pd.read_csv(csv_file)

        if train:
            self.df = self.df[self.df["fold"] != fold]
        else:
            self.df = self.df[self.df["fold"] == fold]

        self.df = self.df.reset_index(drop=True)

        if train:
            self.transforms = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(p=0.2),
                T.RandomRotation(15),
                T.RandomAffine(0, translate=(0.05, 0.05), shear=10),
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "image_path"]
        label = self.df.loc[idx, "label"]

        image = _load_dicom_as_pil(img_path)
        image = self.transforms(image)

        return image, torch.tensor(label, dtype=torch.long)
