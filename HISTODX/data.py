import os
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T


VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def build_histology_train_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])


def build_mammography_train_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(10),
        T.RandomAffine(0, translate=(0.05, 0.05), shear=8),
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


class BreastCancerTorchDataset(Dataset):
    def __init__(self, dataframe, image_loader: Callable[[str], Image.Image], img_size=224, training=False, domain="histology"):
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_loader = image_loader

        if training and domain == "histology":
            self.transforms = build_histology_train_transform(img_size)
        elif training and domain == "mammography":
            self.transforms = build_mammography_train_transform(img_size)
        else:
            self.transforms = build_eval_transform(img_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.loc[idx, "image_path"]
        label = int(self.df.loc[idx, "label"])
        image = self.image_loader(image_path)
        image = self.transforms(image)
        return image, torch.tensor(label, dtype=torch.long)


def collect_breakhis_images(dataset_path, mode="binary"):
    records = []

    for root, _dirs, files in os.walk(dataset_path):
        for file_name in files:
            if not file_name.lower().endswith(VALID_EXTENSIONS):
                continue

            file_path = os.path.join(root, file_name)
            lower_path = file_path.lower()

            if mode == "binary":
                if "benign" in lower_path:
                    label = 0
                    class_name = "benign"
                elif "malignant" in lower_path:
                    label = 1
                    class_name = "malignant"
                else:
                    continue
            else:
                continue

            magnification = None
            for mag in ["40x", "100x", "200x", "400x"]:
                if mag in lower_path:
                    magnification = mag
                    break

            records.append({
                "image_path": file_path,
                "label": label,
                "class_name": class_name,
                "magnification": magnification if magnification else "unknown",
            })

    return pd.DataFrame(records)


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


def build_inbreast_dataframe(inbreast_csv_path, dicom_dir, mode="binary", birads_threshold=4):
    df = pd.read_csv(inbreast_csv_path, sep=";")
    dicom_dir = os.path.abspath(dicom_dir)

    dicom_map = {}
    for file_name in os.listdir(dicom_dir):
        if file_name.lower().endswith(".dcm"):
            prefix = file_name.split("_")[0]
            dicom_map[prefix] = os.path.join(dicom_dir, file_name)

    records = []
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

        records.append({
            "image_path": dicom_map[file_id],
            "label": label,
            "class_name": class_name,
            "birads": birads,
        })

    return pd.DataFrame(records)


def split_train_val_test(df, seed=42, test_size=0.30):
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=seed,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def split_dataframe_holdout(df, val_fraction=0.2, seed=42):
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_size = max(1, int(len(shuffled) * val_fraction))
    val_df = shuffled.iloc[:val_size].copy()
    train_df = shuffled.iloc[val_size:].copy()
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def build_folds_csv(df, csv_path, n_splits=5, seed=42):
    if "fold" not in df.columns:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        df["fold"] = -1
        for fold, (_train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
            df.loc[val_idx, "fold"] = fold

    df.to_csv(csv_path, index=False)
    print("CSV salvo em:", csv_path)
    return df


def make_loaders(
    train_df,
    val_df,
    test_df=None,
    image_loader=load_rgb_pil,
    img_size=224,
    batch_size=32,
    balance=True,
    domain="histology",
):
    train_ds = BreastCancerTorchDataset(train_df, image_loader=image_loader, img_size=img_size, training=True, domain=domain)
    val_ds = BreastCancerTorchDataset(val_df, image_loader=image_loader, img_size=img_size, training=False, domain=domain)

    if balance:
        labels = train_df["label"].values
        class_counts = np.bincount(labels)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    test_loader = None
    if test_df is not None:
        test_ds = BreastCancerTorchDataset(test_df, image_loader=image_loader, img_size=img_size, training=False, domain=domain)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def compute_class_weights(train_df):
    labels = train_df["label"].values.astype(np.int64)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    return torch.tensor(class_weights, dtype=torch.float32)
