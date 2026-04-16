import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def collect_breakhis_images(dataset_path):
    records = []

    for root, _dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(VALID_EXTENSIONS):
                file_path = os.path.join(root, file)
                lower_path = file_path.lower()

                if "benign" in lower_path:
                    label = 0
                    class_name = "benign"
                elif "malignant" in lower_path:
                    label = 1
                    class_name = "malignant"
                else:
                    continue

                magnification = None
                for mag in ["40x", "100x", "200x", "400x"]:
                    if mag in lower_path:
                        magnification = mag
                        break

                records.append({
                    "path": file_path,
                    "label": label,
                    "class_name": class_name,
                    "magnification": magnification if magnification else "unknown",
                })

    df = pd.DataFrame(records)
    return df


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
    return train_df, val_df, test_df


class BreastCancerTorchDataset(Dataset):
    def __init__(self, dataframe, img_size=224, training=False):
        self.df = dataframe.reset_index(drop=True)
        self.training = training

        if training:
            self.transforms = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.1, contrast=0.1),
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
        img_path = self.df.loc[idx, "path"]
        label = int(self.df.loc[idx, "label"])

        image = Image.open(img_path).convert("RGB")
        image = self.transforms(image)

        return image, torch.tensor(label, dtype=torch.long)


def make_loaders(train_df, val_df, test_df, img_size=224, batch_size=32, balance=True):
    train_ds = BreastCancerTorchDataset(train_df, img_size=img_size, training=True)
    val_ds = BreastCancerTorchDataset(val_df, img_size=img_size, training=False)
    test_ds = BreastCancerTorchDataset(test_df, img_size=img_size, training=False)

    if balance:
        labels = train_df["label"].values
        class_counts = np.bincount(labels)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[labels]
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def compute_class_weights(train_df):
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_df["label"].values,
    )
    return {
        0: float(class_weights_array[0]),
        1: float(class_weights_array[1]),
    }
