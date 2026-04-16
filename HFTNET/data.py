import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


def parse_breakhis_dataset(base_path):
    image_paths = []
    labels = []
    class_names = []

    label_map = {
        "adenosis": 0,
        "fibroadenoma": 1,
        "phyllodes_tumor": 2,
        "tubular_adenoma": 3,
        "ductal_carcinoma": 4,
        "lobular_carcinoma": 5,
        "mucinous_carcinoma": 6,
        "papillary_carcinoma": 7,
    }

    aliases = {
        "adenosis": ["adenosis", "A"],
        "fibroadenoma": ["fibroadenoma", "F"],
        "phyllodes_tumor": ["phyllodes_tumor", "PT", "phyllodes"],
        "tubular_adenoma": ["tubular_adenoma", "TA"],
        "ductal_carcinoma": ["ductal_carcinoma", "DC"],
        "lobular_carcinoma": ["lobular_carcinoma", "LC"],
        "mucinous_carcinoma": ["mucinous_carcinoma", "MC"],
        "papillary_carcinoma": ["papillary_carcinoma", "PC"],
    }

    for root, _dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
                full_path = os.path.join(root, file)
                root_lower = root.lower().replace("\\", "/")

                found_label = None
                found_name = None
                for cls_name, terms in aliases.items():
                    if any(term.lower() in root_lower for term in terms):
                        found_label = label_map[cls_name]
                        found_name = cls_name
                        break

                if found_label is not None:
                    image_paths.append(full_path)
                    labels.append(found_label)
                    class_names.append(found_name)

    df = pd.DataFrame({
        "image_path": image_paths,
        "label": labels,
        "class_name": class_names,
    })
    return df


def build_folds_csv(df, csv_path, n_splits=5, seed=42):
    if "fold" not in df.columns:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        df["fold"] = -1
        for fold, (_train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
            df.loc[val_idx, "fold"] = fold

    df.to_csv(csv_path, index=False)
    print("CSV salvo em:", csv_path)
    return df


class HistologyDataset(Dataset):
    def __init__(self, csv_file, fold, train=True, img_size=224):
        self.df = pd.read_csv(csv_file)

        if train:
            self.df = self.df[self.df["fold"] != fold].reset_index(drop=True)
        else:
            self.df = self.df[self.df["fold"] == fold].reset_index(drop=True)

        if train:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.2),
                T.RandomRotation(20),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"].replace("\\", "/")
        label = int(row["label"])

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
