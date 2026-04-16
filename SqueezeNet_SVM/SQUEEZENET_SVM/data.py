import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def parse_breakhis_dataset(base_path, mode="binary"):
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

                if found_label is None:
                    # fallback: benign/malignant only
                    if "benign" in root_lower:
                        found_label = 0
                        found_name = "benign"
                    elif "malignant" in root_lower:
                        found_label = 1
                        found_name = "malignant"

                if found_label is not None:
                    if mode == "binary":
                        label = 0 if "benign" in root_lower else 1
                        class_name = "benign" if label == 0 else "malignant"
                    else:
                        label = found_label
                        class_name = found_name

                    image_paths.append(full_path)
                    labels.append(label)
                    class_names.append(class_name)

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
