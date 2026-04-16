import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from .preprocess import preprocess_image


def build_transform(augment=False, cfg=None):
    cfg = cfg or {}
    transforms = []
    transforms.append(T.Resize((cfg.get("IMG_SIZE", 224), cfg.get("IMG_SIZE", 224))))

    if augment:
        if cfg.get("AUG_FLIP", True):
            transforms.append(T.RandomHorizontalFlip())
            transforms.append(T.RandomVerticalFlip())
        if cfg.get("AUG_ROTATE", True):
            transforms.append(T.RandomRotation(45))
        shear = cfg.get("AUG_SHEAR", 0)
        translate = cfg.get("AUG_TRANSLATE", 0.0)
        if shear or translate:
            transforms.append(T.RandomAffine(0, translate=(translate, translate), shear=shear))
        sigmas = cfg.get("AUG_BLUR_SIGMAS", (0.25, 0.5, 1.0, 2.0))
        if sigmas:
            # torchvision expects a float or (min, max)
            if isinstance(sigmas, (list, tuple)) and len(sigmas) > 2:
                sigma = (min(sigmas), max(sigmas))
            else:
                sigma = sigmas
            transforms.append(T.GaussianBlur(kernel_size=3, sigma=sigma))
        sharp = cfg.get("AUG_SHARPNESS", None)
        if sharp:
            transforms.append(T.RandomAdjustSharpness(sharpness_factor=sharp[1], p=0.5))

    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return T.Compose(transforms)


def generate_patches(img, mask, patch_size=256, stride=256, tissue_threshold=0.6):
    w, h = img.size
    patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            if mask is not None:
                m = mask[y:y+patch_size, x:x+patch_size]
                tissue_ratio = m.mean()
                if tissue_ratio < tissue_threshold:
                    continue
            patches.append((x, y))

    if not patches:
        patches = [(0, 0)]

    return patches


def extract_features(model, df, cfg, augment=False, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    X_list = []
    y_list = []

    transform = build_transform(augment=augment, cfg=cfg)

    with torch.no_grad():
        for _, row in df.iterrows():
            img_path = row["image_path"].replace("\\", "/")
            label = int(row["label"])

            pil_img = Image.open(img_path).convert("RGB")
            proc_img, mask = preprocess_image(pil_img, cfg)

            patches = generate_patches(
                proc_img,
                mask,
                patch_size=cfg.get("PATCH_SIZE", 256),
                stride=cfg.get("PATCH_STRIDE", 256),
                tissue_threshold=cfg.get("TISSUE_THRESHOLD", 0.6),
            )

            # augment multiple times if requested
            aug_times = cfg.get("AUG_PER_IMAGE", 1) if augment else 1

            for _ in range(aug_times):
                batch = []
                for (x, y) in patches:
                    patch = proc_img.crop((x, y, x + cfg.get("PATCH_SIZE", 256), y + cfg.get("PATCH_SIZE", 256)))
                    batch.append(transform(patch))

                if not batch:
                    continue

                batch_tensor = torch.stack(batch).to(device)
                feats = model(batch_tensor).cpu().numpy()

                X_list.append(feats)
                y_list.append(np.full((feats.shape[0],), label, dtype=np.int64))

    X = np.concatenate(X_list, axis=0) if X_list else np.zeros((0, 512))
    y = np.concatenate(y_list, axis=0) if y_list else np.zeros((0,), dtype=np.int64)
    return X, y
