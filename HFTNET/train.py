import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .config import BATCH_SIZE, EPOCHS, IMG_SIZE, LR, NUM_WORKERS, SEED
from .data import (
    ImageClassificationDataset,
    build_inbreast_csv,
    load_dicom_pil,
    load_rgb_pil,
    parse_breakhis_dataset,
    split_dataframe_holdout,
)
from .io_utils import find_best_previous_run, make_run_dirs, save_fig, save_metrics
from .model import HFTNet


BREAKHIS_CLASS_NAMES = [
    "Adenosis",
    "Fibroadenoma",
    "Phyllodes",
    "Tubular Adenoma",
    "Ductal Carcinoma",
    "Lobular Carcinoma",
    "Mucinous Carcinoma",
    "Papillary Carcinoma",
]


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_train_loader(dataset, labels, batch_size, device, balance=True):
    if not balance:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS), None

    labels = np.asarray(labels, dtype=np.int64)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=NUM_WORKERS)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    return loader, weight_tensor


def _filter_compatible_state_dict(model, state_dict):
    model_state = model.state_dict()
    compatible = {}
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible[key] = value
    return compatible


def _resolve_transfer_weights(weights_path, checkpoint_name):
    if weights_path:
        return weights_path

    best_run = find_best_previous_run(checkpoint_name=checkpoint_name)
    print(
        "Usando melhor baseline anterior para transfer learning:",
        best_run["model_path"],
        f"({best_run['metric_name']}={best_run['metric_value']:.4f})",
    )
    return best_run["model_path"]


def compute_metrics(y_true, y_pred, y_prob, num_classes):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    if num_classes == 2:
        positive_prob = y_prob[:, 1]
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics.update({
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "specificity": float(specificity),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "auc": float(roc_auc_score(y_true, positive_prob)),
        })
    else:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        metrics.update({
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "auc_macro_ovr": float(roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")),
        })

    return metrics


def plot_training_curves(history, plots_dir=None, show=True):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig = plt.figure(figsize=(12, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title("Loss por epoca")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.legend()
    if plots_dir:
        save_fig(fig, plots_dir, "loss_curves")
    if show:
        plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=(12, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.title("Accuracy por epoca")
    plt.xlabel("Epoca")
    plt.ylabel("Accuracy")
    plt.legend()
    if plots_dir:
        save_fig(fig, plots_dir, "accuracy_curves")
    if show:
        plt.show()
    plt.close(fig)


def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix", plots_dir=None, show=True):
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    if plots_dir:
        save_fig(fig, plots_dir, "confusion_matrix")
    if show:
        plt.show()
    plt.close(fig)


def plot_roc_curves(y_true, y_prob, num_classes, plots_dir=None, show=True):
    fig = plt.figure(figsize=(8, 6))

    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = roc_auc_score(y_true, y_prob[:, 1])
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    else:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
            plt.plot(fpr, tpr, label=f"Classe {i} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    if plots_dir:
        save_fig(fig, plots_dir, "roc")
    if show:
        plt.show()
    plt.close(fig)


def plot_pr_curves(y_true, y_prob, num_classes, plots_dir=None, show=True):
    fig = plt.figure(figsize=(8, 6))

    if num_classes == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
        plt.plot(recall, precision, label="Classe positiva")
    else:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        for i in range(num_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
            plt.plot(recall, precision, label=f"Classe {i}")

    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    if plots_dir:
        save_fig(fig, plots_dir, "precision_recall")
    if show:
        plt.show()
    plt.close(fig)


def _default_class_names(num_classes):
    if num_classes == 8:
        return BREAKHIS_CLASS_NAMES
    if num_classes == 2:
        return ["Benign", "Malignant"]
    return [f"Class {idx}" for idx in range(num_classes)]


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    preds_all, labels_all = [], []

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds_all.extend(outputs.argmax(dim=1).detach().cpu().numpy())
        labels_all.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(labels_all, preds_all)
    return epoch_loss, epoch_acc


def validate_one_epoch(model, loader, criterion, num_classes, device):
    model.eval()
    running_loss = 0.0
    preds_all, labels_all, probs_all = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)

            running_loss += loss.item()
            preds_all.extend(outputs.argmax(dim=1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
            probs_all.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(labels_all, preds_all)

    y_true = np.array(labels_all)
    y_pred = np.array(preds_all)
    y_prob = np.array(probs_all)
    metrics = compute_metrics(y_true, y_pred, y_prob, num_classes)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    return epoch_loss, epoch_acc, metrics, y_true, y_pred, y_prob


def train_from_dataframes(
    train_df,
    val_df,
    image_loader,
    num_classes,
    domain,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    device=None,
    run_dirs=None,
    img_size=IMG_SIZE,
    balance=True,
    weights_path=None,
    freeze_backbones=False,
    freeze_except_classifier=False,
    model_filename="hftnet_model.pth",
    class_names=None,
):
    seed_everything()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if run_dirs is None:
        run_dirs = make_run_dirs()

    train_ds = ImageClassificationDataset(train_df, image_loader=image_loader, train=True, img_size=img_size, domain=domain)
    val_ds = ImageClassificationDataset(val_df, image_loader=image_loader, train=False, img_size=img_size, domain=domain)

    train_loader, loss_weights = _build_train_loader(
        train_ds,
        train_df["label"].values,
        batch_size=batch_size,
        device=device,
        balance=balance,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    model = HFTNet(num_classes=num_classes, pretrained=True, freeze_backbones=freeze_backbones).to(device)
    if weights_path:
        state = torch.load(weights_path, map_location=device)
        compatible = _filter_compatible_state_dict(model, state)
        model.load_state_dict(compatible, strict=False)

    if freeze_except_classifier:
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_score = -1.0
    best_state = None
    best_outputs = None

    for epoch in range(epochs):
        print(f"\n===== EPOCA {epoch + 1}/{epochs} =====")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, metrics, y_true, y_pred, y_prob = validate_one_epoch(
            model, val_loader, criterion, num_classes, device
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        for key, value in metrics.items():
            if key != "confusion_matrix":
                print(f"{key}: {value:.4f}")

        if val_acc > best_score:
            best_score = val_acc
            best_state = model.state_dict()
            best_outputs = (y_true, y_pred, y_prob, metrics)

    model.load_state_dict(best_state)
    y_true, y_pred, y_prob, metrics = best_outputs

    final_metrics = {key: float(value) for key, value in metrics.items() if key != "confusion_matrix"}
    save_metrics(final_metrics, run_dirs["out_dir"], "final_metrics")
    pd.DataFrame(metrics["confusion_matrix"]).to_csv(
        os.path.join(run_dirs["out_dir"], "final_confusion_matrix.csv"),
        index=False,
    )
    prob_cols = [f"prob_class_{idx}" for idx in range(y_prob.shape[1])]
    predictions_df = pd.DataFrame(y_prob, columns=prob_cols)
    predictions_df.insert(0, "y_pred", y_pred)
    predictions_df.insert(0, "y_true", y_true)
    predictions_df.to_csv(os.path.join(run_dirs["out_dir"], "final_predictions.csv"), index=False)

    report = pd.DataFrame(
        classification_report_safe(y_true, y_pred, class_names or _default_class_names(num_classes))
    )
    report.transpose().to_csv(os.path.join(run_dirs["out_dir"], "classification_report.csv"), index=True)

    plot_training_curves(history, plots_dir=run_dirs["plots_dir"], show=True)
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names=class_names or _default_class_names(num_classes),
        title="HFTNet - Confusion Matrix",
        plots_dir=run_dirs["plots_dir"],
        show=True,
    )
    plot_roc_curves(y_true, y_prob, num_classes, plots_dir=run_dirs["plots_dir"], show=True)
    plot_pr_curves(y_true, y_prob, num_classes, plots_dir=run_dirs["plots_dir"], show=True)

    model_path = os.path.join(run_dirs["models_dir"], model_filename)
    torch.save(model.state_dict(), model_path)
    print("Modelo salvo como", model_path)

    return model, history, best_outputs, run_dirs


def classification_report_safe(y_true, y_pred, class_names):
    from sklearn.metrics import classification_report
    return classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)


def run_breakhis_baseline_holdout(
    base_path,
    mode="multiclass",
    val_fraction=0.2,
    seed=SEED,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    device=None,
    run_dirs=None,
    img_size=IMG_SIZE,
    balance=True,
):
    df = parse_breakhis_dataset(base_path, mode=mode)
    train_df, val_df = split_dataframe_holdout(df, val_fraction=val_fraction, seed=seed)
    num_classes = int(df["label"].nunique())
    return train_from_dataframes(
        train_df=train_df,
        val_df=val_df,
        image_loader=load_rgb_pil,
        num_classes=num_classes,
        domain="histology",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        run_dirs=run_dirs,
        img_size=img_size,
        balance=balance,
        model_filename="hftnet_breakhis.pth",
        class_names=_default_class_names(num_classes),
    )


def run_breakhis_baseline_fold(
    csv_path,
    fold=0,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    device=None,
    run_dirs=None,
    img_size=IMG_SIZE,
    balance=True,
):
    df = pd.read_csv(csv_path)
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)
    num_classes = int(df["label"].nunique())
    return train_from_dataframes(
        train_df=train_df,
        val_df=val_df,
        image_loader=load_rgb_pil,
        num_classes=num_classes,
        domain="histology",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        run_dirs=run_dirs,
        img_size=img_size,
        balance=balance,
        model_filename="hftnet_breakhis.pth",
        class_names=_default_class_names(num_classes),
    )


def run_inbreast_baseline_fold(
    csv_path,
    fold=0,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    device=None,
    run_dirs=None,
    img_size=IMG_SIZE,
    balance=True,
):
    df = pd.read_csv(csv_path)
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)
    num_classes = int(df["label"].nunique())
    return train_from_dataframes(
        train_df=train_df,
        val_df=val_df,
        image_loader=load_dicom_pil,
        num_classes=num_classes,
        domain="mammography",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        run_dirs=run_dirs,
        img_size=img_size,
        balance=balance,
        model_filename="hftnet_inbreast.pth",
        class_names=_default_class_names(num_classes),
    )


def run_inbreast_baseline_holdout(
    csv_path,
    val_fraction=0.2,
    seed=SEED,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    device=None,
    run_dirs=None,
    img_size=IMG_SIZE,
    balance=True,
):
    df = pd.read_csv(csv_path)
    train_df, val_df = split_dataframe_holdout(df, val_fraction=val_fraction, seed=seed)
    num_classes = int(df["label"].nunique())
    return train_from_dataframes(
        train_df=train_df,
        val_df=val_df,
        image_loader=load_dicom_pil,
        num_classes=num_classes,
        domain="mammography",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        run_dirs=run_dirs,
        img_size=img_size,
        balance=balance,
        model_filename="hftnet_inbreast.pth",
        class_names=_default_class_names(num_classes),
    )


def run_transfer_breakhis_to_inbreast(
    inbreast_csv_path,
    histology_weights_path=None,
    val_fraction=0.2,
    seed=SEED,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    device=None,
    run_dirs=None,
    img_size=IMG_SIZE,
    balance=True,
    freeze_except_classifier=True,
):
    histology_weights_path = _resolve_transfer_weights(
        histology_weights_path,
        checkpoint_name="hftnet_breakhis.pth",
    )
    df = pd.read_csv(inbreast_csv_path)
    train_df, val_df = split_dataframe_holdout(df, val_fraction=val_fraction, seed=seed)
    num_classes = int(df["label"].nunique())
    return train_from_dataframes(
        train_df=train_df,
        val_df=val_df,
        image_loader=load_dicom_pil,
        num_classes=num_classes,
        domain="mammography",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        run_dirs=run_dirs,
        img_size=img_size,
        balance=balance,
        weights_path=histology_weights_path,
        freeze_except_classifier=freeze_except_classifier,
        model_filename="hftnet_breakhis_to_inbreast_tl.pth",
        class_names=_default_class_names(num_classes),
    )
