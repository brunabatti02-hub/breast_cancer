import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

from .config import NUM_CLASSES, IMG_SIZE, BATCH_SIZE, EPOCHS, LR, NUM_WORKERS, SEED
from .data import HistologyDataset
from .model import HFTNet
from .io_utils import save_fig, save_metrics


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_multiclass_metrics(y_true, y_pred, y_prob, num_classes):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    try:
        auc_macro = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
    except Exception:
        auc_macro = np.nan

    return {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "auc_macro_ovr": auc_macro,
        "confusion_matrix": cm,
    }


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


def plot_roc_multiclass(y_true, y_prob, num_classes, plots_dir=None, show=True):
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig = plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Classe {i} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC One-vs-Rest")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    if plots_dir:
        save_fig(fig, plots_dir, "roc_ovr")
    if show:
        plt.show()
    plt.close(fig)


def plot_pr_multiclass(y_true, y_prob, num_classes, plots_dir=None, show=True):
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig = plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        plt.plot(recall, precision, label=f"Classe {i}")

    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    if plots_dir:
        save_fig(fig, plots_dir, "pr_curve")
    if show:
        plt.show()
    plt.close(fig)


def plot_probability_histograms(y_true, y_prob, plots_dir=None, show=True):
    max_prob = np.max(y_prob, axis=1)

    fig = plt.figure(figsize=(8, 5))
    plt.hist(max_prob, bins=30, alpha=0.8)
    plt.title("Distribuicao da confianca do modelo")
    plt.xlabel("Maior probabilidade predita")
    plt.ylabel("Frequencia")
    if plots_dir:
        save_fig(fig, plots_dir, "confidence_distribution")
    if show:
        plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    correct = (np.argmax(y_prob, axis=1) == np.array(y_true))
    plt.hist(max_prob[correct], bins=30, alpha=0.6, label="Corretas")
    plt.hist(max_prob[~correct], bins=30, alpha=0.6, label="Erradas")
    plt.title("Confianca: corretas vs erradas")
    plt.xlabel("Maior probabilidade predita")
    plt.ylabel("Frequencia")
    plt.legend()
    if plots_dir:
        save_fig(fig, plots_dir, "confidence_correct_vs_wrong")
    if show:
        plt.show()
    plt.close(fig)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    preds_all, labels_all = [], []

    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = outputs.argmax(dim=1)
        preds_all.extend(preds.detach().cpu().numpy())
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
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            running_loss += loss.item()

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
            probs_all.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(labels_all, preds_all)
    metrics = compute_multiclass_metrics(
        np.array(labels_all),
        np.array(preds_all),
        np.array(probs_all),
        num_classes,
    )

    return epoch_loss, epoch_acc, metrics, np.array(labels_all), np.array(preds_all), np.array(probs_all)


def run_training(csv_path, fold=0, epochs=EPOCHS, run_dirs=None, device=None):
    seed_everything()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if run_dirs is None:
        from .io_utils import make_run_dirs
        run_dirs = make_run_dirs()

    train_ds = HistologyDataset(csv_path, fold=fold, train=True, img_size=IMG_SIZE)
    val_ds = HistologyDataset(csv_path, fold=fold, train=False, img_size=IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = HFTNet(num_classes=NUM_CLASSES, pretrained=True, freeze_backbones=False).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # class balancing
    counts = train_ds.df["label"].value_counts().sort_index()
    class_weights = (1.0 / counts).values
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_acc = 0.0
    best_state = None
    best_outputs = None

    for epoch in range(epochs):
        print(f"\n===== EPOCA {epoch+1}/{epochs} =====")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, metrics, y_true, y_pred, y_prob = validate_one_epoch(
            model, val_loader, criterion, NUM_CLASSES, device
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"Precision Macro: {metrics['precision_macro']:.4f}")
        print(f"Recall Macro:    {metrics['recall_macro']:.4f}")
        print(f"F1 Macro:        {metrics['f1_macro']:.4f}")
        print(f"AUC Macro OVR:   {metrics['auc_macro_ovr']:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
            best_outputs = (y_true, y_pred, y_prob, metrics)

    model.load_state_dict(best_state)

    y_true, y_pred, y_prob, metrics = best_outputs
    save_metrics({
        "accuracy": float(metrics["accuracy"]),
        "precision_macro": float(metrics["precision_macro"]),
        "recall_macro": float(metrics["recall_macro"]),
        "f1_macro": float(metrics["f1_macro"]),
        "auc_macro_ovr": float(metrics["auc_macro_ovr"]),
    }, run_dirs["out_dir"], "final_metrics")

    class_names = [
        "Adenosis",
        "Fibroadenoma",
        "Phyllodes",
        "Tubular Adenoma",
        "Ductal Carcinoma",
        "Lobular Carcinoma",
        "Mucinous Carcinoma",
        "Papillary Carcinoma",
    ]
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    import pandas as pd
    pd.DataFrame(report).transpose().to_csv(os.path.join(run_dirs["out_dir"], "classification_report.csv"), index=True)

    plot_training_curves(history, plots_dir=run_dirs["plots_dir"], show=True)
    plot_confusion_matrix(metrics["confusion_matrix"], class_names=class_names, title="HFT-Net - Confusion Matrix", plots_dir=run_dirs["plots_dir"], show=True)
    plot_roc_multiclass(y_true, y_prob, NUM_CLASSES, plots_dir=run_dirs["plots_dir"], show=True)
    plot_pr_multiclass(y_true, y_prob, NUM_CLASSES, plots_dir=run_dirs["plots_dir"], show=True)
    plot_probability_histograms(y_true, y_prob, plots_dir=run_dirs["plots_dir"], show=True)

    model_path = os.path.join(run_dirs["models_dir"], "hftnet_breakhis.pth")
    torch.save(model.state_dict(), model_path)
    print("Modelo salvo como", model_path)

    return model, history, best_outputs, run_dirs
