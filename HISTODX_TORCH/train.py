import os
import random
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .config import SEED, IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE
from .data import collect_breakhis_images, split_train_val_test, make_loaders, compute_class_weights
from .model import build_histodx_torch
from .io_utils import make_run_dirs, save_metrics, save_fig

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }
    return metrics, y_true, y_pred, y_prob


def run_histodx_torch(
    dataset_path,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    run_dirs=None,
    seed=SEED,
    device=None,
    balance=True,
):
    set_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if run_dirs is None:
        run_dirs = make_run_dirs()

    df = collect_breakhis_images(dataset_path)
    train_df, val_df, test_df = split_train_val_test(df, seed=seed)
    train_loader, val_loader, test_loader = make_loaders(
        train_df, val_df, test_df, img_size=img_size, batch_size=batch_size, balance=balance
    )

    class_weights = compute_class_weights(train_df)
    weights = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32).to(device)

    model = build_histodx_torch(num_classes=2, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights if balance else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_metrics = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for imgs, labels in tqdm(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        metrics, _, _, _ = evaluate(model, val_loader, device)
        val_metrics.append(metrics)

        print(f"\nEpoch {epoch} Loss: {avg_loss:.4f} | Val Acc: {metrics['accuracy']:.4f}")

    # Plot train loss and val accuracy
    fig = plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.title("Loss por epoca")
    plt.legend()
    save_fig(fig, run_dirs["plots_dir"], "train_loss")
    plt.show()
    plt.close(fig)

    fig = plt.figure()
    plt.plot([m["accuracy"] for m in val_metrics], label="Val Accuracy")
    plt.title("Accuracy por epoca")
    plt.legend()
    save_fig(fig, run_dirs["plots_dir"], "val_accuracy")
    plt.show()
    plt.close(fig)

    # Final eval on test
    metrics, y_true, y_pred, y_prob = evaluate(model, test_loader, device)
    save_metrics(metrics, run_dirs["out_dir"], "final_metrics")

    report = classification_report(
        y_true, y_pred, target_names=["benign", "malignant"], output_dict=True
    )
    import pandas as pd
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(run_dirs["out_dir"], "classification_report.csv"), index=True)

    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["benign", "malignant"],
        yticklabels=["benign", "malignant"],
    )
    plt.title("Matriz de Confusao")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    save_fig(fig, run_dirs["plots_dir"], "confusion_matrix")
    plt.show()
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    fig = plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.grid(True)
    save_fig(fig, run_dirs["plots_dir"], "roc_curve")
    plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=(7, 4))
    plt.hist(y_prob[y_true == 0], bins=30, alpha=0.6, label="benign")
    plt.hist(y_prob[y_true == 1], bins=30, alpha=0.6, label="malignant")
    plt.title("Distribuicao das probabilidades previstas")
    plt.xlabel("Probabilidade de malignant")
    plt.ylabel("Quantidade")
    plt.legend()
    plt.grid(True)
    save_fig(fig, run_dirs["plots_dir"], "probability_distribution")
    plt.show()
    plt.close(fig)

    # Save model
    model_path = os.path.join(run_dirs["models_dir"], "histodx_torch.pt")
    torch.save(model.state_dict(), model_path)
    print("Modelo salvo em:", model_path)

    return {
        "run_dirs": run_dirs,
        "df": df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_losses": train_losses,
        "val_metrics": val_metrics,
        "metrics": metrics,
        "model": model,
    }
