import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve,
)

from .io_utils import save_fig, save_metrics


def evaluate(model, loader, device, plot=True, save_dir=None, prefix="val", out_dir=None):
    model.eval()

    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)

            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp)

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "f1_score": float(f1),
        "auc": float(auc),
    }

    print("RESULTADOS:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    if plot and save_dir:
        # Confusion Matrix
        fig = plt.figure()
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion Matrix")
        save_fig(fig, save_dir, f"{prefix}_confusion_matrix")
        plt.show()
        plt.close(fig)

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig = plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], "--")
        plt.title("ROC Curve")
        save_fig(fig, save_dir, f"{prefix}_roc")
        plt.show()
        plt.close(fig)

        # Precision-Recall
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        fig = plt.figure()
        plt.plot(recall, precision)
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        save_fig(fig, save_dir, f"{prefix}_precision_recall")
        plt.show()
        plt.close(fig)

        # Prob distribution
        fig = plt.figure()
        plt.hist(y_prob[y_true == 0], bins=30, alpha=0.5, label="Benign")
        plt.hist(y_prob[y_true == 1], bins=30, alpha=0.5, label="Malignant")
        plt.legend()
        plt.title("Distribuicao de Probabilidades")
        save_fig(fig, save_dir, f"{prefix}_prob_distribution")
        plt.show()
        plt.close(fig)

    if out_dir:
        save_metrics(metrics, out_dir, f"{prefix}_metrics")

    return metrics
