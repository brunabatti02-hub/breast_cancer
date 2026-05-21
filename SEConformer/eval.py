import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
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

from .io_utils import save_fig, save_metrics


def evaluate(model, loader, device, num_classes=2, plot=True, save_dir=None, prefix="val", out_dir=None):
    model.eval()

    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

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
        cm = confusion_matrix(y_true, y_pred)

    print("RESULTADOS:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    if plot and save_dir:
        fig = plt.figure()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        save_fig(fig, save_dir, f"{prefix}_confusion_matrix")
        plt.show()
        plt.close(fig)

        if num_classes == 2:
            positive_prob = y_prob[:, 1]

            fpr, tpr, _ = roc_curve(y_true, positive_prob)
            fig = plt.figure()
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1], "--")
            plt.title("ROC Curve")
            save_fig(fig, save_dir, f"{prefix}_roc")
            plt.show()
            plt.close(fig)

            precision, recall, _ = precision_recall_curve(y_true, positive_prob)
            fig = plt.figure()
            plt.plot(recall, precision)
            plt.title("Precision-Recall Curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            save_fig(fig, save_dir, f"{prefix}_precision_recall")
            plt.show()
            plt.close(fig)

            fig = plt.figure()
            plt.hist(positive_prob[y_true == 0], bins=30, alpha=0.5, label="Class 0")
            plt.hist(positive_prob[y_true == 1], bins=30, alpha=0.5, label="Class 1")
            plt.legend()
            plt.title("Distribuicao de Probabilidades")
            save_fig(fig, save_dir, f"{prefix}_prob_distribution")
            plt.show()
            plt.close(fig)

    if out_dir:
        save_metrics(metrics, out_dir, f"{prefix}_metrics")
        pd.DataFrame(cm).to_csv(f"{out_dir}/{prefix}_confusion_matrix.csv", index=False)
        prob_cols = [f"prob_class_{idx}" for idx in range(y_prob.shape[1])]
        predictions_df = pd.DataFrame(y_prob, columns=prob_cols)
        predictions_df.insert(0, "y_pred", y_pred)
        predictions_df.insert(0, "y_true", y_true)
        predictions_df.to_csv(f"{out_dir}/{prefix}_predictions.csv", index=False)

    return metrics
