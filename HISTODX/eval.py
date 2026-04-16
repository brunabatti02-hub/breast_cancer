import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from .io_utils import save_fig, save_metrics


def plot_training_history(history, plots_dir=None, show=True):
    metrics = ["accuracy", "loss", "precision", "recall", "auc"]

    for metric in metrics:
        fig = plt.figure(figsize=(7, 4))
        plt.plot(history[metric], label=f"train_{metric}")
        plt.plot(history[f"val_{metric}"], label=f"val_{metric}")
        plt.title(f"Treino - {metric}")
        plt.xlabel("Epoca")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        if plots_dir:
            save_fig(fig, plots_dir, f"history_{metric}")
        if show:
            plt.show()
        plt.close(fig)


def evaluate_model(model, test_ds, test_df, plots_dir=None, out_dir=None, show=True):
    y_true = test_df["label"].values
    y_prob = model.predict(test_ds).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics_summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }

    if out_dir:
        save_metrics(metrics_summary, out_dir, "final_metrics")

    report = classification_report(
        y_true,
        y_pred,
        target_names=["benign", "malignant"],
        output_dict=True,
    )
    report_df = pd.DataFrame(report).transpose()
    if out_dir:
        report_df.to_csv(f"{out_dir}/classification_report.csv", index=True)

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
    if plots_dir:
        save_fig(fig, plots_dir, "confusion_matrix")
    if show:
        plt.show()
    plt.close(fig)

    fpr, tpr, _thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    fig = plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.grid(True)
    if plots_dir:
        save_fig(fig, plots_dir, "roc_curve")
    if show:
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
    if plots_dir:
        save_fig(fig, plots_dir, "probability_distribution")
    if show:
        plt.show()
    plt.close(fig)

    return {
        "metrics": metrics_summary,
        "report_df": report_df,
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }
