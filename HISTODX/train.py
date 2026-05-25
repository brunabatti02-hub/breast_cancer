import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from torch import nn
from tqdm import tqdm

from histology_datasets import build_bach_dataframes, build_bracs_dataframes

from .config import BATCH_SIZE, EPOCHS, IMG_SIZE, LEARNING_RATE, SEED
from .data import (
    build_inbreast_dataframe,
    collect_breakhis_images,
    compute_class_weights,
    load_dicom_pil,
    load_rgb_pil,
    make_loaders,
    split_dataframe_holdout,
    split_train_val_test,
)
from .io_utils import find_best_previous_run, make_run_dirs, save_fig, save_metrics
from .model import build_histodx_torch


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    metrics = {"accuracy": float(accuracy_score(y_true, y_pred))}
    if num_classes == 2:
        metrics.update({
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": _safe_auc_binary(y_true, y_prob[:, 1]),
        })
    else:
        metrics.update({
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "auc_macro_ovr": _safe_auc_multiclass(y_true, y_prob, num_classes),
        })
    return metrics, y_true, y_pred, y_prob


def _safe_auc_binary(y_true, positive_prob):
    try:
        return float(roc_auc_score(y_true, positive_prob))
    except ValueError:
        return 0.0


def _safe_auc_multiclass(y_true, y_prob, num_classes):
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        return float(roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr"))
    except ValueError:
        return 0.0


def _plot_binary_outputs(y_true, y_pred, y_prob, run_dirs):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
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

    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = _safe_auc_binary(y_true, y_prob[:, 1])
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
    plt.hist(y_prob[y_true == 0, 1], bins=30, alpha=0.6, label="benign")
    plt.hist(y_prob[y_true == 1, 1], bins=30, alpha=0.6, label="malignant")
    plt.title("Distribuicao das probabilidades previstas")
    plt.xlabel("Probabilidade de malignant")
    plt.ylabel("Quantidade")
    plt.legend()
    plt.grid(True)
    save_fig(fig, run_dirs["plots_dir"], "probability_distribution")
    plt.show()
    plt.close(fig)


def _plot_multiclass_outputs(y_true, y_pred, y_prob, run_dirs, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    fig = plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusao")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    save_fig(fig, run_dirs["plots_dir"], "confusion_matrix")
    plt.show()
    plt.close(fig)

    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    fig = plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            auc_score = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
        except ValueError:
            continue
        plt.plot(fpr, tpr, label=f"Classe {i} (AUC={auc_score:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC One-vs-Rest")
    plt.legend()
    save_fig(fig, run_dirs["plots_dir"], "roc_curve")
    plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        plt.plot(recall, precision, label=f"Classe {i}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    save_fig(fig, run_dirs["plots_dir"], "precision_recall")
    plt.show()
    plt.close(fig)


def train_from_dataframes(
    train_df,
    val_df,
    test_df,
    image_loader,
    domain,
    num_classes,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    run_dirs=None,
    seed=SEED,
    device=None,
    balance=True,
    weights_path=None,
    freeze_except_classifier=False,
    pretrained=True,
    model_filename="histodx_torch.pt",
):
    set_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if run_dirs is None:
        run_dirs = make_run_dirs()

    train_loader, val_loader, test_loader = make_loaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        image_loader=image_loader,
        img_size=img_size,
        batch_size=batch_size,
        balance=balance,
        domain=domain,
    )

    loss_weights = compute_class_weights(train_df).to(device) if balance else None
    model = build_histodx_torch(num_classes=num_classes, pretrained=pretrained).to(device)

    if weights_path:
        state = torch.load(weights_path, map_location=device)
        compatible = _filter_compatible_state_dict(model, state)
        model.load_state_dict(compatible, strict=False)

    if freeze_except_classifier:
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False

    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

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

        metrics, _, _, _ = evaluate(model, val_loader, device, num_classes)
        val_metrics.append(metrics)
        print(f"\nEpoch {epoch} Loss: {avg_loss:.4f} | Val Acc: {metrics['accuracy']:.4f}")

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

    metrics, y_true, y_pred, y_prob = evaluate(model, test_loader, device, num_classes)
    save_metrics(metrics, run_dirs["out_dir"], "final_metrics")
    pd.DataFrame(confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))).to_csv(
        os.path.join(run_dirs["out_dir"], "final_confusion_matrix.csv"),
        index=False,
    )
    prob_cols = [f"prob_class_{idx}" for idx in range(y_prob.shape[1])]
    predictions_df = pd.DataFrame(y_prob, columns=prob_cols)
    predictions_df.insert(0, "y_pred", y_pred)
    predictions_df.insert(0, "y_true", y_true)
    predictions_df.to_csv(os.path.join(run_dirs["out_dir"], "final_predictions.csv"), index=False)

    if num_classes == 2:
        _plot_binary_outputs(y_true, y_pred, y_prob, run_dirs)
    else:
        _plot_multiclass_outputs(y_true, y_pred, y_prob, run_dirs, num_classes)

    report = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
    })
    report.to_csv(os.path.join(run_dirs["out_dir"], "predictions.csv"), index=False)

    hist = pd.DataFrame({
        "epoch": list(range(epochs)),
        "train_loss": train_losses,
        "val_accuracy": [m["accuracy"] for m in val_metrics],
    })
    hist.to_csv(os.path.join(run_dirs["out_dir"], "history.csv"), index=False)

    model_path = os.path.join(run_dirs["models_dir"], model_filename)
    torch.save(model.state_dict(), model_path)
    print("Modelo salvo em:", model_path)

    return {
        "run_dirs": run_dirs,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_losses": train_losses,
        "val_metrics": val_metrics,
        "metrics": metrics,
        "model": model,
    }


def run_histodx_breakhis_baseline(
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
    df = collect_breakhis_images(dataset_path, mode="binary")
    train_df, val_df, test_df = split_train_val_test(df, seed=seed)
    return train_from_dataframes(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        image_loader=load_rgb_pil,
        domain="histology",
        num_classes=2,
        img_size=img_size,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        run_dirs=run_dirs,
        seed=seed,
        device=device,
        balance=balance,
        model_filename="histodx_breakhis.pt",
    )


def run_histodx_inbreast_baseline(
    inbreast_csv_path,
    dicom_dir,
    mode="multiclass",
    birads_threshold=4,
    val_fraction=0.2,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    run_dirs=None,
    seed=SEED,
    device=None,
    balance=True,
):
    df = build_inbreast_dataframe(
        inbreast_csv_path=inbreast_csv_path,
        dicom_dir=dicom_dir,
        mode=mode,
        birads_threshold=birads_threshold,
    )
    train_df, holdout_df = split_dataframe_holdout(df, val_fraction=val_fraction * 2, seed=seed)
    val_df, test_df = split_dataframe_holdout(holdout_df, val_fraction=0.5, seed=seed)
    num_classes = int(df["label"].nunique())
    return train_from_dataframes(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        image_loader=load_dicom_pil,
        domain="mammography",
        num_classes=num_classes,
        img_size=img_size,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        run_dirs=run_dirs,
        seed=seed,
        device=device,
        balance=balance,
        model_filename="histodx_inbreast.pt",
    )


def run_histodx_transfer_breakhis_to_inbreast(
    inbreast_csv_path,
    dicom_dir,
    histology_weights_path=None,
    mode="multiclass",
    birads_threshold=4,
    val_fraction=0.2,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    run_dirs=None,
    seed=SEED,
    device=None,
    balance=True,
    freeze_except_classifier=True,
):
    histology_weights_path = _resolve_transfer_weights(
        histology_weights_path,
        checkpoint_name="histodx_breakhis.pt",
    )
    df = build_inbreast_dataframe(
        inbreast_csv_path=inbreast_csv_path,
        dicom_dir=dicom_dir,
        mode=mode,
        birads_threshold=birads_threshold,
    )
    train_df, holdout_df = split_dataframe_holdout(df, val_fraction=val_fraction * 2, seed=seed)
    val_df, test_df = split_dataframe_holdout(holdout_df, val_fraction=0.5, seed=seed)
    num_classes = int(df["label"].nunique())
    return train_from_dataframes(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        image_loader=load_dicom_pil,
        domain="mammography",
        num_classes=num_classes,
        img_size=img_size,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        run_dirs=run_dirs,
        seed=seed,
        device=device,
        balance=balance,
        weights_path=histology_weights_path,
        freeze_except_classifier=freeze_except_classifier,
        pretrained=True,
        model_filename="histodx_breakhis_to_inbreast_tl.pt",
    )


def run_histodx_bracs_baseline(
    dataset_path,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    run_dirs=None,
    seed=SEED,
    device=None,
    balance=True,
    pretrained=False,
):
    train_df, val_df, test_df, class_names = build_bracs_dataframes(dataset_path)
    del class_names
    return train_from_dataframes(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        image_loader=load_rgb_pil,
        domain="histology",
        num_classes=int(train_df["label"].nunique()),
        img_size=img_size,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        run_dirs=run_dirs,
        seed=seed,
        device=device,
        balance=balance,
        pretrained=pretrained,
        model_filename="histodx_bracs.pt",
    )


def run_histodx_bach_baseline(
    dataset_path,
    val_fraction=0.15,
    test_fraction=0.15,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    run_dirs=None,
    seed=SEED,
    device=None,
    balance=True,
    pretrained=False,
):
    train_df, val_df, test_df, class_names = build_bach_dataframes(
        dataset_path,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    del class_names
    return train_from_dataframes(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        image_loader=load_rgb_pil,
        domain="histology",
        num_classes=int(train_df["label"].nunique()),
        img_size=img_size,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        run_dirs=run_dirs,
        seed=seed,
        device=device,
        balance=balance,
        pretrained=pretrained,
        model_filename="histodx_bach.pt",
    )
