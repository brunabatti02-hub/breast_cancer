import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from histology_datasets import build_bach_dataframes, build_bracs_dataframes

from .config import DEVICE
from .data import (
    ImageClassificationDataset,
    build_eval_transform,
    build_breakhis_dataframe,
    build_histology_moderate_train_transform,
    build_histology_strong_train_transform,
    build_inbreast_csv,
    load_dicom_pil,
    load_rgb_pil,
    split_dataframe_holdout,
)
from .eval import evaluate
from .io_utils import find_best_previous_run, make_run_dirs, save_metrics
from .model import SEConformer


def _build_train_loader(dataset, labels, batch_size, device, balance=True, epoch_size_multiplier=1.0):
    if not balance:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True), None

    labels = np.asarray(labels, dtype=np.int64)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    num_samples = max(len(sample_weights), int(len(sample_weights) * float(epoch_size_multiplier)))
    sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    loss_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    return loader, loss_weights


def _plot_history(train_losses, val_scores, run_dirs):
    fig = plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.title("Loss por epoca")
    plt.legend()
    fig.savefig(os.path.join(run_dirs["plots_dir"], "train_loss.png"), dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    fig = plt.figure()
    plt.plot(val_scores, label="Val Score")
    plt.title("Score por epoca")
    plt.legend()
    fig.savefig(os.path.join(run_dirs["plots_dir"], "val_score.png"), dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def _save_training_artifacts(model, run_dirs, train_losses, val_scores, final_metrics, model_name="seconformer.pt"):
    hist = pd.DataFrame({
        "epoch": list(range(len(train_losses))),
        "train_loss": train_losses,
        "val_score": val_scores,
    })
    hist_path = os.path.join(run_dirs["out_dir"], "history.csv")
    hist.to_csv(hist_path, index=False)
    print("Salvo:", hist_path)

    model_path = os.path.join(run_dirs["models_dir"], model_name)
    torch.save(model.state_dict(), model_path)
    print("Salvo:", model_path)

    save_metrics(final_metrics, run_dirs["out_dir"], "final_metrics")


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


def _load_compatible_weights(model, weights_path, device):
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model_state = model.state_dict()
    compatible_state = {}
    skipped_keys = []

    for key, value in state.items():
        if key not in model_state:
            skipped_keys.append((key, "missing_in_model"))
            continue

        if model_state[key].shape != value.shape:
            skipped_keys.append((key, f"shape_mismatch checkpoint={tuple(value.shape)} model={tuple(model_state[key].shape)}"))
            continue

        compatible_state[key] = value

    missing_keys, unexpected_keys = model.load_state_dict(compatible_state, strict=False)

    print(f"Pesos carregados de: {weights_path}")
    print(f"Camadas reaproveitadas: {len(compatible_state)}")

    if skipped_keys:
        print("Camadas ignoradas na transferencia:")
        for key, reason in skipped_keys:
            print(f"  - {key}: {reason}")

    if missing_keys:
        print("Camadas inicializadas do zero:", ", ".join(missing_keys))

    if unexpected_keys:
        print("Chaves inesperadas no checkpoint:", ", ".join(unexpected_keys))


def train_from_dataframes(
    train_df,
    val_df,
    image_loader,
    num_classes,
    test_df=None,
    epochs=10,
    batch_size=16,
    lr=1e-4,
    device=DEVICE,
    run_dirs=None,
    img_size=224,
    balance=True,
    weights_path=None,
    freeze_backbone=False,
    model_name="seconformer.pt",
    train_transform=None,
    eval_transform=None,
    epoch_size_multiplier=1.0,
):
    if run_dirs is None:
        run_dirs = make_run_dirs()

    train_ds = ImageClassificationDataset(
        train_df,
        image_loader=image_loader,
        train=True,
        img_size=img_size,
        transforms_override=train_transform,
    )
    val_ds = ImageClassificationDataset(
        val_df,
        image_loader=image_loader,
        train=False,
        img_size=img_size,
        transforms_override=eval_transform,
    )
    test_ds = None
    if test_df is not None:
        test_ds = ImageClassificationDataset(
            test_df,
            image_loader=image_loader,
            train=False,
            img_size=img_size,
            transforms_override=eval_transform,
        )

    train_loader, loss_weights = _build_train_loader(
        train_ds,
        train_df["label"].values,
        batch_size=batch_size,
        device=device,
        balance=balance,
        epoch_size_multiplier=epoch_size_multiplier,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size) if test_ds is not None else None

    model = SEConformer(num_classes=num_classes).to(device)
    if weights_path:
        _load_compatible_weights(model, weights_path, device)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=loss_weights)

    train_losses = []
    val_scores = []
    best_score = float("-inf")
    best_state = None
    score_key = "f1_macro" if num_classes > 2 else "accuracy"

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for imgs, labels in tqdm(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            preds = model(imgs)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"\nEpoch {epoch} Loss: {avg_loss:.4f}")

        metrics = evaluate(
            model,
            val_loader,
            device=device,
            num_classes=num_classes,
            plot=False,
            save_dir=run_dirs["plots_dir"],
            prefix=f"epoch_{epoch}",
            out_dir=run_dirs["out_dir"],
        )

        val_scores.append(metrics[score_key])

        if metrics[score_key] > best_score:
            best_score = metrics[score_key]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Melhor modelo restaurado com {score_key}={best_score:.4f}")

    final_loader = test_loader if test_loader is not None else val_loader
    final_metrics = evaluate(
        model,
        final_loader,
        device=device,
        num_classes=num_classes,
        plot=True,
        save_dir=run_dirs["plots_dir"],
        prefix="final",
        out_dir=run_dirs["out_dir"],
    )

    _plot_history(train_losses, val_scores, run_dirs)
    _save_training_artifacts(model, run_dirs, train_losses, val_scores, final_metrics, model_name=model_name)
    return model, run_dirs, final_metrics


def train_breakhis_fold(
    csv_path,
    fold=0,
    epochs=10,
    batch_size=16,
    lr=1e-4,
    device=DEVICE,
    run_dirs=None,
    img_size=224,
    balance=True,
    weights_path=None,
    freeze_backbone=False,
):
    df = pd.read_csv(csv_path)
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)
    num_classes = int(df["label"].nunique())
    return train_from_dataframes(
        train_df=train_df,
        val_df=val_df,
        test_df=None,
        image_loader=load_rgb_pil,
        num_classes=num_classes,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        run_dirs=run_dirs,
        img_size=img_size,
        balance=balance,
        weights_path=weights_path,
        freeze_backbone=freeze_backbone,
        model_name="seconformer_breakhis.pt",
    )


def train_breakhis_holdout(
    base_path,
    mode="binary",
    val_fraction=0.2,
    seed=42,
    epochs=10,
    batch_size=16,
    lr=1e-4,
    device=DEVICE,
    run_dirs=None,
    img_size=224,
    balance=True,
):
    df = build_breakhis_dataframe(base_path, mode=mode)
    train_df, val_df = split_dataframe_holdout(df, val_fraction=val_fraction, seed=seed)
    num_classes = int(df["label"].nunique())
    return train_from_dataframes(
        train_df=train_df,
        val_df=val_df,
        test_df=None,
        image_loader=load_rgb_pil,
        num_classes=num_classes,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        run_dirs=run_dirs,
        img_size=img_size,
        balance=balance,
        model_name="seconformer_breakhis.pt",
    )


def train_inbreast_fold(
    csv_path,
    fold=0,
    epochs=10,
    batch_size=16,
    lr=1e-4,
    device=DEVICE,
    run_dirs=None,
    img_size=224,
    balance=True,
    weights_path=None,
    freeze_backbone=False,
):
    df = pd.read_csv(csv_path)
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)
    num_classes = int(df["label"].nunique())
    return train_from_dataframes(
        train_df=train_df,
        val_df=val_df,
        test_df=None,
        image_loader=load_dicom_pil,
        num_classes=num_classes,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        run_dirs=run_dirs,
        img_size=img_size,
        balance=balance,
        weights_path=weights_path,
        freeze_backbone=freeze_backbone,
        model_name="seconformer_inbreast.pt",
    )


def train_inbreast_holdout(
    csv_path,
    val_fraction=0.2,
    seed=42,
    epochs=10,
    batch_size=16,
    lr=1e-4,
    device=DEVICE,
    run_dirs=None,
    img_size=224,
    balance=True,
    weights_path=None,
    freeze_backbone=False,
):
    df = pd.read_csv(csv_path)
    train_df, val_df = split_dataframe_holdout(df, val_fraction=val_fraction, seed=seed)
    num_classes = int(df["label"].nunique())
    return train_from_dataframes(
        train_df=train_df,
        val_df=val_df,
        test_df=None,
        image_loader=load_dicom_pil,
        num_classes=num_classes,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        run_dirs=run_dirs,
        img_size=img_size,
        balance=balance,
        weights_path=weights_path,
        freeze_backbone=freeze_backbone,
        model_name="seconformer_inbreast.pt",
    )


def train_transfer_breakhis_to_inbreast(
    inbreast_csv_path,
    histology_weights_path=None,
    val_fraction=0.2,
    seed=42,
    epochs=10,
    batch_size=16,
    lr=1e-4,
    device=DEVICE,
    run_dirs=None,
    img_size=224,
    balance=True,
    freeze_backbone=True,
):
    histology_weights_path = _resolve_transfer_weights(
        histology_weights_path,
        checkpoint_name="seconformer_breakhis.pt",
    )
    return train_inbreast_holdout(
        csv_path=inbreast_csv_path,
        val_fraction=val_fraction,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        run_dirs=run_dirs,
        img_size=img_size,
        balance=balance,
        weights_path=histology_weights_path,
        freeze_backbone=freeze_backbone,
    )


def build_inbreast_baseline_csv(*args, **kwargs):
    return build_inbreast_csv(*args, **kwargs)


def train_bracs_baseline(
    dataset_path,
    epochs=10,
    batch_size=16,
    lr=1e-4,
    device=DEVICE,
    run_dirs=None,
    img_size=224,
    balance=True,
    use_moderate_augmentation=True,
    epoch_size_multiplier=1.25,
):
    train_df, val_df, test_df, class_names = build_bracs_dataframes(dataset_path)
    del class_names
    train_transform = build_histology_moderate_train_transform(img_size) if use_moderate_augmentation else None
    eval_transform = build_eval_transform(img_size)
    return train_from_dataframes(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        image_loader=load_rgb_pil,
        num_classes=int(train_df["label"].nunique()),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        run_dirs=run_dirs,
        img_size=img_size,
        balance=balance,
        model_name="seconformer_bracs.pt",
        train_transform=train_transform,
        eval_transform=eval_transform,
        epoch_size_multiplier=epoch_size_multiplier,
    )


def train_bach_baseline(
    dataset_path,
    val_fraction=0.15,
    test_fraction=0.15,
    seed=42,
    epochs=10,
    batch_size=16,
    lr=1e-4,
    device=DEVICE,
    run_dirs=None,
    img_size=224,
    balance=True,
    use_strong_augmentation=True,
    epoch_size_multiplier=2.0,
):
    train_df, val_df, test_df, class_names = build_bach_dataframes(
        dataset_path,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    del class_names
    train_transform = build_histology_strong_train_transform(img_size) if use_strong_augmentation else None
    eval_transform = build_eval_transform(img_size)
    return train_from_dataframes(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        image_loader=load_rgb_pil,
        num_classes=int(train_df["label"].nunique()),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        run_dirs=run_dirs,
        img_size=img_size,
        balance=balance,
        model_name="seconformer_bach.pt",
        train_transform=train_transform,
        eval_transform=eval_transform,
        epoch_size_multiplier=epoch_size_multiplier,
    )
