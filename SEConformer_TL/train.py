import os
import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt

from .config import DEVICE, IMG_SIZE, BATCH_SIZE, EPOCHS, LR
from .data import InbreastDataset
from .model import SEConformer
from .io_utils import make_run_dirs, save_metrics, save_fig


def train_inbreast_transfer(
    csv_path,
    weights_path,
    fold=0,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    device=DEVICE,
    run_dirs=None,
    img_size=IMG_SIZE,
    freeze_backbone=True,
    balance=True,
):
    train_ds = InbreastDataset(csv_path, fold, True, img_size=img_size)
    val_ds = InbreastDataset(csv_path, fold, False, img_size=img_size)

    if balance:
        y = train_ds.df["label"].values
        class_counts = np.bincount(y)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[y]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
        loss_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        loss_weights = None
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = SEConformer().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=False)

    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("classifier"):
                p.requires_grad = False

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=loss_weights)

    train_losses = []
    val_accuracies = []

    if run_dirs is None:
        run_dirs = make_run_dirs()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for imgs, labels in tqdm(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            preds = model(imgs)
            loss = loss_fn(preds, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                preds = model(imgs)
                pred_labels = preds.argmax(dim=1)
                correct += (pred_labels == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / max(1, total)
        val_accuracies.append(val_acc)

        print(f"\nEpoch {epoch} Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    # plots
    fig = plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.title("Loss por epoca")
    plt.legend()
    save_fig(fig, run_dirs["plots_dir"], "train_loss")
    plt.show()
    plt.close(fig)

    fig = plt.figure()
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.title("Accuracy por epoca")
    plt.legend()
    save_fig(fig, run_dirs["plots_dir"], "val_accuracy")
    plt.show()
    plt.close(fig)

    hist = pd.DataFrame({
        "epoch": list(range(epochs)),
        "train_loss": train_losses,
        "val_accuracy": val_accuracies,
    })
    hist_path = os.path.join(run_dirs["out_dir"], "history.csv")
    hist.to_csv(hist_path, index=False)
    print("Salvo:", hist_path)

    model_path = os.path.join(run_dirs["models_dir"], "seconformer_tl.pt")
    torch.save(model.state_dict(), model_path)
    print("Salvo:", model_path)

    save_metrics({
        "val_accuracy": float(val_accuracies[-1]) if val_accuracies else 0.0,
    }, run_dirs["out_dir"], "final_metrics")

    return model, run_dirs


def train_inbreast_transfer_all(
    csv_path,
    weights_path,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    device=DEVICE,
    run_dirs=None,
    img_size=IMG_SIZE,
    freeze_backbone=True,
    val_split=0.2,
    seed=42,
    balance=True,
):
    import numpy as np
    import pandas as pd

    if run_dirs is None:
        run_dirs = make_run_dirs()

    df = pd.read_csv(csv_path)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    val_size = max(1, int(len(df) * val_split))
    val_df = df.iloc[:val_size].copy()
    train_df = df.iloc[val_size:].copy()

    train_csv = os.path.join(run_dirs["out_dir"], "train_split.csv")
    val_csv = os.path.join(run_dirs["out_dir"], "val_split.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    train_ds = InbreastDataset(train_csv, fold=0, train=True, img_size=img_size)
    val_ds = InbreastDataset(val_csv, fold=0, train=False, img_size=img_size)

    if balance:
        y = train_df["label"].values
        class_counts = np.bincount(y)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[y]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
        loss_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        loss_weights = None
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    num_classes = int(df["label"].nunique())
    model = SEConformer(num_classes=num_classes).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=False)

    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("classifier"):
                p.requires_grad = False

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=loss_weights)

    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for imgs, labels in tqdm(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            preds = model(imgs)
            loss = loss_fn(preds, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                preds = model(imgs)
                pred_labels = preds.argmax(dim=1)
                correct += (pred_labels == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / max(1, total)
        val_accuracies.append(val_acc)

        print(f"\nEpoch {epoch} Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    fig = plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.title("Loss por epoca")
    plt.legend()
    save_fig(fig, run_dirs["plots_dir"], "train_loss")
    plt.show()
    plt.close(fig)

    fig = plt.figure()
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.title("Accuracy por epoca")
    plt.legend()
    save_fig(fig, run_dirs["plots_dir"], "val_accuracy")
    plt.show()
    plt.close(fig)

    hist = pd.DataFrame({
        "epoch": list(range(epochs)),
        "train_loss": train_losses,
        "val_accuracy": val_accuracies,
    })
    hist_path = os.path.join(run_dirs["out_dir"], "history.csv")
    hist.to_csv(hist_path, index=False)
    print("Salvo:", hist_path)

    model_path = os.path.join(run_dirs["models_dir"], "seconformer_tl.pt")
    torch.save(model.state_dict(), model_path)
    print("Salvo:", model_path)

    save_metrics({
        "val_accuracy": float(val_accuracies[-1]) if val_accuracies else 0.0,
    }, run_dirs["out_dir"], "final_metrics")

    return model, run_dirs
