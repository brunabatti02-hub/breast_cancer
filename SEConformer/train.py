import os
import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt

from .config import DEVICE
from .data import BreastCancerDataset, InbreastDataset
from .eval import evaluate
from .io_utils import make_run_dirs, save_metrics
from .model import SEConformer


def train(csv_path, fold=0, epochs=10, batch_size=16, lr=1e-4, device=DEVICE, run_dirs=None, balance=True):
    train_ds = BreastCancerDataset(csv_path, fold, True)
    val_ds = BreastCancerDataset(csv_path, fold, False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = SEConformer().to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    val_accuracies = []

    if run_dirs is None:
        run_dirs = make_run_dirs()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for imgs, labels in tqdm(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            preds = model(imgs)
            loss = loss_fn(preds, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        print(f"\nEpoch {epoch} Loss: {avg_loss:.4f}")

        metrics = evaluate(
            model,
            val_loader,
            device=device,
            plot=False,
            save_dir=run_dirs["plots_dir"],
            prefix=f"epoch_{epoch}",
            out_dir=run_dirs["out_dir"],
        )
        val_accuracies.append(metrics["accuracy"])

    # Plot training curves (optional)
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.title("Loss por epoca")
    plt.legend()
    fig.savefig(os.path.join(run_dirs["plots_dir"], "train_loss.png"), dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    fig = plt.figure()
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.title("Accuracy por epoca")
    plt.legend()
    fig.savefig(os.path.join(run_dirs["plots_dir"], "val_accuracy.png"), dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    final_metrics = evaluate(
        model,
        val_loader,
        device=device,
        plot=True,
        save_dir=run_dirs["plots_dir"],
        prefix="final",
        out_dir=run_dirs["out_dir"],
    )

    # save history
    hist = pd.DataFrame({
        "epoch": list(range(epochs)),
        "train_loss": train_losses,
        "val_accuracy": val_accuracies,
    })
    hist_path = os.path.join(run_dirs["out_dir"], "history.csv")
    hist.to_csv(hist_path, index=False)
    print("Salvo:", hist_path)

    # save model
    model_path = os.path.join(run_dirs["models_dir"], "seconformer.pt")
    torch.save(model.state_dict(), model_path)
    print("Salvo:", model_path)

    save_metrics(final_metrics, run_dirs["out_dir"], "final_metrics")

    return model, run_dirs


def train_inbreast(csv_path, fold=0, epochs=10, batch_size=16, lr=1e-4, device=DEVICE, run_dirs=None, img_size=224, balance=True):
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

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=loss_weights)

    train_losses = []
    val_accuracies = []

    if run_dirs is None:
        run_dirs = make_run_dirs()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for imgs, labels in tqdm(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            preds = model(imgs)
            loss = loss_fn(preds, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        print(f"\nEpoch {epoch} Loss: {avg_loss:.4f}")

        metrics = evaluate(
            model,
            val_loader,
            device=device,
            plot=False,
            save_dir=run_dirs["plots_dir"],
            prefix=f"epoch_{epoch}",
            out_dir=run_dirs["out_dir"],
        )
        val_accuracies.append(metrics["accuracy"])

    # Plot training curves
    fig = plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.title("Loss por epoca")
    plt.legend()
    fig.savefig(os.path.join(run_dirs["plots_dir"], "train_loss.png"), dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    fig = plt.figure()
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.title("Accuracy por epoca")
    plt.legend()
    fig.savefig(os.path.join(run_dirs["plots_dir"], "val_accuracy.png"), dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    final_metrics = evaluate(
        model,
        val_loader,
        device=device,
        plot=True,
        save_dir=run_dirs["plots_dir"],
        prefix="final",
        out_dir=run_dirs["out_dir"],
    )

    hist = pd.DataFrame({
        "epoch": list(range(epochs)),
        "train_loss": train_losses,
        "val_accuracy": val_accuracies,
    })
    hist_path = os.path.join(run_dirs["out_dir"], "history.csv")
    hist.to_csv(hist_path, index=False)
    print("Salvo:", hist_path)

    model_path = os.path.join(run_dirs["models_dir"], "seconformer.pt")
    torch.save(model.state_dict(), model_path)
    print("Salvo:", model_path)

    save_metrics(final_metrics, run_dirs["out_dir"], "final_metrics")

    return model, run_dirs


def train_inbreast_all(csv_path, epochs=10, batch_size=16, lr=1e-4, device=DEVICE, run_dirs=None, img_size=224, val_split=0.2, seed=42, balance=True):
    import numpy as np

    df = pd.read_csv(csv_path)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    val_size = max(1, int(len(df) * val_split))
    val_df = df.iloc[:val_size].copy()
    train_df = df.iloc[val_size:].copy()

    # create temp csvs for dataset
    train_csv = os.path.join(run_dirs["out_dir"], "train_split.csv") if run_dirs else "train_split.csv"
    val_csv = os.path.join(run_dirs["out_dir"], "val_split.csv") if run_dirs else "val_split.csv"
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

    model = SEConformer().to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=loss_weights)

    train_losses = []
    val_accuracies = []

    if run_dirs is None:
        run_dirs = make_run_dirs()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for imgs, labels in tqdm(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

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
                imgs, labels = imgs.to(device), labels.to(device)
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
    fig.savefig(os.path.join(run_dirs["plots_dir"], "train_loss.png"), dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    fig = plt.figure()
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.title("Accuracy por epoca")
    plt.legend()
    fig.savefig(os.path.join(run_dirs["plots_dir"], "val_accuracy.png"), dpi=200, bbox_inches="tight")
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

    model_path = os.path.join(run_dirs["models_dir"], "seconformer.pt")
    torch.save(model.state_dict(), model_path)
    print("Salvo:", model_path)

    save_metrics({
        "val_accuracy": float(val_accuracies[-1]) if val_accuracies else 0.0,
    }, run_dirs["out_dir"], "final_metrics")

    return model, run_dirs
