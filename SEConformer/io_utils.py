import json
import os
from datetime import datetime


def make_run_dirs(base_dir="results/SEConformer", run_prefix="seconformer"):
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(base_dir, f"{run_prefix}_{run_id}")
    plots_dir = os.path.join(out_dir, "plots")
    models_dir = os.path.join(out_dir, "models")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    return {
        "run_id": run_id,
        "out_dir": out_dir,
        "plots_dir": plots_dir,
        "models_dir": models_dir,
    }


def save_fig(fig, plots_dir, name):
    path = os.path.join(plots_dir, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("Salvo:", path)


def save_metrics(metrics, out_dir, name="metrics"):
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("Salvo:", path)
