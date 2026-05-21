import json
from datetime import datetime
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = PACKAGE_ROOT / "runs"


def make_run_dirs(base_dir=None, run_prefix="hftnet"):
    if base_dir is None:
        base_dir = RUNS_ROOT
    base_dir = Path(base_dir)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = base_dir / f"{run_prefix}_{run_id}"
    plots_dir = out_dir / "plots"
    models_dir = out_dir / "models"
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_id": run_id,
        "out_dir": str(out_dir),
        "plots_dir": str(plots_dir),
        "models_dir": str(models_dir),
    }


def save_fig(fig, plots_dir, name):
    path = Path(plots_dir) / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("Salvo:", path)


def save_metrics(metrics, out_dir, name="metrics"):
    path = Path(out_dir) / f"{name}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print("Salvo:", path)


def find_best_previous_run(checkpoint_name, metric_name="accuracy", runs_root=None):
    runs_root = Path(runs_root) if runs_root else RUNS_ROOT
    best_run = None
    best_metric = float("-inf")

    if not runs_root.exists():
        raise FileNotFoundError(f"Nenhuma pasta de runs encontrada em {runs_root}")

    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue

        metrics_path = run_dir / "final_metrics.json"
        model_path = run_dir / "models" / checkpoint_name
        if not metrics_path.exists() or not model_path.exists():
            continue

        try:
            with metrics_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
        except Exception:
            continue

        metric_value = metrics.get(metric_name)
        if metric_value is None:
            continue

        if float(metric_value) > best_metric:
            best_metric = float(metric_value)
            best_run = {
                "run_dir": str(run_dir),
                "metrics_path": str(metrics_path),
                "model_path": str(model_path),
                "metric_name": metric_name,
                "metric_value": best_metric,
            }

    if best_run is None:
        raise FileNotFoundError(
            f"Nenhum checkpoint '{checkpoint_name}' com métrica '{metric_name}' foi encontrado em {runs_root}"
        )

    return best_run
