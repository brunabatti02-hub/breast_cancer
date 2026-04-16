import os
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import (
    BASE_PATH, CSV_PATH, IMG_SIZE, BATCH_SIZE, NUM_FOLDS, SEED, MODE,
    PATCH_SIZE, PATCH_STRIDE, TISSUE_THRESHOLD, AUGMENT_TRAIN, AUG_PER_IMAGE,
    USE_GRAYSCALE, APPLY_ADAPTIVE_MEDIAN, ADAPTIVE_MEDIAN_MAX, APPLY_ANISOTROPIC_DIFFUSION,
    DIFFUSION_ITER, DIFFUSION_KAPPA, DIFFUSION_GAMMA, APPLY_HIST_EQ, APPLY_MORPH_OPEN_CLOSE,
    MORPH_KERNEL, APPLY_OTSU_MASK, AUG_ROTATE, AUG_FLIP, AUG_SHEAR, AUG_TRANSLATE,
    AUG_BLUR_SIGMAS, AUG_SHARPNESS
)
from .data import parse_breakhis_dataset, build_folds_csv
from .model import build_squeezenet_feature_extractor
from .features import extract_features
from .optimizer import qbgwo_optimize
from .eval import evaluate_binary, plot_confusion, plot_roc
from .io_utils import make_run_dirs, save_metrics


def _cfg_dict():
    return {
        "IMG_SIZE": IMG_SIZE,
        "PATCH_SIZE": PATCH_SIZE,
        "PATCH_STRIDE": PATCH_STRIDE,
        "TISSUE_THRESHOLD": TISSUE_THRESHOLD,
        "USE_GRAYSCALE": USE_GRAYSCALE,
        "APPLY_ADAPTIVE_MEDIAN": APPLY_ADAPTIVE_MEDIAN,
        "ADAPTIVE_MEDIAN_MAX": ADAPTIVE_MEDIAN_MAX,
        "APPLY_ANISOTROPIC_DIFFUSION": APPLY_ANISOTROPIC_DIFFUSION,
        "DIFFUSION_ITER": DIFFUSION_ITER,
        "DIFFUSION_KAPPA": DIFFUSION_KAPPA,
        "DIFFUSION_GAMMA": DIFFUSION_GAMMA,
        "APPLY_HIST_EQ": APPLY_HIST_EQ,
        "APPLY_MORPH_OPEN_CLOSE": APPLY_MORPH_OPEN_CLOSE,
        "MORPH_KERNEL": MORPH_KERNEL,
        "APPLY_OTSU_MASK": APPLY_OTSU_MASK,
        "AUG_PER_IMAGE": AUG_PER_IMAGE,
        "AUG_ROTATE": AUG_ROTATE,
        "AUG_FLIP": AUG_FLIP,
        "AUG_SHEAR": AUG_SHEAR,
        "AUG_TRANSLATE": AUG_TRANSLATE,
        "AUG_BLUR_SIGMAS": AUG_BLUR_SIGMAS,
        "AUG_SHARPNESS": AUG_SHARPNESS,
    }


def run_squeezenet_svm(
    base_path=BASE_PATH,
    csv_path=CSV_PATH,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    num_folds=NUM_FOLDS,
    seed=SEED,
    mode=MODE,
    run_dirs=None,
):
    if run_dirs is None:
        run_dirs = make_run_dirs()

    df = parse_breakhis_dataset(base_path, mode=mode)
    print("Total imagens:", len(df))
    df = build_folds_csv(df, csv_path, n_splits=num_folds, seed=seed)

    feature_model = build_squeezenet_feature_extractor(pretrained=True)
    cfg = _cfg_dict()

    fold_metrics = []

    for fold in range(num_folds):
        print(f"\n=== FOLD {fold} ===")
        train_df = df[df["fold"] != fold]
        val_df = df[df["fold"] == fold]

        # Cache paths
        train_cache = os.path.join(run_dirs["cache_dir"], f"fold_{fold}_train.npz")
        val_cache = os.path.join(run_dirs["cache_dir"], f"fold_{fold}_val.npz")

        if os.path.exists(train_cache):
            data = np.load(train_cache)
            X_train, y_train = data["X"], data["y"]
        else:
            X_train, y_train = extract_features(
                feature_model,
                train_df,
                cfg,
                augment=AUGMENT_TRAIN,
            )
            np.savez(train_cache, X=X_train, y=y_train)

        if os.path.exists(val_cache):
            data = np.load(val_cache)
            X_val, y_val = data["X"], data["y"]
        else:
            X_val, y_val = extract_features(
                feature_model,
                val_df,
                cfg,
                augment=False,
            )
            np.savez(val_cache, X=X_val, y=y_val)

        # Hyperparameter optimization (Q-BGWO on binary)
        best_C, best_gamma, best_score = qbgwo_optimize(X_train, y_train, cv=3, seed=seed)
        print(f"Melhor C={best_C:.4g}, gamma={best_gamma:.4g}, cv_score={best_score:.4f}")

        clf = make_pipeline(
            StandardScaler(),
            SVC(C=best_C, gamma=best_gamma, kernel="rbf", probability=True)
        )
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_val)[:, 1]
        metrics, y_pred = evaluate_binary(y_val, y_prob)

        print("Metrics:", metrics)
        fold_metrics.append(metrics)

        # Save per-fold plots
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(y_val, y_pred):
            cm[t, p] += 1

        plot_confusion(cm, plots_dir=run_dirs["plots_dir"], show=True)
        plot_roc(y_val, y_prob, plots_dir=run_dirs["plots_dir"], show=True)

    avg_metrics = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0].keys()}
    save_metrics({"folds": fold_metrics, "average": avg_metrics}, run_dirs["out_dir"], "final_metrics")

    return {
        "df": df,
        "fold_metrics": fold_metrics,
        "avg_metrics": avg_metrics,
        "run_dirs": run_dirs,
    }
