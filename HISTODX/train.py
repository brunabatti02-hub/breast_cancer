import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from .config import SEED, IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE
from .data import collect_breakhis_images, split_train_val_test, make_datasets, compute_class_weights
from .model import build_histodx_breakhis
from .eval import plot_training_history, evaluate_model
from .io_utils import make_run_dirs


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def run_histodx_breakhis(
    dataset_path,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    fine_tune_epochs=10,
    run_dirs=None,
    seed=SEED,
):
    set_seed(seed)

    if run_dirs is None:
        run_dirs = make_run_dirs()

    df = collect_breakhis_images(dataset_path)
    train_df, val_df, test_df = split_train_val_test(df, seed=seed)

    train_ds, val_ds, test_ds = make_datasets(
        train_df, val_df, test_df, img_size=img_size, batch_size=batch_size, seed=seed
    )

    class_weights = compute_class_weights(train_df)

    model, base_model = build_histodx_breakhis(img_size=img_size, learning_rate=learning_rate)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            os.path.join(run_dirs["models_dir"], "best_histodx_breakhis.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    base_model.trainable = True
    fine_tune_at = int(len(base_model.layers) * 0.7)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    history_2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=fine_tune_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # merge histories
    history = {}
    for key in history_1.history.keys():
        history[key] = history_1.history[key] + history_2.history[key]

    plot_training_history(history, plots_dir=run_dirs["plots_dir"], show=True)

    eval_results = evaluate_model(
        model,
        test_ds,
        test_df,
        plots_dir=run_dirs["plots_dir"],
        out_dir=run_dirs["out_dir"],
        show=True,
    )

    model_path = os.path.join(run_dirs["models_dir"], "histodx_breakhis_final.keras")
    model.save(model_path)
    print("Modelo salvo em:", model_path)

    return {
        "run_dirs": run_dirs,
        "df": df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "history": history,
        "eval": eval_results,
        "model": model,
        "base_model": base_model,
    }
