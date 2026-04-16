import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def collect_breakhis_images(dataset_path):
    records = []

    for root, _dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(VALID_EXTENSIONS):
                file_path = os.path.join(root, file)
                lower_path = file_path.lower()

                if "benign" in lower_path:
                    label = 0
                    class_name = "benign"
                elif "malignant" in lower_path:
                    label = 1
                    class_name = "malignant"
                else:
                    continue

                magnification = None
                for mag in ["40x", "100x", "200x", "400x"]:
                    if mag in lower_path:
                        magnification = mag
                        break

                records.append({
                    "path": file_path,
                    "label": label,
                    "class_name": class_name,
                    "magnification": magnification if magnification else "unknown",
                })

    df = pd.DataFrame(records)
    return df


def split_train_val_test(df, seed=42, test_size=0.30):
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=seed,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=seed,
    )
    return train_df, val_df, test_df


def make_datasets(train_df, val_df, test_df, img_size=224, batch_size=32, seed=42):
    autotune = tf.data.AUTOTUNE

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, (img_size, img_size))
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ], name="data_augmentation")

    def make_dataset(dataframe, training=False):
        paths = dataframe["path"].values
        labels = dataframe["label"].values.astype(np.int32)

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(load_image, num_parallel_calls=autotune)

        if training:
            ds = ds.shuffle(buffer_size=len(dataframe), seed=seed)

            def augment(image, label):
                image = data_augmentation(image, training=True)
                return image, label

            ds = ds.map(augment, num_parallel_calls=autotune)

        ds = ds.batch(batch_size).prefetch(autotune)
        return ds

    train_ds = make_dataset(train_df, training=True)
    val_ds = make_dataset(val_df, training=False)
    test_ds = make_dataset(test_df, training=False)
    return train_ds, val_ds, test_ds


def compute_class_weights(train_df):
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_df["label"].values,
    )
    return {
        0: float(class_weights_array[0]),
        1: float(class_weights_array[1]),
    }
