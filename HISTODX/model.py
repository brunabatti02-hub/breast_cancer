import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2B3


def build_histodx_breakhis(img_size=224, learning_rate=1e-4):
    inputs = layers.Input(shape=(img_size, img_size, 3), name="input_image")

    base_model = EfficientNetV2B3(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )
    base_model.trainable = False

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(base_model.output)
    x = layers.Dense(1024, activation="relu", name="dense_1024")(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Dense(512, activation="relu", name="dense_512")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs, outputs, name="HistoDX_BreakHis")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model, base_model
