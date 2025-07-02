import tensorflow as tf

def build_model(train_ds):
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(train_ds.map(lambda x, y: x))

    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(128),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model
