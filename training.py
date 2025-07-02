from data import load_data
from model import build_model
import tensorflow as tf
import random

def train_model():
    # Choisir une graine aléatoire à chaque appel
    seed = random.randint(0, 9999)
    print(f"🎲 Utilisation de la graine random_state = {seed}")

    train_ds, test_ds, *_ = load_data(random_state=seed)
    model = build_model(train_ds)

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=20,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True
            )
        ]
    )

    model.save("model.keras")
    return model, test_ds, history
