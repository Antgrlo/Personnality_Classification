# classification.py

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 1. Chargement des données
csv_path = "data/personality_cleaned.csv"
df = pd.read_csv(csv_path)
print(f"✅ {len(df)} lignes chargées depuis {csv_path}")

# 2. Séparation features / cible
X = df.drop("Personality", axis=1).values   # toutes les colonnes sauf la cible
y = df["Personality"].values                # la colonne binaire (0/1)

# 3. Train/Test split (80 % / 20 %), avec stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"• Train : {len(X_train)} échantillons, Test : {len(X_test)} échantillons")

# 4. Construction des tf.data.Dataset
batch_size = 32
train_ds = (
    tf.data.Dataset
      .from_tensor_slices((X_train, y_train))       # création du Dataset depuis tableaux :contentReference[oaicite:0]{index=0}
      .shuffle(buffer_size=len(X_train))            # mélange
      .batch(batch_size)                            # découpage en batchs
      .prefetch(tf.data.AUTOTUNE)                   # préchargement asynchrone
)
test_ds = (
    tf.data.Dataset
      .from_tensor_slices((X_test, y_test))
      .batch(batch_size)
      .prefetch(tf.data.AUTOTUNE)
)

# 5. Normalisation automatique des features
normalizer = tf.keras.layers.Normalization(axis=-1)  # couche de preprocessing :contentReference[oaicite:1]{index=1}
normalizer.adapt(train_ds.map(lambda x, y: x))        # calcul de la moyenne et de la variance

# 6. Définition du modèle
model = tf.keras.Sequential([
    normalizer,                                       # intègre le prétraitement dans le modèle
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")    # sortie binaire
])

# 7. Compilation
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 8. Entraînement
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

# 9. Évaluation
loss, accuracy = model.evaluate(test_ds)
print(f"🎯 Précision sur le test : {accuracy:.2%}")