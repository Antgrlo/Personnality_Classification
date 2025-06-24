import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_data(csv_path="data/personality_cleaned.csv", batch_size=32):
    df = pd.read_csv(csv_path)
    print(f"✅ {len(df)} lignes chargées depuis {csv_path}")

    X = df.drop("Personality", axis=1).values
    y = df["Personality"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"• Train : {len(X_train)} échantillons, Test : {len(X_test)} échantillons")

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
        .shuffle(buffer_size=len(X_train)) \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)) \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds, X_train, X_test, y_train, y_test
