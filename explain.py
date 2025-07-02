import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def explain_representative_profiles(csv_path="data/personality_cleaned.csv", model_path="model.keras", show_plot=True):
    df = pd.read_csv(csv_path)
    X = df.drop("Personality", axis=1).values
    feature_names = df.drop("Personality", axis=1).columns
    y = df["Personality"].values

    model = load_model(model_path)
    probas = model.predict(X).flatten()

    idx_intro = np.argmin(probas)
    idx_extro = np.argmax(probas)

    intro_profile = df.iloc[idx_intro].drop("Personality")
    extro_profile = df.iloc[idx_extro].drop("Personality")

    print("💡 Profil typique introverti (score = {:.3f})".format(probas[idx_intro]))
    print(intro_profile)

    print("\n💡 Profil typique extraverti (score = {:.3f})".format(probas[idx_extro]))
    print(extro_profile)

    if show_plot:
        labels = feature_names
        intro_values = intro_profile.values
        extro_values = extro_profile.values

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        intro_values = np.concatenate((intro_values, [intro_values[0]]))
        extro_values = np.concatenate((extro_values, [extro_values[0]]))
        angles += angles[:1]

        plt.figure(figsize=(8, 6))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, intro_values, label="Introverti", linestyle='-', marker='o')
        ax.plot(angles, extro_values, label="Extraverti", linestyle='-', marker='o')
        ax.fill(angles, intro_values, alpha=0.25)
        ax.fill(angles, extro_values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_title("Comparaison des profils types")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()
