from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, test_ds):
    loss, accuracy = model.evaluate(test_ds)
    print(f"🎯 Précision sur le test : {accuracy:.2%}")

    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred = (model.predict(test_ds) > 0.5).astype(int).flatten()

    print("📊 Matrice de confusion :")
    print(confusion_matrix(y_true, y_pred))
    print("\n📋 Rapport de classification :")
    print(classification_report(y_true, y_pred))
