from training import train_model
from evaluate import evaluate_model

model, test_ds, history = train_model()
evaluate_model(model, test_ds)
