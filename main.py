from training import train_model
from evaluate import evaluate_model
from explain import explain_representative_profiles

model, test_ds, history = train_model()
evaluate_model(model, test_ds)
explain_representative_profiles()