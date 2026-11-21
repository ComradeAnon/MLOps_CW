import sys

import joblib
import pandas as pd

model = joblib.load("model.joblib")
clients = pd.read_csv(sys.stdin, na_values=" ")

predictions = pd.DataFrame({
    "churn": model.predict(clients),
    "churn_probability": model.predict_proba(clients)[:, 1],
})
predictions.to_csv(sys.stdout, index=False)
