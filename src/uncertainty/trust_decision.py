import pandas as pd
import numpy as np

TAU = 0.2  # trust threshold

df = pd.read_csv("experiments/test_predictions_calibrated.csv")

df["trust"] = np.abs(df["prob_calibrated"] - 0.5) >= TAU
df["pred"] = (df["prob_calibrated"] >= 0.5).astype(int)

trusted = df[df["trust"]]

acc = (trusted["pred"] == trusted["label"]).mean()
coverage = len(trusted) / len(df)

print(f"Trusted accuracy: {acc:.4f}")
print(f"Coverage: {coverage:.4f}")
