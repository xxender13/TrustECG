import pandas as pd
import numpy as np

df = pd.read_csv("experiments/test_predictions_calibrated.csv")

for tau in [0.05, 0.1, 0.2, 0.3]:
    trusted = df[np.abs(df["prob_calibrated"] - 0.5) >= tau]
    coverage = len(trusted) / len(df)
    acc = (trusted["label"] == (trusted["prob_calibrated"] >= 0.5)).mean()

    print(f"tau={tau:.2f} | coverage={coverage:.2f} | trusted accuracy={acc:.3f}")
