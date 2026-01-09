import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("experiments/test_predictions_calibrated.csv")

taus = np.linspace(0.0, 0.45, 25)
coverages = []
accuracies = []

for tau in taus:
    trusted = df[np.abs(df["prob_calibrated"] - 0.5) >= tau]

    if len(trusted) == 0:
        continue

    coverage = len(trusted) / len(df)
    acc = (trusted["label"] == (trusted["prob_calibrated"] >= 0.5)).mean()

    coverages.append(coverage)
    accuracies.append(acc)

plt.figure()
plt.plot(coverages, accuracies, marker="o")
plt.xlabel("Coverage")
plt.ylabel("Accuracy on Trusted Predictions")
plt.title("Accuracy–Coverage Tradeoff (TrustECG)")
plt.grid(True)
plt.savefig("experiments/accuracy_coverage_curve.png", dpi=150)
plt.close()

print("Accuracy–coverage curve saved.")
