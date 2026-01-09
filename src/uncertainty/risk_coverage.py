import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("experiments/test_predictions_calibrated.csv")

taus = np.linspace(0.0, 0.45, 25)
coverages = []
risks = []

for tau in taus:
    trusted = df[np.abs(df["prob_calibrated"] - 0.5) >= tau]

    if len(trusted) == 0:
        continue

    coverage = len(trusted) / len(df)
    error = 1 - (trusted["label"] == (trusted["prob_calibrated"] >= 0.5)).mean()

    coverages.append(coverage)
    risks.append(error)

plt.figure()
plt.plot(coverages, risks, marker="o")
plt.xlabel("Coverage (fraction of predictions made)")
plt.ylabel("Risk (error rate on trusted predictions)")
plt.title("TrustECG Risk–Coverage Curve")
plt.grid(True)

plt.savefig("experiments/risk_coverage_curve.png", dpi=150)
plt.show()

print("Risk–coverage curve saved.")
