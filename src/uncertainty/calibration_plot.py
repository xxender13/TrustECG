import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

df = pd.read_csv("experiments/test_predictions_calibrated.csv")

prob = df["prob_calibrated"].values
label = df["label"].values

frac_pos, mean_pred = calibration_curve(label, prob, n_bins=10)

plt.figure()
plt.plot(mean_pred, frac_pos, marker="o", label="TrustECG")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration Curve (TrustECG)")
plt.legend()
plt.grid(True)
plt.savefig("experiments/calibration_curve.png", dpi=150)
plt.close()

print("Calibration curve saved.")
