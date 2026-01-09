import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("experiments/test_predictions_calibrated.csv")

df["pred"] = (df["prob_calibrated"] >= 0.5).astype(int)
df["error"] = df["pred"] != df["label"]

plt.figure()
plt.hist(df[df["error"]]["prob_calibrated"], bins=20)
plt.xlabel("Predicted probability")
plt.ylabel("Count")
plt.title("Distribution of Misclassified Cases")
plt.grid(True)
plt.savefig("experiments/error_confidence_distribution.png", dpi=150)
plt.close()

print("Failure analysis plot saved.")
