import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

df = pd.read_csv("experiments/test_predictions_calibrated.csv")

y_true = df["label"].values
y_prob = df["prob_calibrated"].values

# ROC
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUROC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("TrustECG ROC Curve")
plt.legend()
plt.grid(True)
plt.savefig("experiments/roc_curve.png", dpi=150)
plt.close()

# Precision–Recall
precision, recall, _ = precision_recall_curve(y_true, y_prob)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("TrustECG Precision–Recall Curve")
plt.grid(True)
plt.savefig("experiments/pr_curve.png", dpi=150)
plt.close()

print("ROC and PR curves saved.")
