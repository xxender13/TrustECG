import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from src.data.ecg_dataset import ECGImageDataset
from src.models.baseline_cnn import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate():
    test_ds = ECGImageDataset(
        "data/processed/ptbxl_labels.csv",
        "data/processed/ecg_images",
        split="test"
    )
    loader = DataLoader(test_ds, batch_size=16)

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load("experiments/baseline_model.pt", map_location=DEVICE))
    model.eval()

    probs, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            p = torch.softmax(out, dim=1)[:, 1]
            probs.extend(p.cpu().numpy())
            labels.extend(y.numpy())

    auc = roc_auc_score(labels, probs)
    print(f"Test AUROC: {auc:.4f}")

    from pathlib import Path

    out_dir = Path("experiments")
    out_dir.mkdir(exist_ok=True)

    pd.DataFrame(
        {"prob": probs, "label": labels}
    ).to_csv(out_dir / "test_predictions.csv", index=False)

    print("Saved predictions to experiments/test_predictions.csv")


if __name__ == "__main__":
    evaluate()
