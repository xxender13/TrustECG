import wfdb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data/ptb-xl")
LABELS_PATH = Path("data/processed/ptbxl_labels.csv")
OUT_DIR = Path("data/processed/ecg_images")

OUT_DIR.mkdir(parents=True, exist_ok=True)

LEADS = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

def plot_ecg(signal, fs, ecg_id):
    fig, axes = plt.subplots(6, 2, figsize=(10, 12))
    axes = axes.flatten()

    for i, lead in enumerate(LEADS):
        axes[i].plot(signal[:, i], linewidth=0.8)
        axes[i].set_title(lead, fontsize=8)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.suptitle(f"ECG ID: {ecg_id}", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{ecg_id}.png", dpi=150)
    plt.close()

def main():
    df = pd.read_csv(LABELS_PATH)

    for _, row in df.iterrows():
        ecg_id = row["ecg_id"]
        record_path = DATA_DIR / row["filename_lr"]

        try:
            record = wfdb.rdrecord(str(record_path))
            signal = record.p_signal
            fs = record.fs

            if signal.shape[1] != 12:
                continue

            plot_ecg(signal, fs, ecg_id)

        except Exception as e:
            print(f"Skipped {ecg_id}: {e}")

    print("ECG image generation complete.")

if __name__ == "__main__":
    main()
