import torch
from torch.utils.data import DataLoader
from src.data.ecg_dataset import ECGImageDataset
from src.models.baseline_cnn import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    train_ds = ECGImageDataset(
        "data/processed/ptbxl_labels.csv",
        "data/processed/ecg_images",
        split="train"
    )
    val_ds = ECGImageDataset(
        "data/processed/ptbxl_labels.csv",
        "data/processed/ecg_images",
        split="val"
    )

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    model = build_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Train loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "experiments/baseline_model.pt")
    print("Model saved.")

if __name__ == "__main__":
    train()
