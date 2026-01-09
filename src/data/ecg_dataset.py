from pathlib import Path
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ECGImageDataset(Dataset):
    def __init__(self, csv_path, image_dir, split="train"):
        self.image_dir = Path(image_dir)

        df = pd.read_csv(csv_path)

        # keep only rows with existing images
        df["img_path"] = df["ecg_id"].apply(
            lambda x: self.image_dir / f"{x}.png"
        )
        df = df[df["img_path"].apply(lambda p: p.exists())]

        # fixed, global split
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(df)

        df["split"] = "train"
        df.loc[int(0.8*n):int(0.9*n), "split"] = "val"
        df.loc[int(0.9*n):, "split"] = "test"

        self.df = df[df["split"] == split].reset_index(drop=True)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["img_path"]).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(row["label"], dtype=torch.long)
        return image, label
