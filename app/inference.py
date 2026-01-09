import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from src.models.baseline_cnn import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# trust threshold
TAU = 0.2

# image preprocessing (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

model = build_model().to(DEVICE)
model.load_state_dict(
    torch.load("experiments/baseline_model.pt", map_location=DEVICE)
)
model.eval()

def predict_ecg(image: Image.Image):
    x = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)[0, 1].item()

    trust = abs(prob - 0.5) >= TAU
    pred = "Abnormal" if prob >= 0.5 else "Normal"

    return {
        "prediction": pred,
        "probability": round(prob, 3),
        "trust": "Safe to trust" if trust else "Uncertain – human review needed"
    }
