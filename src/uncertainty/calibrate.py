import pandas as pd
from netcal.scaling import TemperatureScaling

df = pd.read_csv("experiments/test_predictions.csv")

calibrator = TemperatureScaling()
calibrator.fit(df["prob"].values, df["label"].values)

df["prob_calibrated"] = calibrator.transform(df["prob"].values)
df.to_csv("experiments/test_predictions_calibrated.csv", index=False)

print("Calibration complete.")
