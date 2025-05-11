import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_stats.csv")

metrics = [
    ("train_loss", "Training Loss"),
    ("val_loss",   "Validation Loss"),
    ("train_iou",  "Training IoU"),
    ("val_iou",    "Validation IoU"),
    ("val_psnr",   "Validation PSNR"),
    ("val_ssim",   "Validation SSIM"),
    ("duration_s", "Epoch Duration (s)")
]

for col, label in metrics:
    plt.figure()
    plt.plot(df["epoch"], df[col])
    plt.title(label)
    plt.xlabel("Epoch")
    plt.ylabel(label)
    plt.show()
