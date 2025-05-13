import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot(ax, x, y, title):
    ax.plot(x, y, marker='o')
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(title)

def main(csv_path):
    df = pd.read_csv(csv_path)
    print("\n=== Описательная статистика ===")
    print(df.describe().T)

    metrics = [
        ("train_loss", "Train Loss"),
        ("val_loss", "Val Loss"),
        ("train_iou", "Train IoU"),
        ("val_iou", "Val IoU"),
        ("val_psnr", "Val PSNR"),
        ("val_ssim", "Val SSIM"),
        ("lr", "Learning Rate"),
        ("time_s", "Epoch Time (s)")
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for (key, title), ax in zip(metrics, axes.flatten()):
        plot(ax, df["epoch"], df[key], title)
    plt.tight_layout()
    plt.show()

    corr = df[[k for k,_ in metrics]].corr()
    print("\n=== Корреляция метрик ===")
    print(corr)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="Path to training stats CSV")
    args = p.parse_args()
    main(args.csv)
