import torch
import argparse
import glob
import os
import random
import cv2
import numpy as np
import datetime
import time
import csv
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from model.unet import UNet
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


IMG_SIZE=512; DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark=True; LAMBDA_IMG=10.


def iou(logits, gt, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs > .5).float().mul(gt).sum()
    union = (probs > .5).float().sum() + gt.sum() - inter
    return (inter + eps) / (union + eps)


class SynDataset(Dataset):
    def __init__(self, img_dir, clean_dir, mask_dir, augment=True):
        img_dir, clean_dir, mask_dir = map(Path, (img_dir, clean_dir, mask_dir))
        self.imgs = sorted(img_dir.glob("*.png"))
        self.cleans = {p.stem: clean_dir/p.name for p in self.imgs}
        self.masks = {p.stem: mask_dir/f"{p.stem}.png" for p in self.imgs}
        self.aug = augment
        print(f"{len(self.imgs)} synthetic images | augment={augment}")

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        w = cv2.imread(str(self.imgs[idx]))[:, :, ::-1]
        c = cv2.imread(str(self.cleans[self.imgs[idx].stem]))[:, :, ::-1]
        m = cv2.imread(str(self.masks[self.imgs[idx].stem]), cv2.IMREAD_GRAYSCALE)

        w = cv2.resize(w, (IMG_SIZE, IMG_SIZE))
        c = cv2.resize(c, (IMG_SIZE, IMG_SIZE))
        m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), cv2.INTER_NEAREST)

        if self.aug and random.random() < .5:
            w, c, m = cv2.flip(w, 1), cv2.flip(c, 1), cv2.flip(m, 1)

        w = torch.from_numpy(w.transpose(2, 0, 1)).float() / 255.
        c = torch.from_numpy(c.transpose(2, 0, 1)).float() / 255.
        m = torch.from_numpy(m[None]).float() / 255.
        return w, c, m


def run_epoch(model, loader, optim=None, scaler=None):
    train = optim is not None
    model.train() if train else model.eval()

    BCE = torch.nn.BCEWithLogitsLoss()
    L1 = torch.nn.L1Loss()

    t_l, t_iou, t_psnr, t_ssim, n = 0, 0, 0, 0, 0
    for w, c, m in tqdm(loader, leave=False, desc='train' if train else 'val'):
        w, c, m = w.to(DEVICE), c.to(DEVICE), m.to(DEVICE)

        with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):
            logits, pclean = model(w)
            loss = BCE(logits, m) + LAMBDA_IMG * L1(pclean * m, c * m)

        if train:
            optim.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward(); optim.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            t_l += loss.item()
            t_iou += iou(logits, m).item()
            p = probs[0, 0].cpu().numpy(); g = m[0, 0].cpu().numpy()
            t_psnr += psnr(g, p, data_range=1)
            t_ssim += ssim(g, p, data_range=1)
            n += 1

    m_len = len(loader)
    return t_l / m_len, t_iou / m_len, t_psnr / n, t_ssim / n


def main(opt):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    tr_ds = SynDataset(opt.syn_img, opt.syn_clean, opt.syn_mask, augment=True)
    vl_ds = SynDataset(opt.val_img, opt.val_clean, opt.val_mask, augment=False)

    tr_dl = DataLoader(tr_ds, opt.bs, shuffle=True, num_workers=8, pin_memory=True)
    vl_dl = DataLoader(vl_ds, opt.bs*2, shuffle=False, num_workers=8, pin_memory=True)

    model = UNet(out_clean=True, use_ag=True).to(DEVICE)
    model.load_state_dict(torch.load("models/unet_base.pt", map_location=DEVICE))

    optim  = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'max', factor=.5, patience=3)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    best_iou, wait = 0, 0
    stats = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "train_iou": [], "val_iou": [],
        "val_psnr": [], "val_ssim": []
    }

    for ep in range(1, opt.epochs + 1):
        tr = run_epoch(model, tr_dl, optim, scaler)
        vl = run_epoch(model, vl_dl)

        sched.step(vl[1])

        print(f"[F{ep:02d}/{opt.epochs}] "
              f"L {tr[0]:.3f}/{vl[0]:.3f} | "
              f"IoU {tr[1]:.3f}/{vl[1]:.3f} | "
              f"PSNR {vl[2]:.1f} SSIM {vl[3]:.3f}")

        stats["epoch"].append(ep)
        stats["train_loss"].append(tr[0])
        stats["val_loss"].append(vl[0])
        stats["train_iou"].append(tr[1])
        stats["val_iou"].append(vl[1])
        stats["val_psnr"].append(vl[2])
        stats["val_ssim"].append(vl[3])

        if vl[1] > best_iou:
            best_iou, wait = vl[1], 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/unet_improved.pt")
            print("saved improved model")
        else:
            wait += 1

        if wait >= 8:
            print("Early stop"); break

    os.makedirs("statistics", exist_ok=True)
    csv_path = "statistics/fine_tune_stats.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(stats.keys())
        for i in range(len(stats["epoch"])):
            writer.writerow([stats[k][i] for k in stats.keys()])
    print(f"Saved stats → {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--syn_img", default="dataset/synthetic/wm")
    ap.add_argument("--syn_clean", default="dataset/synthetic/clean")
    ap.add_argument("--syn_mask", default="dataset/synthetic/mask")
    ap.add_argument("--val_img", default="dataset/synthetic_val/wm")
    ap.add_argument("--val_clean", default="dataset/synthetic_val/clean")
    ap.add_argument("--val_mask", default="dataset/synthetic_val/mask")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    main(ap.parse_args())
