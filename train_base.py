import torch
import argparse
import glob
import os
import random
import time
import numpy as np
import cv2
import datetime
import csv
from tqdm import tqdm
from model.unet import UNet
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


IMG_SIZE = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

def print_gpu():
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU found: {p.name}, {p.total_memory/1e9:.1f} GB")
    else:
        print("No GPU found, using CPU")

def iou(logits, gt, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs>.5).float().mul(gt).sum()
    union = (probs>.5).float().sum() + gt.sum() - inter
    return (inter+eps)/(union+eps)

class SegmDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=False):
        img_dir, mask_dir = Path(img_dir), Path(mask_dir)
        self.imgs  = sorted(img_dir.glob("*.[pj][pn]g"))
        self.masks = {p.stem: mask_dir/f"{p.stem}.png" for p in mask_dir.glob("*.png")}
        self.aug = augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.imgs[idx]))[:,:,::-1]
        mask= cv2.imread(str(self.masks[self.imgs[idx].stem]),0)

        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        mask = cv2.resize(mask,(IMG_SIZE,IMG_SIZE),cv2.INTER_NEAREST)

        if self.aug and random.random()<.5:
            img,mask = cv2.flip(img,1), cv2.flip(mask,1)

        img = torch.from_numpy(img.transpose(2,0,1)).float()/255.
        mask = torch.from_numpy(mask[None]).float()/255.
        return img,mask

def run_epoch(model, loader, optim=None, scaler=None):
    train = optim is not None
    model.train() if train else model.eval()

    criterion = torch.nn.BCEWithLogitsLoss()

    t_l,t_iou,t_psnr,t_ssim,n = 0,0,0,0,0
    for imgs,masks in tqdm(loader,leave=False,desc='train' if train else 'val'):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        with torch.amp.autocast(device_type='cuda',enabled=scaler is not None):
            logits,_ = model(imgs)
            loss = criterion(logits,masks)

        if train:
            optim.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            t_l += loss.item()
            t_iou += iou(logits,masks).item()
            p = probs[0,0].cpu().numpy()
            g = masks[0,0].cpu().numpy()
            t_psnr += psnr(g,p,data_range=1)
            t_ssim += ssim(g,p,data_range=1)
            n += 1

    m = len(loader)
    return t_l/m, t_iou/m, t_psnr/n, t_ssim/n

def main(opt):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    print_gpu()

    stats = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "train_iou": [], "val_iou": [],
        "val_psnr": [], "val_ssim": [],
        "duration_s": []
    }

    tr_ds = SegmDataset(opt.train_img, opt.train_mask, True)
    vl_ds = SegmDataset(opt.val_img, opt.val_mask, False)

    tr_dl = DataLoader(tr_ds, batch_size=opt.bs, shuffle=True, num_workers=8, pin_memory=True)
    vl_dl = DataLoader(vl_ds, batch_size=opt.bs*2, shuffle=False, num_workers=8, pin_memory=True)

    model = UNet(out_clean=False).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,'max',factor=.5,patience=3)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    best_iou, wait = 0, 0
    for ep in range(1, opt.epochs+1):
        start = time.time()
        tr = run_epoch(model, tr_dl, optim, scaler)
        vl = run_epoch(model, vl_dl)

        sched.step(vl[1])
        dur = time.time() - start

        print(f"[{ep:02d}/{opt.epochs}] L {tr[0]:.3f}/{vl[0]:.3f} | "
              f"IoU {tr[1]:.3f}/{vl[1]:.3f} | PSNR {vl[2]:.1f} SSIM {vl[3]:.3f}")
        print(f"    time: {dur:.1f}s")

        stats["epoch"].append(ep)
        stats["train_loss"].append(tr[0])
        stats["val_loss"].append(vl[0])
        stats["train_iou"].append(tr[1])
        stats["val_iou"].append(vl[1])
        stats["val_psnr"].append(vl[2])
        stats["val_ssim"].append(vl[3])
        stats["duration_s"].append(dur)

        if vl[1] > best_iou:
            best_iou, wait = vl[1], 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/unet_base.pt")
            print("saved best")
        else:
            wait += 1
            if wait >= 8:
                print("Early stop")
                break

        torch.cuda.empty_cache()

    with open("statistics/training_stats.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(stats.keys())
        for i in range(len(stats["epoch"])):
            writer.writerow([stats[k][i] for k in stats.keys()])
    print("Saved stats to training_stats.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_img", default="dataset/images/train")
    ap.add_argument("--train_mask", default="dataset/masks/train")
    ap.add_argument("--val_img", default="dataset/images/val")
    ap.add_argument("--val_mask", default="dataset/masks/val")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    main(ap.parse_args())