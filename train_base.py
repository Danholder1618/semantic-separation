import torch
import argparse
import os
import random
import time
import numpy as np
import cv2
import datetime
import csv
import json
from model.unet import UNet
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
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
    inter = (probs > .5).float().mul(gt).sum()
    union = (probs > .5).float().sum() + gt.sum() - inter
    return (inter + eps) / (union + eps)

class SegmDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=False):
        img_dir, mask_dir = Path(img_dir), Path(mask_dir)
        self.imgs = sorted(img_dir.glob("*.[pj][pn]g"))
        self.masks = {p.stem: mask_dir / f"{p.stem}.png" for p in mask_dir.glob("*.png")}
        self.aug = augment
        print(f"Loaded {len(self.imgs)} images (augment={augment})")

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.imgs[idx]))[:, :, ::-1]
        mask = cv2.imread(str(self.masks[self.imgs[idx].stem]), 0)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), cv2.INTER_NEAREST)
        if self.aug and random.random() < .5:
            img, mask = cv2.flip(img, 1), cv2.flip(mask, 1)
        img = torch.from_numpy(img.transpose(2,0,1)).float() / 255.
        mask = torch.from_numpy(mask[None]).float() / 255.
        return img, mask

def run_epoch(model, loader, optim=None, scaler=None):
    train = optim is not None
    if train:
        model.train()
    else:
        model.eval()

    criterion = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_iou  = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count      = 0

    if not train:
        with torch.no_grad():
            for imgs, masks in tqdm(loader, desc="val ", leave=False):
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                with torch.amp.autocast(device_type='cuda', enabled=bool(scaler)):
                    logits, _ = model(imgs)
                    loss = criterion(logits, masks)
                probs = torch.sigmoid(logits)

                batch_iou = iou(logits, masks).item()
                total_loss += loss.item()
                total_iou  += batch_iou

                p_np = probs[:,0].cpu().numpy()
                g_np = masks[:,0].cpu().numpy()
                for p_img, g_img in zip(p_np, g_np):
                    total_psnr += psnr(g_img, p_img, data_range=1)
                    total_ssim += ssim(g_img, p_img, data_range=1)
                    count += 1
    else:
        for imgs, masks in tqdm(loader, desc="train", leave=False):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            with torch.amp.autocast(device_type='cuda', enabled=bool(scaler)):
                logits, _ = model(imgs)
                loss = criterion(logits, masks)

            optim.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward(); optim.step()

            batch_iou = iou(logits, masks).item()
            total_loss += loss.item()
            total_iou  += batch_iou

    batches = len(loader)
    avg_loss = total_loss / batches
    avg_iou  = total_iou  / batches
    if not train:
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
    else:
        avg_psnr = avg_ssim = 0.0

    return avg_loss, avg_iou, avg_psnr, avg_ssim

def save_checkpoint(model, optim, scaler, epoch, stats, fname):
    ck = {'epoch': epoch,
          'model_state': model.state_dict(),
          'optim_state': optim.state_dict(),
          'stats': stats}
    if scaler: ck['scaler'] = scaler.state_dict()
    torch.save(ck, fname)
    print(f"Model Saved to {fname}")

def load_checkpoint(model, optim, scaler, fname):
    if not os.path.exists(fname):
        print(f"Model {fname} not found → from scratch")
        return 0, {}
    ck = torch.load(fname, map_location=DEVICE)
    model.load_state_dict(ck['model_state'])
    optim.load_state_dict(ck['optim_state'])
    if scaler and 'scaler' in ck:
        scaler.load_state_dict(ck['scaler'])
    print(f"Model Resumed epoch {ck['epoch']} from {fname}")
    return ck['epoch'], ck.get('stats', {})

def main(opt):
    random.seed(42); torch.manual_seed(42);
    print_gpu()

    os.makedirs("models", exist_ok=True)
    os.makedirs("statistics", exist_ok=True)

    writer = SummaryWriter()
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    tr_dl = DataLoader(
        SegmDataset(opt.train_img, opt.train_mask, True),
        batch_size=opt.bs, shuffle=True,
        num_workers=opt.workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )
    vl_dl = DataLoader(
        SegmDataset(opt.val_img,   opt.val_mask, False),
        batch_size=opt.bs*2, shuffle=False,
        num_workers=opt.workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )

    model = UNet(out_clean=False, use_ag=opt.use_attention).to(DEVICE)
    optim  = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'max', factor=0.5, patience=3)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    start_epoch = 0
    stats = {"epoch":[], "train_loss":[], "val_loss":[],
             "train_iou":[], "val_iou":[], "val_psnr":[], "val_ssim":[],
             "lr":[], "time_s":[]}

    if opt.resume:
        start_epoch, stats = load_checkpoint(model, optim, scaler, opt.resume)

    best_iou = 0; wait = 0

    for ep in range(start_epoch+1, opt.epochs+1):
        t0 = time.time()
        print(f"\n=== Epoch {ep}/{opt.epochs} ===  lr={optim.param_groups[0]['lr']:.2e}")

        tr_loss, tr_iou, _, _ = run_epoch(model, tr_dl, optim, scaler)
        val_loss, val_iou, val_psnr, val_ssim = run_epoch(model, vl_dl)

        dur = time.time() - t0
        print(f"Train Loss {tr_loss:.3f} IoU {tr_iou:.3f}")
        print(f" Val  Loss {val_loss:.3f} IoU {val_iou:.3f}"
              f" | PSNR {val_psnr:.2f} SSIM {val_ssim:.3f}"
              f" | time {dur:.1f}s")

        writer.add_scalar("Loss/train", tr_loss, ep)
        writer.add_scalar("Loss/val", val_loss, ep)
        writer.add_scalar("IoU/train", tr_iou, ep)
        writer.add_scalar("IoU/val", val_iou, ep)
        writer.add_scalar("PSNR/val", val_psnr, ep)
        writer.add_scalar("SSIM/val", val_ssim, ep)

        stats["epoch"].append(ep)
        stats["train_loss"].append(tr_loss)
        stats["val_loss"].append(val_loss)
        stats["train_iou"].append(tr_iou)
        stats["val_iou"].append(val_iou)
        stats["val_psnr"].append(val_psnr)
        stats["val_ssim"].append(val_ssim)
        stats["lr"].append(optim.param_groups[0]['lr'])
        stats["time_s"].append(dur)

        if val_iou > best_iou:
            best_iou, wait = val_iou, 0
            torch.save(model.state_dict(), "models/unet_base.pt")
            print(f"[BEST] Saved best base model → models/unet_base.pt (IoU={val_iou:.4f})")
        else:
            wait += 1
            if wait >= opt.patience:
                print(f"[STOP] No improvement for {wait} epochs")
                break

        sched.step(val_iou)

        if ep % opt.checkpoint_freq == 0:
            save_checkpoint(model, optim, scaler, ep, stats, f"models/base_ckpt_{run_id}.pt")

        with open(f"statistics/base_stats_{run_id}.csv", "w", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(stats.keys())
            for i in range(len(stats["epoch"])):
                writer_csv.writerow([stats[k][i] for k in stats.keys()])

        if ep % opt.save_detailed_freq == 0:
            with open(f"statistics/base_detailed_{run_id}.json","w") as f:
                json.dump({"epoch":ep}, f, indent=2)

    writer.close()
    print("=== Training completed! ===")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_img", default="dataset/images/train")
    p.add_argument("--train_mask", default="dataset/masks/train")
    p.add_argument("--val_img", default="dataset/images/val")
    p.add_argument("--val_mask", default="dataset/masks/val")
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--use_attention", action="store_true")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--checkpoint_freq", type=int, default=5)
    p.add_argument("--save_detailed_freq", type=int, default=5)
    main(p.parse_args())
