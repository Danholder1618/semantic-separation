import torch
import argparse
import os
import random
import time
import datetime
import csv
import json
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from model.unet import UNet

IMG_SIZE = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
LAMBDA_IMG = 10.0

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
    return (inter + eps)/(union + eps)

class SynDataset(Dataset):
    def __init__(self, img_dir, clean_dir, mask_dir, augment=True):
        img_dir, clean_dir, mask_dir = map(Path, (img_dir, clean_dir, mask_dir))
        self.imgs   = sorted(img_dir.glob("*.png"))
        self.cleans = {p.stem: clean_dir / p.name for p in self.imgs}
        self.masks  = {p.stem: mask_dir / f"{p.stem}.png" for p in self.imgs}
        self.aug = augment
        print(f"Loaded {len(self.imgs)} synthetic samples (aug={augment})")

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        import cv2
        w = cv2.imread(str(self.imgs[idx]))[:,:,::-1]
        c = cv2.imread(str(self.cleans[self.imgs[idx].stem]))[:,:,::-1]
        m = cv2.imread(str(self.masks[self.imgs[idx].stem]), cv2.IMREAD_GRAYSCALE)
        w = cv2.resize(w, (IMG_SIZE, IMG_SIZE))
        c = cv2.resize(c, (IMG_SIZE, IMG_SIZE))
        m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), cv2.INTER_NEAREST)
        if self.aug and random.random()<.5:
            w,c,m = cv2.flip(w,1), cv2.flip(c,1), cv2.flip(m,1)
        w = torch.from_numpy(w.transpose(2,0,1)).float()/255.
        c = torch.from_numpy(c.transpose(2,0,1)).float()/255.
        m = torch.from_numpy(m[None]).float()/255.
        return w, c, m

def run_epoch(model, loader, optim=None, scaler=None):
    train = optim is not None
    if train:
        model.train()
    else:
        model.eval()

    BCE = torch.nn.BCEWithLogitsLoss()
    L1  = torch.nn.L1Loss()

    total_loss = 0.0
    total_iou  = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count      = 0

    if not train:
        with torch.no_grad():
            for w,c,m in tqdm(loader, desc="val ", leave=False):
                w,c,m = w.to(DEVICE), c.to(DEVICE), m.to(DEVICE)
                logits, pclean = model(w)
                seg_loss   = BCE(logits, m)
                clean_loss = LAMBDA_IMG * L1(pclean*m, c*m)
                loss = seg_loss + clean_loss

                probs = torch.sigmoid(logits)
                biou = iou(logits, m).item()
                total_loss += loss.item()
                total_iou  += biou

                p_np = probs[:,0].cpu().numpy()
                g_np = m[:,0].cpu().numpy()
                for p_img, g_img in zip(p_np, g_np):
                    total_psnr += psnr(g_img, p_img, data_range=1)
                    total_ssim += ssim(g_img, p_img, data_range=1)
                    count += 1
    else:
        for w,c,m in tqdm(loader, desc="train", leave=False):
            w,c,m = w.to(DEVICE), c.to(DEVICE), m.to(DEVICE)
            with torch.amp.autocast(device_type='cuda', enabled=bool(scaler)):
                logits, pclean = model(w)
                seg_loss   = BCE(logits, m)
                clean_loss = LAMBDA_IMG * L1(pclean*m, c*m)
                loss = seg_loss + clean_loss

            optim.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward(); optim.step()

            total_loss += loss.item()
            total_iou  += iou(logits, m).item()

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
    print(f"[CKPT] Saved to {fname}")

def load_checkpoint(model, optim, scaler, fname):
    if not os.path.exists(fname):
        print(f"[CKPT] {fname} not found → scratch")
        return 0, {}
    ck = torch.load(fname, map_location=DEVICE)
    model.load_state_dict(ck['model_state'])
    optim.load_state_dict(ck['optim_state'])
    if scaler and 'scaler' in ck:
        scaler.load_state_dict(ck['scaler'])
    print(f"[CKPT] Resume epoch {ck['epoch']} from {fname}")
    return ck['epoch'], ck.get('stats', {})

def main(opt):
    random.seed(42); torch.manual_seed(42)
    print_gpu()

    os.makedirs("models", exist_ok=True)
    os.makedirs("statistics", exist_ok=True)
    writer = SummaryWriter()
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    tr_dl = DataLoader(
        SynDataset(opt.syn_img, opt.syn_clean, opt.syn_mask, True),
        batch_size=opt.bs, shuffle=True,
        num_workers=opt.workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )
    vl_dl = DataLoader(
        SynDataset(opt.val_img, opt.val_clean, opt.val_mask, False),
        batch_size=opt.bs*2, shuffle=False,
        num_workers=opt.workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )

    model = UNet(out_clean=True, use_ag=True).to(DEVICE)
    if os.path.exists(opt.base_model):
        model.load_state_dict(torch.load(opt.base_model, map_location=DEVICE))
        print(f"Loaded base model {opt.base_model}")
    else:
        print("Base model not found — training from scratch")

    optim = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'max', factor=0.5, patience=3)
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
        print(f"\nFine-tune Epoch {ep}/{opt.epochs}")

        tr_loss, tr_iou, _, _ = run_epoch(model, tr_dl, optim, scaler)
        val_loss, val_iou, val_psnr, val_ssim = run_epoch(model, vl_dl)

        dur = time.time() - t0
        print(f"Train L {tr_loss:.3f} IoU {tr_iou:.3f}")
        print(f" Val  L {val_loss:.3f} IoU {val_iou:.3f}"
              f" | PSNR {val_psnr:.2f} SSIM {val_ssim:.3f}"
              f" | {dur:.1f}s")

        # TensorBoard
        writer.add_scalar("FT/Loss/train", tr_loss, ep)
        writer.add_scalar("FT/Loss/val", val_loss, ep)
        writer.add_scalar("FT/IoU/train", tr_iou, ep)
        writer.add_scalar("FT/IoU/val", val_iou, ep)
        writer.add_scalar("FT/PSNR/val", val_psnr, ep)
        writer.add_scalar("FT/SSIM/val", val_ssim, ep)

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
            torch.save(model.state_dict(), "models/unet_improved.pt")
            print(f"[BEST FT] Saved best improved model → models/unet_improved.pt (IoU={val_iou:.4f})")
        else:
            wait += 1
            if wait >= opt.patience:
                print(f"[STOP FT] no imp. for {wait} epochs")
                break

        sched.step(val_iou)

        if ep % opt.checkpoint_freq == 0:
            save_checkpoint(model, optim, scaler, ep, stats,
                            f"models/ft_ckpt_{run_id}.pt")

        with open(f"statistics/ft_stats_{run_id}.csv","w",newline="") as f:
            w = csv.writer(f)
            w.writerow(stats.keys())
            for i in range(len(stats["epoch"])):
                w.writerow([stats[k][i] for k in stats.keys()])

        if ep % opt.save_detailed_freq == 0:
            with open(f"statistics/ft_detailed_{run_id}.json","w") as f:
                json.dump({"epoch":ep}, f, indent=2)

    writer.close()
    print("=== Fine-tuning completed! ===")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--syn_img", default="dataset/synthetic/wm")
    p.add_argument("--syn_clean", default="dataset/synthetic/clean")
    p.add_argument("--syn_mask", default="dataset/synthetic/mask")
    p.add_argument("--val_img", default="dataset/synthetic_val/wm")
    p.add_argument("--val_clean", default="dataset/synthetic_val/clean")
    p.add_argument("--val_mask", default="dataset/synthetic_val/mask")
    p.add_argument("--base_model",default="models/unet_base.pt")
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--checkpoint_freq", type=int, default=5)
    p.add_argument("--save_detailed_freq", type=int, default=5)
    main(p.parse_args())
