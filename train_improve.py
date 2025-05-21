import torch
import os
import random
import time
import datetime
import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from model.unet import UNet

# Global constants
IMG_SIZE = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LAMBDA_IMG = 10.0  # weight for clean image reconstruction loss

# Enable CuDNN benchmark for performance (note: can introduce nondeterminism)
torch.backends.cudnn.benchmark = True

def print_gpu():
    """Print GPU info or CPU notice."""
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU found: {p.name}, {p.total_memory/1e9:.1f} GB")
    else:
        print("No GPU found – using CPU")

def iou(logits: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> float:
    """Compute IoU metric for segmentation mask logits vs ground truth mask."""
    probs = torch.sigmoid(logits)
    # Binarize predictions at 0.5 threshold and compute intersection and union
    inter = (probs > 0.5).float().mul(gt).sum()
    union = (probs > 0.5).float().sum() + gt.sum() - inter
    return float((inter + eps) / (union + eps))

class SynDataset(Dataset):
    """
    Synthetic dataset that provides tuples (watermarked_image, clean_image, mask).
    Expects directory structure with separate folders for images, clean images, and masks.
    """
    def __init__(self, img_dir: str, clean_dir: str, mask_dir: str, augment: bool = False):
        self.img_dir = Path(img_dir)
        self.clean_dir = Path(clean_dir)
        self.mask_dir = Path(mask_dir)
        # Collect image file paths (assuming images are .png or .jpg; masks and clean images use same stem)
        self.imgs = sorted(list(self.img_dir.glob("*.[pj][pn]g")))
        # Create lookup for clean images and masks by stem name
        self.clean_paths = {p.stem: (self.clean_dir / p.name) for p in self.imgs}
        self.mask_paths  = {p.stem: (self.mask_dir / f"{p.stem}.png") for p in self.imgs}
        self.augment = augment
        print(f"Loaded {len(self.imgs)} samples from {img_dir} (augment={augment})")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx: int):
        import cv2
        # Read watermarked image (BGR to RGB)
        w_img = cv2.imread(str(self.imgs[idx]))
        # Read corresponding clean image and mask
        c_img = cv2.imread(str(self.clean_paths[self.imgs[idx].stem]))
        m_img = cv2.imread(str(self.mask_paths[self.imgs[idx].stem]), cv2.IMREAD_GRAYSCALE)
        # Convert BGR to RGB for color images
        w_img = w_img[:, :, ::-1] if w_img is not None else np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
        c_img = c_img[:, :, ::-1] if c_img is not None else np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
        # Resize to uniform size (IMG_SIZE x IMG_SIZE)
        w_img = cv2.resize(w_img, (IMG_SIZE, IMG_SIZE))
        c_img = cv2.resize(c_img, (IMG_SIZE, IMG_SIZE))
        m_img = cv2.resize(m_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        # Optional augmentation (horizontal flip)
        if self.augment and random.random() < 0.5:
            w_img = cv2.flip(w_img, 1)
            c_img = cv2.flip(c_img, 1)
            m_img = cv2.flip(m_img, 1)
        # Convert images and mask to torch tensors and normalize to [0,1]
        w_tensor = torch.from_numpy(w_img.transpose(2, 0, 1)).float() / 255.0   # shape (3, H, W)
        c_tensor = torch.from_numpy(c_img.transpose(2, 0, 1)).float() / 255.0   # shape (3, H, W)
        m_tensor = torch.from_numpy(m_img[np.newaxis, ...]).float() / 255.0     # shape (1, H, W)
        return w_tensor, c_tensor, m_tensor

def run_epoch(model, loader, optim=None, scaler=None):
    """Run one training or validation epoch. Returns metrics: loss, IoU, PSNR(mask), SSIM(mask), PSNR(clean), SSIM(clean)."""
    train_mode = optim is not None
    model.train() if train_mode else model.eval()
    # Define loss functions
    seg_criterion = torch.nn.BCEWithLogitsLoss()
    img_criterion = torch.nn.L1Loss()
    # Initialize accumulators for metrics
    total_loss = 0.0
    total_iou = 0.0
    total_psnr_mask = 0.0
    total_ssim_mask = 0.0
    total_psnr_clean = 0.0
    total_ssim_clean = 0.0
    count_mask = 0
    count_clean = 0
    # Use no_grad for validation (to save memory) and enable_grad for training
    context_manager = torch.no_grad() if not train_mode else torch.enable_grad()
    with context_manager:
        for w, c, m in tqdm(loader, desc=("train" if train_mode else "val  "), leave=False):
            # Move data to device
            w = w.to(DEVICE, non_blocking=True)
            c = c.to(DEVICE, non_blocking=True)
            m = m.to(DEVICE, non_blocking=True)
            # Mixed precision context if using GPU (scaler is not None when GPU is available)
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                logits, pred_clean = model(w)  # forward pass through UNet
                # Segmentation mask loss (binary cross-entropy)
                seg_loss = seg_criterion(logits, m)
                # Clean image reconstruction loss (L1 on masked regions only)
                clean_loss = 0.0
                if pred_clean is not None:
                    # Only compute L1 loss on regions where mask = 1 (corrupted regions in watermarked image)
                    clean_loss = img_criterion(pred_clean * m, c * m)
                loss = seg_loss + LAMBDA_IMG * clean_loss
            # Backpropagation and optimizer step if training
            if train_mode:
                optim.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
                else:
                    loss.backward()
                    optim.step()
            # Accumulate loss and IoU for this batch
            total_loss += loss.item()
            total_iou += iou(logits, m)
            # If validation, compute PSNR/SSIM metrics for mask and cleaned image
            if not train_mode:
                # Convert logits to probability map for mask
                probs = torch.sigmoid(logits).cpu().numpy()        # shape (N, 1, H, W)
                gt_mask = m.cpu().numpy()                          # shape (N, 1, H, W)
                pred_clean_np = pred_clean.cpu().numpy() if pred_clean is not None else None  # shape (N, 3, H, W)
                clean_np = c.cpu().numpy()                         # shape (N, 3, H, W)
                batch_size = probs.shape[0]
                for i in range(batch_size):
                    # Flatten mask and ground truth for metrics
                    p_mask = probs[i, 0]    # predicted mask (2D array)
                    g_mask = gt_mask[i, 0]  # ground truth mask (2D array)
                    # PSNR and SSIM for segmentation mask (treat mask as image with values 0-1)
                    total_psnr_mask += psnr(g_mask, p_mask, data_range=1)
                    total_ssim_mask += ssim(g_mask, p_mask, data_range=1)
                    count_mask += 1
                    if pred_clean_np is not None:
                        # Clean image metrics on masked region only
                        # Multiply by ground truth mask to consider only originally corrupted region
                        p_clean_img = (pred_clean_np[i] * gt_mask[i])  # shape (3, H, W)
                        g_clean_img = (clean_np[i] * gt_mask[i])       # shape (3, H, W)
                        # Transpose to (H, W, 3) for SSIM (skimage expects channel axis = 2 for color images)
                        p_clean_img = p_clean_img.transpose(1, 2, 0)
                        g_clean_img = g_clean_img.transpose(1, 2, 0)
                        # PSNR and SSIM for cleaned image region
                        total_psnr_clean += psnr(g_clean_img, p_clean_img, data_range=1)
                        total_ssim_clean += ssim(g_clean_img, p_clean_img, data_range=1, channel_axis=2)
                        count_clean += 1
    # Compute average metrics for the epoch
    avg_loss = total_loss / len(loader)
    avg_iou = total_iou / len(loader)
    if train_mode:
        # During training epoch, we do not compute image quality metrics
        return avg_loss, avg_iou, 0.0, 0.0, 0.0, 0.0
    # For validation, compute average metrics over all images
    avg_psnr_mask = total_psnr_mask / count_mask if count_mask > 0 else 0.0
    avg_ssim_mask = total_ssim_mask / count_mask if count_mask > 0 else 0.0
    avg_psnr_clean = total_psnr_clean / count_clean if count_clean > 0 else 0.0
    avg_ssim_clean = total_ssim_clean / count_clean if count_clean > 0 else 0.0
    return avg_loss, avg_iou, avg_psnr_mask, avg_ssim_mask, avg_psnr_clean, avg_ssim_clean

def save_checkpoint(model, optim, scaler, epoch, stats, fname):
    """Save training checkpoint including model, optimizer, scaler states and statistics."""
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optim_state': optim.state_dict(),
        'stats': stats
    }
    if scaler:
        checkpoint['scaler'] = scaler.state_dict()
    torch.save(checkpoint, fname)
    print(f"Model checkpoint saved to {fname}")

def main():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    print_gpu()
    # Ensure output directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("statistics", exist_ok=True)
    # Prepare TensorBoard writer with a unique log directory
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/improve_{run_id}")
    # Paths for training and validation data
    train_img_dir = "dataset/synthetic/images/train"
    train_clean_dir = "dataset/synthetic/clean/train"
    train_mask_dir = "dataset/synthetic/masks/train"
    val_img_dir = "dataset/synthetic/images/val"
    val_clean_dir = "dataset/synthetic/clean/val"
    val_mask_dir = "dataset/synthetic/masks/val"
    # Training parameters
    batch_size = 4
    epochs = 20
    learning_rate = 3e-4
    patience = 8
    save_model_path = "models/unet_improved.pt"
    # Create DataLoaders for training and validation
    train_loader = DataLoader(
        SynDataset(train_img_dir, train_clean_dir, train_mask_dir, augment=True),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        SynDataset(val_img_dir, val_clean_dir, val_mask_dir, augment=False),
        batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=True
    )
    # Initialize model, optimizer, scheduler, and AMP scaler
    model = UNet(out_clean=True, use_ag=True).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    # Load base model weights if available
    if os.path.exists("models/unet_base.pt"):
        base_weights = torch.load("models/unet_base.pt", map_location=DEVICE)
        model.load_state_dict(base_weights, strict=False)
        print("Loaded base model weights from models/unet_base.pt")
    else:
        print("Base model weights not found, training from scratch.")
    # Prepare statistics storage
    stats = {"epoch": [], "train_loss": [], "val_loss": [], "train_iou": [], "val_iou": [],
             "val_psnr": [], "val_ssim": [], "val_psnr_clean": [], "val_ssim_clean": [],
             "lr": [], "time_s": []}
    best_iou = 0.0
    wait = 0
    # Training loop
    for ep in range(1, epochs + 1):
        t0 = time.time()
        print(f"\n=== Epoch {ep}/{epochs} === lr={optim.param_groups[0]['lr']:.2e}")
        # Run one epoch for training and validation
        tr_loss, tr_iou, _, _, _, _ = run_epoch(model, train_loader, optim, scaler)
        val_loss, val_iou, val_psnr_mask, val_ssim_mask, val_psnr_clean, val_ssim_clean = run_epoch(model, val_loader)
        duration = time.time() - t0
        # Print epoch results
        print(f"Train Loss: {tr_loss:.4f} | IoU: {tr_iou:.4f}")
        print(f" Val  Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | "
              f"Mask PSNR: {val_psnr_mask:.2f} SSIM: {val_ssim_mask:.3f} | "
              f"Clean PSNR: {val_psnr_clean:.2f} SSIM: {val_ssim_clean:.3f} | Time: {duration:.1f}s")
        # Log metrics to TensorBoard
        writer.add_scalar("FT/Loss/train", tr_loss, ep)
        writer.add_scalar("FT/Loss/val", val_loss, ep)
        writer.add_scalar("FT/IoU/train", tr_iou, ep)
        writer.add_scalar("FT/IoU/val", val_iou, ep)
        writer.add_scalar("FT/PSNR/mask", val_psnr_mask, ep)
        writer.add_scalar("FT/SSIM/mask", val_ssim_mask, ep)
        writer.add_scalar("FT/PSNR/clean", val_psnr_clean, ep)
        writer.add_scalar("FT/SSIM/clean", val_ssim_clean, ep)
        # Store metrics in stats dictionary
        stats["epoch"].append(ep)
        stats["train_loss"].append(tr_loss)
        stats["val_loss"].append(val_loss)
        stats["train_iou"].append(tr_iou)
        stats["val_iou"].append(val_iou)
        stats["val_psnr"].append(val_psnr_mask)
        stats["val_ssim"].append(val_ssim_mask)
        stats["val_psnr_clean"].append(val_psnr_clean)
        stats["val_ssim_clean"].append(val_ssim_clean)
        stats["lr"].append(optim.param_groups[0]['lr'])
        stats["time_s"].append(duration)
        # Check for improvement and handle early stopping
        if val_iou > best_iou:
            best_iou = val_iou
            wait = 0
            save_checkpoint(model, optim, scaler, ep, stats, save_model_path)
            print(f"[BEST] Saved best model to {save_model_path} (IoU={val_iou:.4f})")
        else:
            wait += 1
            if wait >= patience:
                print(f"[STOP] No improvement for {wait} epochs. Early stopping.")
                break
        # Step learning rate scheduler (monitoring validation IoU)
        sched.step(val_iou)
    # Save metrics to CSV file
    csv_path = f"statistics/improve_stats_{run_id}.csv"
    with open(csv_path, "w", newline="") as f:
        writer_csv = csv.writer(f)
        # Write header
        writer_csv.writerow(stats.keys())
        # Write each epoch's metrics
        for i in range(len(stats["epoch"])):
            writer_csv.writerow([stats[key][i] for key in stats.keys()])
    print(f"Saved training metrics to {csv_path}")
    writer.close()
    print("=== Training completed ===")

if __name__ == "__main__":
    main()
