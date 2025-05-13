from pathlib import Path
from tqdm import tqdm
from albumentations import (Compose, Resize, PadIfNeeded, RandomBrightnessContrast, Perspective, ImageCompression, Blur, RandomGamma, HueSaturationValue, RGBShift)
import numpy as np
import cv2
import glob
import random
import argparse
import yaml


def load_path(path):
    with open(path, "r", encoding="utf8") as f:
        return yaml.safe_load(f)


def augment_logo(logo):
    color_aug = Compose([
        RandomGamma(gamma_limit=(80, 120), p=0.5),
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),
        RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.5),
        RandomBrightnessContrast(0.2, 0.2, p=0.5),
        Blur(3, p=0.3),
        Perspective(scale=(0.05, 0.12), p=0.5),
        ImageCompression(quality_lower=70, quality_upper=100, p=1.0)
    ])
    logo_rgb = logo[..., :3]
    logo_rgb_aug = color_aug(image=logo_rgb)['image']
    logo[..., :3] = logo_rgb_aug
    return logo


def rand_logo(logo_paths, out_size):
    lp = random.choice(logo_paths)
    logo = cv2.imread(lp, cv2.IMREAD_UNCHANGED)
    if logo.shape[2] == 3:
        logo = cv2.cvtColor(logo, cv2.COLOR_BGR2BGRA)
        logo[..., 3] = 255

    logo = augment_logo(logo)

    scale = random.uniform(0.2, 0.5)
    nh = int(out_size * scale)
    logo = cv2.resize(logo, (0, nh), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return logo


def paste_logo(bg, logo):
    h, w = bg.shape[:2]
    lh, lw = logo.shape[:2]
    top = random.randint(0, h - lh)
    left = random.randint(0, w - lw)

    result = bg.copy()
    alpha = logo[..., 3:] / 255.0
    alpha = np.dstack([alpha] * 3)

    result[top:top + lh, left:left + lw] = (alpha * logo[..., :3] + (1 - alpha) * result[top:top + lh, left:left + lw])

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[top:top + lh, left:left + lw] = (logo[..., 3] > 0).astype(np.uint8) * 255
    return result, mask


def main(cfg_path):
    cfg = load_path(cfg_path)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    bg_paths = [f for f in glob.glob(cfg["background_path"]) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(bg_paths)
    assert bg_paths, "No images!"

    logo_paths = glob.glob(cfg["logo_path"])
    random.shuffle(logo_paths)
    assert logo_paths, "No logos!"

    n_images = min(cfg["n_images"], len(bg_paths))
    n_val = int(n_images * cfg["val_split"])

    root = Path(cfg["synthetic_path"])
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "masks" / split).mkdir(parents=True, exist_ok=True)
        (root / "clean" / split).mkdir(parents=True, exist_ok=True)

    resize_aug = Compose([
        Resize(cfg["output_size"], cfg["output_size"]),
        PadIfNeeded(cfg["output_size"], cfg["output_size"])
    ], additional_targets={"mask": "mask"})

    for i in tqdm(range(n_images), desc="Generating synthetic data"):
        bg = cv2.imread(bg_paths[i % len(bg_paths)])
        logo = rand_logo([logo_paths[i % len(logo_paths)]], cfg["output_size"])

        dirty, mask = paste_logo(bg, logo)

        t = random.uniform(*cfg["transparency_range"])
        dirty = cv2.addWeighted(bg, 1 - t, dirty, t, 0)

        data = resize_aug(image=dirty, mask=mask)
        dirty_aug, mask_aug = data["image"], data["mask"]
        clean = cv2.resize(bg, (cfg["output_size"], cfg["output_size"]))

        split = "val" if i < n_val else "train"
        fn = f"syn_{i:05d}.png"
        cv2.imwrite(str(root / "images" / split / fn), dirty_aug)
        cv2.imwrite(str(root / "masks" / split / fn), mask_aug)
        cv2.imwrite(str(root / "clean" / split / fn), clean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="watermark.yaml")
    main(parser.parse_args().cfg)
