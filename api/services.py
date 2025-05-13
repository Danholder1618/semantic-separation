import io
import cv2
import numpy as np
import glob
from dataset_generator.generator import load_path, rand_logo, paste_logo

cfg = load_path("dataset/watermark.yaml")
logo_paths = glob.glob(cfg["logo_path"])
IMG_SZ = cfg["output_size"]

def apply_artifact(img: np.ndarray, transparency: float = 0.4):
    logo = rand_logo(logo_paths, IMG_SZ)

    dirty, mask = paste_logo(img, logo)

    dirty = cv2.addWeighted(img, 1 - transparency, dirty, transparency, 0)
    return dirty, mask

def preprocess(img_bytes: bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    inp = cv2.resize(img, (IMG_SZ, IMG_SZ)).astype(np.float32) / 255.0
    return img, inp[None], (h, w)

def encode_image(img: np.ndarray, ext: str = ".png"):
    ok, buf = cv2.imencode(ext, img)
    return io.BytesIO(buf.tobytes())
