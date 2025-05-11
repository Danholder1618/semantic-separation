import numpy as np
import cv2
import base64
import random
import glob


IMG_SZ = 512
LOGOS = glob.glob("dataset/logos/combined/*.png")


def b64(img_arr, ext=".png"):
    ok, buf = cv2.imencode(ext, img_arr)
    return base64.b64encode(buf).decode()

def prep(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    inp = cv2.resize(img, (IMG_SZ, IMG_SZ)).astype(np.float32) / 255.
    return img, inp[None], (h, w)

def random_logo():
    lg = cv2.imread(random.choice(LOGOS), cv2.IMREAD_UNCHANGED)
    if lg.shape[2] == 3:
        lg = cv2.cvtColor(lg, cv2.COLOR_BGR2BGRA)
        lg[..., 3] = 255
    return lg

def put_logo(img, logo, alpha):
    h, w = img.shape[:2]
    scale = random.uniform(0.2, 0.4)
    logo = cv2.resize(logo, (0, int(h * scale)))
    lh, lw = logo.shape[:2]
    top, left = random.randint(0, h - lh), random.randint(0, w - lw)

    blend = img.copy()
    a = (logo[..., 3:] / 255.0) * alpha
    a3 = np.dstack([a]*3)
    blend[top:top+lh, left:left+lw] = a3 * logo[..., :3] + (1 - a3) * blend[top:top+lh, left:left+lw]

    mask = np.zeros((h, w), np.uint8)
    mask[top:top+lh, left:left+lw] = (logo[..., 3] > 0).astype(np.uint8) * 255
    return blend, mask
