import torch
import cv2
import base64
import numpy as np
from fastapi import FastAPI, UploadFile, File
from model.unet import UNet
from api.services import random_logo, put_logo, b64, prep
from pathlib import Path


IMG_SZ = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    "base":     UNet(out_clean=False).to(DEVICE),
    "improved": UNet(out_clean=True, use_ag=True).to(DEVICE)
}
MODELS["base"].load_state_dict(torch.load("models/unet_base.pt", map_location=DEVICE))
MODELS["improved"].load_state_dict(torch.load("models/unet_improved.pt", map_location=DEVICE))
for m in MODELS.values():
    m.eval()


app = FastAPI(title="Semantic-Separation API (PyTorch)")


@app.post("/apply_watermark")
async def apply_watermark(src: UploadFile = File(...), transparency: float = 0.4):
    orig = cv2.imdecode(np.frombuffer(await src.read(), np.uint8), cv2.IMREAD_COLOR)
    wm, mask = put_logo(orig, random_logo(), transparency)
    return {"watermarked_base64": b64(wm, ".jpg"),
            "mask_base64":        b64(mask, ".png")}

@app.post("/separate/{version}")
async def separate(version: str, file: UploadFile = File(...)):
    if version not in MODELS:
        return {"error": "version must be 'base' or 'improved'"}
    orig, inp, (h,w) = prep(await file.read())
    with torch.no_grad():
        logits, clean = MODELS[version](inp)
    mask = (torch.sigmoid(logits)[0,0].cpu().numpy()>0.5).astype(np.uint8)
    mask = cv2.resize(mask, (w, h), cv2.INTER_NEAREST)
    if clean is not None:
        clean = (clean[0].cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
        clean = cv2.resize(clean,(w,h),cv2.INTER_LINEAR)
    watermark = cv2.bitwise_and(orig, orig, mask=mask*255)
    return {
        "mask":       b64(mask*255),
        "watermark":  b64(watermark),
        "clean":      b64(clean) if clean is not None else None
    }
