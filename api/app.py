import io
import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from pathlib import Path
from model.unet import UNet
from api.services import apply_artifact, preprocess, IMG_SZ, encode_image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


base_model = UNet(out_clean=False, use_ag=False).to(DEVICE)
base_path  = Path("models/unet_base.pt")
if base_path.exists():
    base_model.load_state_dict(torch.load(base_path, map_location=DEVICE))
    base_model.eval()
else:
    print(f"[WARN] Base model not found: {base_path}")

imp_model = UNet(out_clean=True, use_ag=True).to(DEVICE)
imp_path  = Path("models/unet_improved.pt")
if imp_path.exists():
    imp_model.load_state_dict(torch.load(imp_path, map_location=DEVICE))
    imp_model.eval()
else:
    print(f"[WARN] Improved model not found: {imp_path}")

app = FastAPI(
    title="Semantic Separation API",
    description="API for image processing and artifact separation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.post("/apply_artefact")
async def apply_artefact_endpoint(file: UploadFile = File(...), transparency: float = 0.4):
    """Возвращает изображение с артефактами"""
    img_bytes = await file.read()
    orig, _, _ = preprocess(img_bytes)
    orig_rs = cv2.resize(orig, (IMG_SZ, IMG_SZ))
    dirty, _ = apply_artifact(orig_rs, transparency)
    return StreamingResponse(encode_image(dirty), media_type="image/png")


@app.post("/separate_base_mask")
async def separate_base_mask_endpoint(file: UploadFile = File(...)):
    """Возвращает только маску"""
    if not base_path.exists():
        raise HTTPException(404, "Base model not available")

    img_bytes = await file.read()
    _, inp_np, (H, W) = preprocess(img_bytes)
    inp = torch.from_numpy(inp_np.transpose(0, 3, 1, 2)).float().to(DEVICE)

    with torch.no_grad():
        logits, _ = base_model(inp)

    mask_np = (torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask_np).resize((W, H), Image.NEAREST)

    mask_buf = io.BytesIO()
    mask_pil.save(mask_buf, format="PNG")
    mask_buf.seek(0)

    return StreamingResponse(mask_buf, media_type="image/png")


@app.post("/separate/extract")
async def extract_endpoint(file: UploadFile = File(...)):
    """Возвращает извлеченное наложенное изображение"""
    if not imp_path.exists():
        raise HTTPException(404, "Improved model not available")

    img_bytes = await file.read()
    orig, inp_np, (H, W) = preprocess(img_bytes)
    inp = torch.from_numpy(inp_np.transpose(0, 3, 1, 2)).float().to(DEVICE)

    with torch.no_grad():
        logits, _ = imp_model(inp)

    mask_np = (torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
    mask_img = (mask_np * 255).astype(np.uint8)
    extracted = cv2.bitwise_and(orig, orig, mask=mask_img)

    return StreamingResponse(encode_image(extracted), media_type="image/png")


@app.post("/separate/clean")
async def clean_endpoint(file: UploadFile = File(...)):
    """Возвращает очищенное изображение"""
    if not imp_path.exists():
        raise HTTPException(404, "Improved model not available")

    img_bytes = await file.read()
    _, inp_np, (H, W) = preprocess(img_bytes)
    inp = torch.from_numpy(inp_np.transpose(0, 3, 1, 2)).float().to(DEVICE)

    with torch.no_grad():
        _, clean_pred = imp_model(inp)

    clean_np = (clean_pred[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    clean_pil = Image.fromarray(clean_np).resize((W, H), Image.BILINEAR)

    clean_buf = io.BytesIO()
    clean_pil.save(clean_buf, format="PNG")
    clean_buf.seek(0)

    return StreamingResponse(clean_buf, media_type="image/png")
