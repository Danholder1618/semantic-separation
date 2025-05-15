import io
import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from pathlib import Path
from model.unet import UNet
from api.services import apply_artifact, preprocess, encode_image

torch.serialization.add_safe_globals([np._core.multiarray.scalar])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load models ---
def load_unet(path, out_clean, use_ag):
    model = UNet(out_clean=out_clean, use_ag=use_ag).to(DEVICE)
    if path.exists():
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        state = ck.get("model_state", ck)
        model.load_state_dict(state, strict=False)
        model.eval()
        return model
    else:
        print(f"[WARN] Model not found: {path}")
        return None

base_model     = load_unet(Path("models/unet_base.pt"),          out_clean=False, use_ag=False)
synth_rec_model= load_unet(Path("models/unet_synth_rec.pt"),     out_clean=True,  use_ag=False)
synth_ag_model = load_unet(Path("models/unet_synth_rec_ag.pt"),  out_clean=True,  use_ag=True)
imp_model      = load_unet(Path("models/unet_improved.pt"),      out_clean=True,  use_ag=True)

app = FastAPI(
    title="Semantic Separation API",
    version="1.0",
    docs_url="/docs", redoc_url="/redoc"
)

# --- Utility: composite clean output ---
def composite_clean(orig: np.ndarray, logits: torch.Tensor, clean_pred: torch.Tensor):
    # orig: HxWx3 uint8, logits/clean_pred: (1,1,H,W)/(1,3,H,W)
    prob = torch.sigmoid(logits)                     # (1,1,H,W) в [0,1]
    orig_t = torch.from_numpy(orig.transpose(2,0,1))[None].float().to(DEVICE)/255.0
    comp = orig_t*(1-prob) + clean_pred*prob         # плавное смешивание
    comp_np = (comp[0].cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
    return comp_np

# --- Endpoints ---

@app.post("/apply_artefact")
async def apply_artefact_endpoint(file: UploadFile = File(...), transparency: float = 0.4):
    img_bytes = await file.read()
    orig, inp, _ = preprocess(img_bytes)
    dirty, _ = apply_artifact(orig, transparency)
    return StreamingResponse(encode_image(dirty), media_type="image/png")

@app.post("/separate_base_mask")
async def separate_base_mask(file: UploadFile = File(...)):
    if base_model is None:
        raise HTTPException(404, "Base-Real model not available")
    img = await file.read()
    _, inp_np, (H,W) = preprocess(img)
    inp = torch.from_numpy(inp_np.transpose(0,3,1,2)).float().to(DEVICE)
    with torch.no_grad():
        logits, _ = base_model(inp)
    mask = (torch.sigmoid(logits)[0,0].cpu().numpy()>0.5).astype(np.uint8)*255
    pil = Image.fromarray(mask).resize((W,H), Image.NEAREST)
    buf = io.BytesIO(); pil.save(buf,"PNG"); buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.post("/separate/synth_rec_mask")
async def synth_rec_mask(file: UploadFile = File(...)):
    if synth_rec_model is None:
        raise HTTPException(404, "Base-Synth-REC model not available")
    img = await file.read()
    _, inp_np, (H,W) = preprocess(img)
    inp = torch.from_numpy(inp_np.transpose(0,3,1,2)).float().to(DEVICE)
    with torch.no_grad():
        logits, _ = synth_rec_model(inp)
    mask = (torch.sigmoid(logits)[0,0].cpu().numpy()>0.5).astype(np.uint8)*255
    pil = Image.fromarray(mask).resize((W,H), Image.NEAREST)
    buf = io.BytesIO(); pil.save(buf,"PNG"); buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.post("/separate/synth_rec_clean")
async def synth_rec_clean(file: UploadFile = File(...)):
    if synth_rec_model is None:
        raise HTTPException(404, "Base-Synth-REC model not available")
    img_bytes = await file.read()
    orig, inp_np, (H, W) = preprocess(img_bytes)
    orig = cv2.resize(orig, (512, 512))
    inp = torch.from_numpy(inp_np.transpose(0,3,1,2)).float().to(DEVICE)
    with torch.no_grad():
        logits, clean_pred = synth_rec_model(inp)
    comp_np = composite_clean(orig, logits, clean_pred)
    buf = io.BytesIO(); Image.fromarray(comp_np).resize((W,H),Image.BILINEAR)\
                .save(buf,"PNG"); buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.post("/separate/synth_ag_clean")
async def synth_ag_clean(file: UploadFile = File(...)):
    if synth_ag_model is None:
        raise HTTPException(404, "Base-Synth-REC+AG model not available")
    img_bytes = await file.read()
    orig, inp_np, (H, W) = preprocess(img_bytes)
    orig = cv2.resize(orig, (512, 512))
    inp = torch.from_numpy(inp_np.transpose(0,3,1,2)).float().to(DEVICE)
    with torch.no_grad():
        logits, clean_pred = synth_ag_model(inp)
    comp_np = composite_clean(orig, logits, clean_pred)
    buf = io.BytesIO(); Image.fromarray(comp_np).resize((W,H),Image.BILINEAR)\
                .save(buf,"PNG"); buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.post("/separate/extract")
async def extract(file: UploadFile = File(...)):
    if imp_model is None:
        raise HTTPException(404, "Improved model not available")
    img_bytes = await file.read()
    orig, inp_np, _ = preprocess(img_bytes)
    inp = torch.from_numpy(inp_np.transpose(0,3,1,2)).float().to(DEVICE)
    with torch.no_grad():
        logits, _ = imp_model(inp)
    mask = (torch.sigmoid(logits)[0,0].cpu().numpy()>0.5).astype(np.uint8)*255
    mask_img = cv2.resize(mask, (orig.shape[1],orig.shape[0]), cv2.INTER_NEAREST)
    extracted = cv2.bitwise_and(orig, orig, mask=mask_img)
    return StreamingResponse(encode_image(extracted), media_type="image/png")

@app.post("/separate/clean")
async def clean(file: UploadFile = File(...)):
    if imp_model is None:
        raise HTTPException(404, "Improved model not available")
    img_bytes = await file.read()
    orig, inp_np, (H, W) = preprocess(img_bytes)
    orig = cv2.resize(orig, (512, 512))
    inp = torch.from_numpy(inp_np.transpose(0,3,1,2)).float().to(DEVICE)
    with torch.no_grad():
        logits, clean_pred = imp_model(inp)
    comp_np = composite_clean(orig, logits, clean_pred)
    buf = io.BytesIO(); Image.fromarray(comp_np).resize((W,H),Image.BILINEAR)\
                .save(buf,"PNG"); buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
