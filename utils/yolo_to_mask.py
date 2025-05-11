from pathlib import Path
from tqdm import tqdm
import cv2
import sys
import numpy as np


img_dir = Path(sys.argv[1]); lbl_dir = Path(sys.argv[2])
out_dir = lbl_dir.parent.parent / "masks" / lbl_dir.name
out_dir.mkdir(parents=True, exist_ok=True)

for txt in tqdm(sorted(lbl_dir.glob("*.txt"))):
    img_path = img_dir / (txt.stem + ".jpg")
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    with open(txt, 'r') as f:
        for line in f:
            _, xc, yc, bw, bh = map(float, line.split())
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    cv2.imwrite(str(out_dir / f"{txt.stem}.png"), mask)
