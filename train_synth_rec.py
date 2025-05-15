import os
import time
import datetime
import random
import csv
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from model.unet import UNet
import cv2


IMG_SIZE = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LAMBDA_IMG = 10.0
LAMBDA_BG  = 0.1
EPOCHS = 20
BS = 4
LR = 3e-4
PATIENCE = 8


def print_gpu():
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name}, {p.total_memory/1e9:.1f} GB")
    else:
        print("Using CPU")

def iou(logits, gt, eps=1e-6):
    p = torch.sigmoid(logits)
    inter = (p>0.5).float().mul(gt).sum()
    union = (p>0.5).float().sum() + gt.sum() - inter
    return (inter+eps)/(union+eps)

class SynDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, clean_dir, mask_dir, augment=False):
        self.imgs = sorted(Path(img_dir).glob('*.[pj][pn]g'))
        self.clean = {p.stem: Path(clean_dir)/p.name for p in self.imgs}
        self.mask  = {p.stem: Path(mask_dir)/f'{p.stem}.png' for p in self.imgs}
        self.aug = augment

    def __len__(self): return len(self.imgs)
    def __getitem__(self,i):
        p = self.imgs[i]
        imp   = cv2.resize(cv2.imread(str(p))[:,:,::-1],(IMG_SIZE,IMG_SIZE))
        clean = cv2.resize(cv2.imread(str(self.clean[p.stem]))[:,:,::-1],(IMG_SIZE,IMG_SIZE))
        m    = cv2.resize(cv2.imread(str(self.mask[p.stem]),0),(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_NEAREST)
        if self.aug and random.random()<0.5:
            imp, clean, m = map(lambda x: cv2.flip(x,1), (imp,clean,m))
        w = torch.from_numpy(imp.transpose(2,0,1)).float()/255.
        c = torch.from_numpy(clean.transpose(2,0,1)).float()/255.
        m = torch.from_numpy((m[None]/255.)).float()
        return w,c,m

def run_epoch(model, loader, optim=None, scaler=None):
    train = optim is not None
    model.train() if train else model.eval()
    seg_loss = torch.nn.BCEWithLogitsLoss()
    img_loss = torch.nn.L1Loss()
    stats = {'loss':0,'iou':0,'psnr_m':0,'ssim_m':0,'psnr_c':0,'ssim_c':0}
    cnt_m=cnt_c=0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for w,c,m in tqdm(loader,desc='train' if train else 'val'):
            w,c,m = w.to(DEVICE),c.to(DEVICE),m.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                logits,clean_pred = model(w)
                l_seg   = seg_loss(logits, m)
                l_clean = img_loss(clean_pred*m, c*m)
                l_bg    = img_loss(clean_pred*(1-m), c*(1-m))
                loss = l_seg + LAMBDA_IMG*l_clean + LAMBDA_BG*l_bg
            if train:
                optim.zero_grad(); scaler.scale(loss).backward()
                scaler.step(optim); scaler.update()
            stats['loss'] += loss.item()
            stats['iou']  += iou(logits,m).item()
            if not train:
                probs = torch.sigmoid(logits).cpu().numpy()
                gt    = m.cpu().numpy()
                cp    = clean_pred.cpu().numpy(); cc = c.cpu().numpy()
                bs = probs.shape[0]
                for i in range(bs):
                    pm,gm = probs[i,0], gt[i,0]
                    stats['psnr_m'] += psnr(gm, pm, data_range=1)
                    stats['ssim_m'] += ssim(gm, pm, data_range=1)
                    cnt_m+=1
                    pc = (cp[i]*gm).transpose(1,2,0)
                    gc = (cc[i]*gm).transpose(1,2,0)
                    stats['psnr_c'] += psnr(gc, pc, data_range=1)
                    stats['ssim_c'] += ssim(gc, pc, data_range=1, channel_axis=2)
                    cnt_c+=1
    n = len(loader)
    avg = {k: stats[k]/n for k in ('loss','iou')}
    if not train:
        avg.update({
            'psnr_m':stats['psnr_m']/cnt_m,
            'ssim_m':stats['ssim_m']/cnt_m,
            'psnr_c':stats['psnr_c']/cnt_c,
            'ssim_c':stats['ssim_c']/cnt_c
        })
    else:
        avg.update({'psnr_m':0,'ssim_m':0,'psnr_c':0,'ssim_c':0})
    return avg

def save_ckpt(m,o,s,ep,st,path):
    ck={'epoch':ep,'model_state':m.state_dict(),'optim_state':o.state_dict(),'stats':st}
    if s: ck['scaler']=s.state_dict()
    torch.save(ck,path); print(f'Saved {path}')

def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    os.makedirs('models',exist_ok=True); os.makedirs('statistics',exist_ok=True)
    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/synth_rec_{run_id}')

    tr = DataLoader(SynDataset('dataset/synthetic/images/train',
                               'dataset/synthetic/clean/train',
                               'dataset/synthetic/masks/train', augment=True),
                    batch_size=BS,shuffle=True, num_workers=0)
    vl = DataLoader(SynDataset('dataset/synthetic/images/val',
                               'dataset/synthetic/clean/val',
                               'dataset/synthetic/masks/val', augment=False),
                    batch_size=BS*2,shuffle=False, num_workers=0)

    model = UNet(out_clean=True,use_ag=False).to(DEVICE)
    br = Path('models/unet_base.pt')
    if br.exists():
        ck = torch.load(br,map_location=DEVICE)
        model.load_state_dict(ck.get('model_state',ck),strict=False)
        print('Loaded base-real weights')
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    stats={k:[] for k in ['epoch','loss','iou','psnr_m','ssim_m','psnr_c','ssim_c','lr']}
    best=0; wait=0
    for ep in range(1,EPOCHS+1):
        t0=time.time()
        trm = run_epoch(model,tr,opt,scaler)
        vlm = run_epoch(model,vl)
        dt=time.time()-t0; lr=opt.param_groups[0]['lr']
        print(f'Ep{ep}/{EPOCHS} | tr_loss={trm["loss"]:.4f} iou={trm["iou"]:.4f} | '
              f'val_loss={vlm["loss"]:.4f} iou={vlm["iou"]:.4f} '
              f'psnr_c={vlm["psnr_c"]:.2f} ssim_c={vlm["ssim_c"]:.3f} time={dt:.1f}s')
        for tag in ['loss','iou','psnr_c','ssim_c']:
            writer.add_scalar(f'synth_rec/{tag}',vlm[tag],ep)
        for k in stats:
            if k=='epoch': stats[k].append(ep)
            elif k=='lr':    stats[k].append(lr)
            else:            stats[k].append(vlm[k])
        if vlm['iou']>best:
            best=vlm['iou']; wait=0
            save_ckpt(model,opt,scaler,ep,stats,'models/unet_synth_rec.pt')
        else:
            wait+=1
            if wait>=PATIENCE: print('Early stop'); break
        sch.step(vlm['iou'])
    # CSV
    csvf=f'statistics/synth_rec_stats_{run_id}.csv'
    with open(csvf,'w',newline='') as f:
        wr=csv.writer(f); wr.writerow(stats.keys())
        for i in range(len(stats['epoch'])):
            wr.writerow([stats[k][i] for k in stats.keys()])
    writer.close()
    print('Done synth_rec')

if __name__=='__main__':
    main()
