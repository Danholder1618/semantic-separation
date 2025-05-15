import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


BASE_CSV = 'statistics/base_stats.csv'
SYNTH_REC_CSV = 'statistics/synth_rec_stats.csv'
SYNTH_REC_AG_CSV = 'statistics/synth_rec_ag_stats.csv'

BASE_TB_LOGDIR      = 'runs/base_run'
SYNTH_REC_TB_LOGDIR = 'runs/synth_rec_run'
SYNTH_AG_TB_LOGDIR  = 'runs/synth_rec_ag_run'


def load_tb_scalars(logdir, tag):
    """Load a list of (step, value) for a given tag from TB logs."""
    ea = event_accumulator.EventAccumulator(
        logdir,
        size_guidance={
            event_accumulator.SCALARS: 0,
        }
    )
    ea.Reload()
    try:
        events = ea.Scalars(tag)
    except KeyError:
        return None
    return pd.DataFrame({'epoch': [e.step for e in events], 'value': [e.value for e in events]})

def main():
    df_base = pd.read_csv(BASE_CSV)
    df_synth = pd.read_csv(SYNTH_REC_CSV)
    df_synth_ag  = pd.read_csv(SYNTH_REC_AG_CSV)

    plt.figure(figsize=(8,5))
    plt.plot(df_base['epoch'], df_base['val_iou'],   label='Base-Real')
    plt.plot(df_synth['epoch'], df_synth['iou'],      label='Synth-REC')
    plt.xlabel('Epoch')
    plt.ylabel('Validation IoU')
    plt.title('Segmentation IoU: Base-Real vs Synth-REC')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('mask_iou_comparison.png', dpi=200)
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(df_synth['epoch'], df_synth['psnr_c'],     label='Synth-REC')
    plt.plot(df_synth_ag['epoch'], df_synth_ag['psnr_c'], label='Synth-REC+AG')
    plt.xlabel('Epoch')
    plt.ylabel('Validation PSNR (clean)')
    plt.title('Restoration PSNR: Synth-REC vs Synth-REC+AG')
    plt.legend()
    plt.grid(True)
    plt.savefig('psnr_clean_comparison.png', dpi=200)
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(df_synth['epoch'], df_synth['ssim_c'],     label='Synth-REC')
    plt.plot(df_synth_ag['epoch'], df_synth_ag['ssim_c'], label='Synth-REC+AG')
    plt.xlabel('Epoch')
    plt.ylabel('Validation SSIM (clean)')
    plt.title('Restoration SSIM: Synth-REC vs Synth-REC+AG')
    plt.legend()
    plt.grid(True)
    plt.savefig('ssim_clean_comparison.png', dpi=200)
    plt.show()

    tags = {
        'base_iou': 'FT/IoU/val',
        'synth_iou': 'synth_rec/iou',
        'ag_psnr':   'synth_rec_ag/psnr_c'
    }
    tb_base_iou = load_tb_scalars(BASE_TB_LOGDIR, tags['base_iou'])
    tb_synth_iou = load_tb_scalars(SYNTH_REC_TB_LOGDIR, tags['synth_iou'])
    tb_ag_psnr = load_tb_scalars(SYNTH_AG_TB_LOGDIR, tags['ag_psnr'])

    if tb_base_iou is not None and tb_synth_iou is not None:
        plt.figure(figsize=(8,5))
        plt.plot(tb_base_iou['epoch'], tb_base_iou['value'],   label='TB Base-Real IoU')
        plt.plot(tb_synth_iou['epoch'], tb_synth_iou['value'], label='TB Synth-REC IoU')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.title('TB: Segmentation IoU Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig('tb_mask_iou.png', dpi=200)
        plt.show()

    if tb_ag_psnr is not None:
        plt.figure(figsize=(8,5))
        plt.plot(tb_ag_psnr['epoch'], tb_ag_psnr['value'], label='TB Synth-REC+AG PSNR_clean')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR (clean)')
        plt.title('TB: Restoration PSNR for AttentionGate')
        plt.legend()
        plt.grid(True)
        plt.savefig('tb_psnr_ag.png', dpi=200)
        plt.show()

    print("\nSummary statistics:")
    for name, df in [('Base-Real IoU', df_base['val_iou']),
                     ('Synth-REC IoU', df_synth['iou']),
                     ('Synth-REC PSNR', df_synth['psnr_c']),
                     ('Synth-REC+AG PSNR', df_synth_ag['psnr_c'])]:
        print(f"{name}: max={df.max():.3f}, mean={df.mean():.3f}")

if __name__ == '__main__':
    main()
