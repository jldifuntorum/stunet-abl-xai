"""Plot main numerical summaries from the Zenodo Figure_Source_Data.zip archive.

Run: python scripts/plot_source_data.py --archive Figure_Source_Data.zip
"""
import argparse
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REGIMES=('neutral','transitional','convective')
LABELS=('Near-neutral','Weakly convective','Convective')
SPACINGS=(2,4,16,40,80)


def source(z,filename):
    paths=[p for p in z.namelist() if p.endswith('/'+filename) or p==filename]
    if len(paths)!=1:raise FileNotFoundError(f'Expected one {filename}; found {len(paths)}')
    return BytesIO(z.read(paths[0]))


def figure3(z,out):
    fig,axes=plt.subplots(2,3,figsize=(11,5),constrained_layout=True)
    for i,(key,title) in enumerate(zip(REGIMES,LABELS)):
        f=f'IG_ht_{key}_dt=4_samples_norm.npy'
        a=np.load(source(z,f),allow_pickle=False)
        b=np.load(source(z,f'Occ_ht_{key}_dt=4_samples_mean_prediction.npy'),allow_pickle=False)
        axes[0,i].imshow(a.mean(axis=0),cmap='viridis',vmin=0,vmax=.4)
        limit=float(max(abs(b.mean(axis=0)).max(),1e-8))
        axes[1,i].imshow(b.mean(axis=0),cmap='coolwarm',vmin=-limit,vmax=limit)
        axes[0,i].set_title(title)
        for row in range(2):
            ax=axes[row,i];ax.set_xticks(range(4),['12','8','4','0']);ax.set_yticks(range(3),['150','100','50'])
            ax.set_xlabel('Lag (s)')
            if i==0:ax.set_ylabel(('IG attribution','Occlusion response')[row]+'\nHeight (m)')
    fig.savefig(out/'height_lag_attribution.png',dpi=220);plt.close(fig)


def figure5(z,out):
    try: src=source(z,'Figure_5_RMSE_SSIM_points.csv')
    except FileNotFoundError:
        try: src=source(z,'Figure_5_RMSE_SSIM_temporal_sampling_RECOVERED_FROM_NOTEBOOK.csv')
        except FileNotFoundError: src=source(z,'dt_sweep_metrics.csv')
    df=pd.read_csv(src)
    needed={'regime','history_s','rmse','ssim'}
    if not needed.issubset(df.columns):raise ValueError(f'Figure 5 columns required: {sorted(needed)}; got {list(df.columns)}')
    fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for regime,g in df.groupby('regime',sort=False):
        g=g.sort_values('history_s')
        axs[0].plot(g.history_s,g.rmse,'o-',label=str(regime))
        axs[1].plot(g.history_s,g.ssim,'o-',label=str(regime))
    for ax,ylabel in zip(axs,('RMSE (m s⁻¹)','SSIM')):
        ax.set_xlabel('Input history (s)');ax.set_ylabel(ylabel);ax.legend(fontsize=8)
    fig.savefig(out/'temporal_sampling.png',dpi=220);plt.close(fig)


def figure6(z,out):
    fig,axes=plt.subplots(3,5,figsize=(12,6.5),constrained_layout=True)
    for i,(key,label) in enumerate(zip(REGIMES,LABELS)):
        for j,spacing in enumerate(SPACINGS):
            if spacing==4:
                a=np.load(source(z,f'IG_ht_{key}_dt=4_samples_norm.npy'),allow_pickle=False).mean(axis=0)
            else:
                a=np.load(source(z,f'IG_ht_{key}_dt={spacing}_norm.npy'),allow_pickle=False)
                a=a/np.maximum(a.sum(),1e-12)
            ax=axes[i,j]
            ax.imshow(a,cmap='viridis',vmin=0,vmax=.4)
            ax.set_xticks(range(4),[str(spacing*n) for n in (3,2,1,0)],fontsize=8)
            ax.set_yticks(range(3),['150','100','50'],fontsize=8)
            if i==0:ax.set_title(f'Δt={spacing} s')
            if j==0:ax.set_ylabel(label+'\nHeight (m)')
    fig.savefig(out/'temporal_attribution.png',dpi=220);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',required=True,help='Figure_Source_Data.zip from the public dataset')
    p.add_argument('--output',default='figures')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    with ZipFile(args.archive) as z:
        figure3(z,out);figure5(z,out);figure6(z,out)
    print('Saved source-data plots in:',out)


if __name__=='__main__':main()
