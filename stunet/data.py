"""Chronological LES data partitioning and training-set normalization."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

REGIMES = ('neutral', 'transitional', 'convective')
HEIGHTS = ('148.5', '103.5', '49.5')  # model height order: 150, 100, 50 m


def read_paths(folder, regime):
    if regime not in REGIMES:
        raise ValueError(f'Regime must be one of {REGIMES}')
    folder = Path(folder)
    inputs = [folder / f'{regime}_wspeed_{height}.npy' for height in HEIGHTS]
    target = folder / f'{regime}_wspeed_4.5.npy'
    for path in inputs + [target]:
        if not path.is_file():
            raise FileNotFoundError(path)
    return inputs, target


def partition_indices(n_frames, frame_spacing=4, n_input=4, fractions=(.70, .15, .15)):
    """Split ordered, valid target times contiguously into train/val/test.

    For N=3600, frame_spacing=4, n_input=4, counts are 2511/538/539.
    """
    if n_frames <= (n_input - 1) * frame_spacing:
        raise ValueError('Insufficient frames for specified input history')
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError('Fractions must sum to one')
    valid = np.arange((n_input-1)*frame_spacing, n_frames, dtype=np.int64)
    a = int(len(valid)*fractions[0])
    b = a + int(len(valid)*fractions[1])
    return {'train': valid[:a], 'val': valid[a:b], 'test': valid[b:]}


def input_frame_indices(target_times, frame_spacing=4, n_input=4):
    target_times = np.asarray(target_times, dtype=np.int64)
    return target_times[:, None] - frame_spacing * np.arange(n_input-1, -1, -1)


def fit_training_stats(input_paths, train_times, frame_spacing, n_input=4, crop=None):
    """Estimate per-input-height mean and SD from training input frames only.

    Unique training frames are processed in small chunks to bound RAM usage.
    """
    train_frames = np.unique(input_frame_indices(train_times,frame_spacing,n_input).ravel())
    result=[]
    for path in input_paths:
        arr=np.load(path,mmap_mode='r',allow_pickle=False)
        mean=0.0;m2=0.0;n=0
        for offset in range(0,len(train_frames),8):
            v=np.asarray(arr[train_frames[offset:offset+8],:crop,:crop],dtype=np.float64)
            chunk_n=v.size
            cm=float(v.mean()); cm2=float(((v-cm)**2).sum())
            delta=cm-mean
            total=n+chunk_n
            mean+=delta*chunk_n/total
            m2+=cm2+(delta**2)*n*chunk_n/total
            n=total
        result.append((float(mean),max(float(np.sqrt(m2/n)),1e-6)))
    return result


class LESDataset(Dataset):
    """Return (3, 4, H, W) input sequences and (1, H, W) targets."""
    def __init__(self, folder, regime, times, spacing=4, crop=None, stats=None):
        input_paths, target_path=read_paths(folder,regime)
        self.fields=[np.load(p,mmap_mode='r',allow_pickle=False) for p in input_paths]
        self.target=np.load(target_path,mmap_mode='r',allow_pickle=False)
        if self.target.ndim!=3 or any(a.shape!=self.target.shape for a in self.fields):
            raise ValueError('All input and target arrays must share (time,y,x) dimensions')
        h,w=self.target.shape[-2:]
        self.crop=crop or min(h,w)
        if self.crop>min(h,w): raise ValueError('Crop exceeds horizontal dimensions')
        self.times=np.asarray(times,dtype=np.int64)
        self.spacing=int(spacing)
        if self.spacing<1:raise ValueError('Frame spacing must be >=1')
        if len(self.times) and (self.times.min()<3*self.spacing or self.times.max()>=len(self.target)):
            raise ValueError('Invalid target time indices')
        if stats is not None and len(stats)!=3:
            raise ValueError('Expected one mean/SD pair per predictor height')
        self.stats=stats
    def __len__(self):return len(self.times)
    def __getitem__(self,index):
        t=int(self.times[index])
        positions=t-self.spacing*np.arange(3,-1,-1)
        x=np.stack([np.asarray(a[positions,:self.crop,:self.crop],dtype=np.float32)
                    for a in self.fields])
        if self.stats is not None:
            for i,(mu,sd) in enumerate(self.stats):x[i]=(x[i]-mu)/sd
        y=np.asarray(self.target[t,:self.crop,:self.crop],dtype=np.float32)[None].copy()
        return torch.from_numpy(x.copy()),torch.from_numpy(y)
