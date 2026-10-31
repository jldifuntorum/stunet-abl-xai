"""Integrated Gradients, input occlusion, and ROI attribution."""
import numpy as np
import torch


def integrated_gradients(model,x,steps=16,roi=None):
    """Signed IG using Gauss–Legendre quadrature from a zero input baseline."""
    if steps<1:raise ValueError('Integration steps must be positive')
    model.eval()
    points,weights=np.polynomial.legendre.leggauss(steps)
    points=(points+1)/2;weights=weights/2
    running=torch.zeros_like(x)
    for alpha,weight in zip(points,weights):
        v=(x.detach()*float(alpha)).requires_grad_(True)
        prediction=model(v)
        if roi is not None:
            y0,y1,x0,x1=roi
            if not (0<=y0<y1<=prediction.shape[-2] and 0<=x0<x1<=prediction.shape[-1]):
                raise ValueError('ROI bounds outside output field')
            prediction=prediction[:,:,y0:y1,x0:x1]
        scalar=prediction.mean(dim=(1,2,3)).sum()
        gradient=torch.autograd.grad(scalar,v)[0]
        running+=float(weight)*gradient
    return x.detach()*running


def height_lag_ig(model,x,steps=16):
    a=integrated_gradients(model,x,steps=steps)
    raw=a.abs().sum(dim=(-2,-1))
    norm=raw/raw.sum(dim=(1,2),keepdim=True).clamp_min(1e-12)
    return raw.detach().cpu().numpy(),norm.detach().cpu().numpy()


@torch.no_grad()
def height_lag_occlusion(model,x):
    model.eval()
    full=model(x).mean(dim=(1,2,3))
    out=np.empty((x.size(0),x.size(1),x.size(2)),dtype='float64')
    for height in range(x.size(1)):
        for lag in range(x.size(2)):
            modified=x.clone();modified[:,height,lag]=0.
            prediction=model(modified).mean(dim=(1,2,3))
            out[:,height,lag]=(full-prediction).detach().cpu().numpy()
    return out


def export_xai(model,loader,device,out_dir,regime,spacing=4,steps=16,limit=None):
    from pathlib import Path
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    ig_raw=[];ig_norm=[];occluded=[];count=0
    for x,_ in loader:
        if limit is not None:
            if count>=limit:break
            x=x[:limit-count]
        x=x.to(device)
        raw,norm=height_lag_ig(model,x,steps)
        ig_raw.append(raw);ig_norm.append(norm)
        occluded.append(height_lag_occlusion(model,x))
        count+=len(x)
    if not ig_raw:raise ValueError('No samples to attribute')
    arrays={
        f'IG_ht_{regime}_dt={spacing}_samples_raw.npy':np.concatenate(ig_raw),
        f'IG_ht_{regime}_dt={spacing}_samples_norm.npy':np.concatenate(ig_norm),
        f'Occ_ht_{regime}_dt={spacing}_samples_mean_prediction.npy':np.concatenate(occluded),
    }
    arrays[f'IG_ht_{regime}_dt={spacing}_mean_norm.npy']=arrays[f'IG_ht_{regime}_dt={spacing}_samples_norm.npy'].mean(axis=0)
    arrays[f'Occ_ht_{regime}_dt={spacing}_mean_prediction.npy']=arrays[f'Occ_ht_{regime}_dt={spacing}_samples_mean_prediction.npy'].mean(axis=0)
    for name,values in arrays.items():np.save(out_dir/name,values,allow_pickle=False)
    return count
