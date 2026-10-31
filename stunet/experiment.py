"""STUNet training, checkpoint management and evaluation metrics."""
import json
from pathlib import Path
import numpy as np
import torch
from skimage.metrics import structural_similarity


def configure_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def train(model, train_loader, val_loader, checkpoint, device, epochs=100,
          patience=10, lr=1e-4, weight_decay=1e-5, gradient_clip=1., metadata=None):
    model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    scaler=torch.amp.GradScaler('cuda',enabled=(device.type=='cuda'))
    history=[]; best=float('inf');epochs_since_best=0
    checkpoint=Path(checkpoint);checkpoint.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,epochs+1):
        model.train();total=0.; count=0
        for x,y in train_loader:
            x=x.to(device); y=y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                loss=torch.nn.functional.smooth_l1_loss(model(x),y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),gradient_clip)
            scaler.step(optimizer);scaler.update()
            total+=float(loss.detach())*len(x);count+=len(x)
        model.eval();val_loss=0.;n_val=0
        with torch.no_grad():
            for x,y in val_loader:
                x=x.to(device);y=y.to(device)
                loss=torch.nn.functional.smooth_l1_loss(model(x),y)
                val_loss+=float(loss)*len(x); n_val+=len(x)
        train_loss=total/max(count,1);validation_loss=val_loss/max(n_val,1)
        history.append({'epoch':epoch,'train_huber':train_loss,'val_huber':validation_loss})
        print(f'{epoch:03d}  train={train_loss:.5f}  val={validation_loss:.5f}',flush=True)
        if validation_loss<best:
            best=validation_loss;epochs_since_best=0
            torch.save({'model':model.state_dict(),'metadata':metadata or {},'epoch':epoch},checkpoint)
        else:
            epochs_since_best+=1
            if epochs_since_best>=patience:break
    return history


def load_weights(model,path,device):
    """Read trusted research checkpoints using PyTorch safe tensor loading."""
    obj=torch.load(path,map_location='cpu',weights_only=True)
    state=obj.get('model',obj.get('state_dict',obj))
    model.load_state_dict(state,strict=True)
    model.to(device).eval()
    return obj.get('metadata',obj.get('meta',{}))


@torch.no_grad()
def evaluate(model,loader,device):
    model.to(device).eval();metrics=[]
    for x,y in loader:
        pred=model(x.to(device)).cpu().numpy()[:,0]
        actual=y.cpu().numpy()[:,0]
        for p,t in zip(pred,actual):
            err=p-t
            pp=p.ravel().astype(float);tt=t.ravel().astype(float)
            denom=np.linalg.norm(pp-pp.mean())*np.linalg.norm(tt-tt.mean())
            r=float(np.dot(pp-pp.mean(),tt-tt.mean())/denom) if denom>1e-12 else 0.
            metrics.append({'mae':float(np.abs(err).mean()),
                            'rmse':float(np.sqrt((err*err).mean())),
                            'r':r,
                            'ssim':float(structural_similarity(t,p,data_range=max(float(t.max()-t.min()),1e-6)))})
    if not metrics:raise ValueError('Evaluation set is empty')
    return {k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
