"""STUNet experiment interface: train, evaluate, attribute, and ROI attribution."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import LESDataset, partition_indices, read_paths, fit_training_stats, REGIMES
from .models import create_model, MODEL_NAMES
from .experiment import train, evaluate, configure_seed, load_weights
from .explain import export_xai, integrated_gradients


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for command in ('train','evaluate','xai','roi'):
        q=sub.add_parser(command)
        q.add_argument('--data',required=True,help='Directory of the 12 wind-speed NPY arrays')
        q.add_argument('--regime',choices=REGIMES,required=True)
        q.add_argument('--spacing',type=int,default=4,help='Seconds between input frames')
        q.add_argument('--model',choices=MODEL_NAMES,default='stunet')
        q.add_argument('--base',type=int,default=32)
        q.add_argument('--batch-size',type=int,default=1 if command in ('xai','roi') else 8)
        q.add_argument('--crop',type=int,default=None)
        q.add_argument('--normalization',choices=('train','per_field','none'),default='train')
        q.add_argument('--workers',type=int,default=0)
        q.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
        q.add_argument('--seed',type=int,default=42)
        q.add_argument('--checkpoint',required=True)
        if command=='train':
            q.add_argument('--epochs',type=int,default=100)
            q.add_argument('--patience',type=int,default=10)
        if command in ('xai','roi'):
            q.add_argument('--output',required=True)
            q.add_argument('--steps',type=int,default=32 if command=='roi' else 16)
        if command=='xai':q.add_argument('--limit',type=int,default=None)
        if command=='roi':
            q.add_argument('--sample',type=int,default=0)
            q.add_argument('--bounds',nargs=4,type=int,required=True,metavar=('Y0','Y1','X0','X1'))
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    if a.spacing<1 or a.base<1 or a.batch_size<1:raise ValueError('Positive spacing/base/batch size required')
    configure_seed(a.seed)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device)
    input_paths,target_path=read_paths(a.data,a.regime)
    frame_count=np.load(target_path,mmap_mode='r',allow_pickle=False).shape[0]
    splits=partition_indices(frame_count,frame_spacing=a.spacing)
    stats=None
    if a.normalization=='train':
        if a.command=='train':
            stats=fit_training_stats(input_paths,splits['train'],a.spacing,crop=a.crop)
        else:
            obj=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
            metadata=obj.get('metadata',obj.get('meta',{}))
            stats=metadata.get('training_stats')
            if stats is None:raise ValueError('Checkpoint has no training_stats: select the matching normalization option')
    def loader(split,shuffle=False):
        ds=LESDataset(a.data,a.regime,splits[split],spacing=a.spacing,crop=a.crop,stats=stats)
        if a.normalization=='per_field':
            # Field-wise standardization is an explicit option for archived workflows.
            ds=FieldNormalizedDataset(ds)
        return DataLoader(ds,batch_size=a.batch_size,shuffle=shuffle,num_workers=a.workers)
    model=create_model(a.model,a.base).to(device)
    if a.command=='train':
        tr=loader('train',shuffle=True);vl=loader('val')
        config={'regime':a.regime,'model':a.model,'base':a.base,'spacing':a.spacing,
                'normalization':a.normalization,'training_stats':stats,'seed':a.seed,
                'train_n':len(tr.dataset),'val_n':len(vl.dataset),'test_n':len(splits['test'])}
        history=train(model,tr,vl,a.checkpoint,device,epochs=a.epochs,patience=a.patience,metadata=config)
        destination=Path(a.checkpoint).with_suffix('.history.csv')
        with destination.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=('epoch','train_huber','val_huber'))
            writer.writeheader();writer.writerows(history)
        print('Saved:',a.checkpoint,destination)
        return
    load_weights(model,a.checkpoint,device)
    if a.command=='evaluate':
        print(json.dumps({'n_test':len(splits['test']),**evaluate(model,loader('test'),device)},indent=2))
    elif a.command=='xai':
        if a.model!='stunet':raise ValueError('Height–lag analysis uses the full STUNet')
        n=export_xai(model,loader('test'),device,a.output,a.regime,a.spacing,steps=a.steps,limit=a.limit)
        print(f'Exported IG and occlusion for {n} sequences to {a.output}')
    else:
        if a.model!='stunet':raise ValueError('ROI attribution uses the full STUNet')
        ds=loader('test').dataset
        if not 0<=a.sample<len(ds):raise IndexError('Sample outside test set')
        x,y=ds[a.sample];x=x[None].to(device)
        ig=integrated_gradients(model,x,a.steps,roi=tuple(a.bounds))
        dest=Path(a.output);dest.mkdir(parents=True,exist_ok=True)
        np.save(dest/'roi_input.npy',x[0].cpu().numpy(),allow_pickle=False)
        np.save(dest/'roi_ig.npy',ig[0].detach().cpu().numpy(),allow_pickle=False)
        with torch.no_grad():np.save(dest/'roi_prediction.npy',model(x)[0,0].cpu().numpy(),allow_pickle=False)
        (dest/'metadata.json').write_text(json.dumps({'sample':a.sample,'roi_bounds':a.bounds,'regime':a.regime,'spacing':a.spacing},indent=2))
        print('Saved ROI source arrays:',dest)


class FieldNormalizedDataset(torch.utils.data.Dataset):
    """Spatial normalization of each input height–time field."""
    def __init__(self,dataset):self.dataset=dataset
    def __len__(self):return len(self.dataset)
    def __getitem__(self,i):
        x,y=self.dataset[i]
        mu=x.mean(dim=(-2,-1),keepdim=True)
        sd=x.std(dim=(-2,-1),keepdim=True,unbiased=False).clamp_min(1e-6)
        return (x-mu)/sd,y


if __name__=='__main__':main()
