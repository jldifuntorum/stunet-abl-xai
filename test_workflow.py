"""Fast, deterministic checks on data partitioning and core model interfaces."""
import numpy as np
import torch
from stunet.data import partition_indices, fit_training_stats, LESDataset
from stunet.models import create_model
from stunet.explain import integrated_gradients, height_lag_occlusion


def test_chronological_partition():
    split=partition_indices(3600,4)
    assert [len(split[k]) for k in ('train','val','test')]==[2511,538,539]
    assert split['train'][-1]<split['val'][0]<split['test'][0]
    assert split['train'][0]==12


def test_dataset_and_train_stats(tmp_path):
    rng=np.random.default_rng(4)
    for height in ('4.5','49.5','103.5','148.5'):
        np.save(tmp_path/f'neutral_wspeed_{height}.npy',rng.normal(size=(24,16,16)).astype('float32'))
    split=partition_indices(24,4)
    paths=[tmp_path/f'neutral_wspeed_{h}.npy' for h in ('148.5','103.5','49.5')]
    stats=fit_training_stats(paths,split['train'],4)
    ds=LESDataset(tmp_path,'neutral',split['test'],spacing=4,stats=stats)
    x,y=ds[0]
    assert x.shape==(3,4,16,16) and y.shape==(1,16,16)
    assert torch.isfinite(x).all()


def test_model_output_shapes():
    x=torch.randn(1,3,4,16,16)
    for name in ('stunet','stunet50','stunet50_tm','unet','tinycnn','linear'):
        model=create_model(name,base=4).eval()
        with torch.no_grad():assert model(x).shape==(1,1,16,16)


def test_explainability_shapes():
    class Basic(torch.nn.Module):
        def forward(self,x):return x.mean(dim=(1,2)).unsqueeze(1)
    model=Basic();x=torch.randn(1,3,4,8,8)
    ig=integrated_gradients(model,x,steps=4)
    assert ig.shape==x.shape
    assert torch.allclose(ig,x/(12*8*8),atol=1e-5)
    occ=height_lag_occlusion(model,x)
    assert occ.shape==(1,3,4)
