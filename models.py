"""Spatiotemporal U-Net with a ConvLSTM bottleneck and benchmark models.

The main STUNetAvg definition retains the parameter names of the archived
trained checkpoints. Model inputs have shape (batch, 3, 4, height, width).
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, k, s, p, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, k, 1, p, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, stride=2)
        self.conv = ConvBlock(cin, cout)
    def forward(self, x, skip):
        x = self.up(x)
        dh = skip.size(-2) - x.size(-2); dw = skip.size(-1) - x.size(-1)
        if dh != 0 or dw != 0: x = F.pad(x, (0,dw,0,dh))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, padding=1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(input_channels + hidden_channels, 4*hidden_channels, kernel_size, padding=padding)
        
    def forward(self, x, state):
        h, c = state
        xh = torch.cat([x, h], dim=1)
        gates = self.conv(xh)
        i,f,o,g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f); o = torch.sigmoid(o); g = torch.tanh(g)
        c = f*c + i*g
        h = o*torch.tanh(c)
        return h, c
        
    def init_state(self, b, h, w, device, dtype=torch.float32):
        return (torch.zeros(b, self.hidden_channels, h, w, device=device, dtype=dtype),
                torch.zeros(b, self.hidden_channels, h, w, device=device, dtype=dtype))


class STUNetAvg(nn.Module):
    def __init__(self, in_heights=4, T=5, base=32, ): #use_temporal_mix=True): #, T_IN=4):
        super().__init__()
        C_in = in_heights
        self.enc1 = ConvBlock(C_in, base)
        self.enc2 = ConvBlock(base, base*2)
        self.enc3 = ConvBlock(base*2, base*4)
        
        self.pool = nn.AvgPool2d(2)
        
        self.clstm = ConvLSTMCell(base*4, base*4)
        
        self.up2 = UpBlock(base*4, base*2)
        self.up1 = UpBlock(base*2, base)
        
        self.outc = nn.Conv2d(base, 1, 1)
        self.t_in = T #T_IN
    
    def encode_step(self, x_t):
        e1 = self.enc1(x_t); p1 = self.pool(e1)
        e2 = self.enc2(p1);  p2 = self.pool(e2)
        e3 = self.enc3(p2)
        return e1, e2, e3
    
    def forward(self, X):
        B,L,T,H,W = X.shape
        e1s, e2s, e3s = [], [], []
        for t in range(T):
            e1,e2,e3 = self.encode_step(X[:, :, t])
            e1s.append(e1); e2s.append(e2); e3s.append(e3)
        h,c = self.clstm.init_state(B, e3s[0].size(-2), e3s[0].size(-1), X.device, X.dtype)
        state = (h,c)

        for e3 in e3s:
            h, c = self.clstm(e3, state)   # returns two tensors
            state = (h, c)                 # keep as a 2-tuple for the next step
        
        bottleneck = h
            
        idx_last = min(T-1, max(0, self.t_in-1))
        d2 = self.up2(h, e2s[idx_last]); d1 = self.up1(d2, e1s[idx_last])
        return self.outc(d1)


def idx_for_height(heights, target_h=50.0, tol=1e-6):
    arr = np.array(heights, dtype=float)
    i = int(np.argmin(np.abs(arr - float(target_h))))
    return i


class STUNet_OneHeight(nn.Module):
    def __init__(self, L_full, T_in, heights, base=16, which_height=50.0,):
        super().__init__()
        self.L_full, self.T_in = L_full, T_in
        self.which_idx = idx_for_height(heights, which_height)
        self.core = STUNetAvg(in_heights=1, T=T_in, base=base)
    def forward(self, X):
        if X.dim()==4:
            B,C,H,W = X.shape; X = X.view(B, self.L_full, self.T_in, H, W)
        X50 = X[:, self.which_idx:self.which_idx+1]   # keep dims
        return self.core(X50)


class TemporalMix(nn.Module):
    def __init__(self, L, T):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(L, T))
        
    def forward(self, X):
        B, L, T, H, W = X.shape
        w = torch.softmax(self.alpha.to(device=X.device, dtype=X.dtype), dim=-1)
        fused = (X * w[None, :, :, None, None]).sum(dim=2)
        return fused, w


class STUNet(nn.Module):
    def __init__(self, in_heights=4, T=5, base=32, use_temporal_mix=True): #, T_IN=4):
        super().__init__()
        C_in = in_heights
        self.enc1 = ConvBlock(C_in, base)
        self.enc2 = ConvBlock(base, base*2)
        self.enc3 = ConvBlock(base*2, base*4)
        
        self.pool = nn.MaxPool2d(2)
        
        self.clstm = ConvLSTMCell(base*4, base*4)
        
        self.up2 = UpBlock(base*4, base*2)
        self.up1 = UpBlock(base*2, base)
        
        self.outc = nn.Conv2d(base, 1, 1)
        self.use_tm = use_temporal_mix
        self.temporal_mix = TemporalMix(in_heights, T) if use_temporal_mix else None
        self.t_in = T #T_IN
    
    def encode_step(self, x_t):
        e1 = self.enc1(x_t); p1 = self.pool(e1)
        e2 = self.enc2(p1);  p2 = self.pool(e2)
        e3 = self.enc3(p2)
        return e1, e2, e3
    
    def forward(self, X):
        B,L,T,H,W = X.shape
        if self.use_tm:
            fused, _ = self.temporal_mix(X)
            X = torch.cat([X, fused[:, :, None]], dim=2)
            T = T + 1
        e1s, e2s, e3s = [], [], []
        for t in range(T):
            e1,e2,e3 = self.encode_step(X[:, :, t])
            e1s.append(e1); e2s.append(e2); e3s.append(e3)
        h,c = self.clstm.init_state(B, e3s[0].size(-2), e3s[0].size(-1), X.device, X.dtype)
        state = (h,c)

        for e3 in e3s:
            h, c = self.clstm(e3, state)   # returns two tensors
            state = (h, c)                 # keep as a 2-tuple for the next step
        
        bottleneck = h
            
        idx_last = min(T-1, max(0, self.t_in-1))
        d2 = self.up2(h, e2s[idx_last]); d1 = self.up1(d2, e1s[idx_last])
        return self.outc(d1)


class STUNetOneHeightTM(nn.Module):
    """One-height temporal-mixing architecture from the neutral benchmark."""
    def __init__(self, base=32, T=4):
        super().__init__()
        self.L_full, self.T_in, self.which_idx=3,T,2
        self.core=STUNet(in_heights=1,T=T,base=base,use_temporal_mix=True)

    def forward(self, x):
        return self.core(x[:,self.which_idx:self.which_idx+1])
class Linear1x1(nn.Module):
    def __init__(self, L, T_in):
        super().__init__()
        self.L = L
        self.T_in = T_in
        self.proj = nn.Conv2d(L*T_in, 1, kernel_size=1, bias=True)

    def forward(self, X):
        X = X
        B,L,T,H,W = X.shape
        X = X.reshape(B, L*T, H, W)
        return self.proj(X)


class TinyCNN50m(nn.Module):
    def __init__(self, L, T_in, heights, which_height=50.0):
        super().__init__()
        self.L = L
        self.T_in = T_in
        self.h_idx = idx_for_height(heights, which_height)
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 1, 1)
        )
    def forward(self, X):
        X = X
        x50 = X[:, self.h_idx, -1]   # (B,H,W)
        return self.net(x50[:,None])


class UNetNoTime(nn.Module):
    def __init__(self, L, T_in, base=16):
        super().__init__()
        self.L = L
        self.T_in = T_in
        C_in = L*T_in
        def C(in_c,out_c):
            return nn.Sequential(
                nn.Conv2d(in_c,out_c,3,padding=1), nn.BatchNorm2d(out_c), nn.GELU(),
                nn.Conv2d(out_c,out_c,3,padding=1), nn.BatchNorm2d(out_c), nn.GELU()
            )
        self.enc1 = C(C_in, base)
        self.enc2 = C(base, base*2)
        self.enc3 = C(base*2, base*4)
        self.pool = nn.AvgPool2d(2)
        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = C(base*4, base*2)
        self.up1  = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = C(base*2, base)
        self.outc = nn.Conv2d(base, 1, 1)
    def forward(self, X):
        X = X
        B,L,T,H,W = X.shape
        X = X.view(B, L*T, H, W)
        e1 = self.enc1(X); p1 = self.pool(e1)
        e2 = self.enc2(p1); p2 = self.pool(e2)
        e3 = self.enc3(p2)
        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.outc(d1)


MODEL_NAMES = ('stunet', 'stunet50', 'stunet50_tm', 'unet', 'tinycnn', 'linear')


def create_model(name: str, base: int = 32) -> nn.Module:
    """Build a checkpoint-compatible architecture."""
    if name == 'stunet': return STUNetAvg(in_heights=3,T=4,base=base)
    if name == 'stunet50': return STUNet_OneHeight(L_full=3,T_in=4,heights=[150,100,50],base=base)
    if name == 'stunet50_tm': return STUNetOneHeightTM(base=base,T=4)
    if name == 'unet': return UNetNoTime(L=3,T_in=4,base=base)
    if name == 'tinycnn': return TinyCNN50m(L=3,T_in=4,heights=[150,100,50])
    if name == 'linear': return Linear1x1(L=3,T_in=4)
    raise ValueError(f'Model must be one of: {", ".join(MODEL_NAMES)}')
