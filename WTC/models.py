import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from pathlib import Path
import re
import random
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import pandas as pd


# NAOMI
import math
import numpy as np
import torch
from torch import nn

class ChannelAwareNAOMI(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=64, decoder_hidden=128, highest=16,
                 rnn_layers=1, dropout=0.0, normalize=True):
        super().__init__()
        assert highest > 0 and highest & (highest - 1) == 0
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.highest = highest
        self.normalize = normalize
        self.encoder = nn.GRU(data_dim*2, hidden_dim, num_layers=rnn_layers,
                              bidirectional=True, batch_first=True,
                              dropout=dropout if rnn_layers > 1 else 0.)
        self.decoders = nn.ModuleDict()
        for step in (2**p for p in range(int(math.log2(highest))+1)):
            self.decoders[str(step)] = nn.Sequential(
                nn.Linear(hidden_dim*2 + data_dim*2 + 2, decoder_hidden),
                nn.ReLU(), nn.Linear(decoder_hidden, data_dim))
        # CPU schedule cache, not checkpoint parameters. One entry per unique mask.
        self._schedule_cache = {}

    def _norm(self, x, m):
        valid = m.bool()
        count = valid.sum(dim=1, keepdim=True)
        safe_count = count.clamp_min(1)
        observed = torch.where(valid, x, torch.zeros_like(x))
        mean = observed.sum(dim=1, keepdim=True) / safe_count
        variance = torch.where(valid, (x-mean).square(), torch.zeros_like(x)).sum(dim=1, keepdim=True) / safe_count
        sd = (variance + 1e-6).sqrt()
        empty = count == 0
        mean = torch.where(empty, torch.zeros_like(mean), mean)
        sd = torch.where(empty, torch.ones_like(sd), sd)
        if not self.normalize:
            mean = torch.zeros_like(mean)
            sd = torch.ones_like(sd)
        normalized = torch.where(valid, (x-mean)/sd, torch.zeros_like(x))
        return normalized, mean, sd

    def _make_schedule(self, mask):
        """Generate original midpoint pivots from an unchanging (T,D) CPU mask."""
        T,D = mask.shape
        available = mask.copy()
        rounds = []
        max_rounds = int(math.ceil(math.log2(T)))+2
        for _ in range(max_rounds):
            if available.all():
                break
            entries = []
            for d in range(D):
                missing = np.flatnonzero(~available[:,d])
                if missing.size == 0:
                    continue
                splits = np.flatnonzero(np.diff(missing) != 1)+1
                for run in np.split(missing, splits):
                    a,z = int(run[0]),int(run[-1])
                    if a == 0 and z == T-1:
                        l,r = 0,T-1
                    else:
                        l,r = max(0,a-1),min(T-1,z+1)
                        if a == 0: l=r
                        if z == T-1: r=l
                    pivot=(a+z)//2
                    dist=max(1,r-l)
                    step=min(self.highest, 2**min(int(math.log2(dist)),int(math.log2(self.highest))))
                    entries.append((pivot,d,l,r,step,dist/T,(pivot-l)/max(1,dist)))
                    available[pivot,d] = True
            if not entries:
                raise RuntimeError('No missing pivots generated')
            rounds.append(entries)
        if not available.all():
            raise RuntimeError('Failed to fill all gaps')
        return rounds

    def forward(self, x_in, m_hat):
        if x_in.ndim != 3 or x_in.shape != m_hat.shape or x_in.shape[-1] != self.data_dim:
            raise ValueError('Inputs must be (B,T,D) and have matching masks')
        if not torch.isfinite(x_in).all().item():
            raise ValueError('X_in must be finite and prefilled')
        x,mean,sd = self._norm(x_in,m_hat)
        B,T,D = x.shape
        device = x.device
        mask_np = m_hat.detach().to('cpu').numpy() > 0.5   # once per forward; no per-pivot GPU sync
        per_sample = []
        for b in range(B):
            key = (T,D, np.packbits(mask_np[b].reshape(-1)).tobytes())
            schedule = self._schedule_cache.get(key)
            if schedule is None:
                schedule = self._make_schedule(mask_np[b])
                self._schedule_cache[key] = schedule
            per_sample.append(schedule)
        max_rounds = max((len(s) for s in per_sample),default=0)
        work = x
        available = m_hat.bool().clone()
        for round_i in range(max_rounds):
            recurrent,_ = self.encoder(torch.cat([work, available.to(work.dtype)],dim=-1))
            fwd = recurrent[:,:,:self.hidden_dim]
            bwd = recurrent[:,:,self.hidden_dim:]
            groups = {step:[] for step in (2**p for p in range(int(math.log2(self.highest))+1))}
            for b,schedule in enumerate(per_sample):
                if round_i >= len(schedule):
                    continue
                for t,d,l,r,step,dist_fraction,pivot_fraction in schedule[round_i]:
                    groups[step].append((b,t,d,l,r,dist_fraction,pivot_fraction))
            positions_list = []
            values_list = []
            for step, entries in groups.items():
                if not entries:
                    continue
                info_np = np.asarray(entries,dtype=np.float64)
                b = torch.as_tensor(info_np[:,0],device=device,dtype=torch.long)
                t = torch.as_tensor(info_np[:,1],device=device,dtype=torch.long)
                d = torch.as_tensor(info_np[:,2],device=device,dtype=torch.long)
                l = torch.as_tensor(info_np[:,3],device=device,dtype=torch.long)
                r = torch.as_tensor(info_np[:,4],device=device,dtype=torch.long)
                positions = b*(T*D)+t*D+d
                features = torch.cat((fwd[b,l],bwd[b,r],work[b,l],work[b,r],
                                      torch.as_tensor(info_np[:,5:7],device=device,dtype=work.dtype)),dim=-1)
                values = self.decoders[str(step)](features).gather(1,d[:,None]).squeeze(1)
                positions_list.append(positions)
                values_list.append(values)
            if not positions_list:
                raise RuntimeError('No missing pivots generated')
            positions=torch.cat(positions_list)
            values=torch.cat(values_list)
            work=work.reshape(-1).scatter(0,positions,values).view(B,T,D)
            available=available.reshape(-1).scatter(0,positions,
                        torch.ones_like(positions,dtype=torch.bool)).view(B,T,D)
        x_hat=work*sd+mean
        x_c=torch.where(m_hat.bool(),x_in,x_hat)
        return {'x_hat':x_hat,'x_c':x_c,'x_norm_hat':work,'mu':mean,'sigma':sd}


def masked_target_loss(outputs,truth,observed,holdout,loss_type='mse'):
    mask=holdout.bool() & (~observed.bool()) & torch.isfinite(truth)
    if not mask.any().item():
        return None
    errors=outputs['x_c'][mask]-truth[mask]
    if loss_type=='mse': return errors.square().mean()
    if loss_type=='mae': return errors.abs().mean()
    raise ValueError(loss_type)


### SAITS

"""SAITS source-aligned, self-contained PyTorch architecture for (B, 1080, 2).
Architecture adapted from Du, Cote & Liu, SAITS, Expert Systems with
Applications 219 (2023), 119619. Official MIT-licensed PyTorch code:
https://github.com/WenjieDu/SAITS/blob/main/modeling/saits.py
https://github.com/WenjieDu/SAITS/blob/main/modeling/layers.py
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class _SAITSPositionEncoding(nn.Module):
    """Fixed sinusoidal encoding from the authors' implementation."""
    def __init__(self, d_model, seq_len):
        super().__init__()
        pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        j = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)
        phase = pos / torch.pow(10000.0, 2 * torch.floor(j / 2) / d_model)
        pe = torch.empty(seq_len, d_model)
        pe[:, 0::2] = torch.sin(phase[:, 0::2])
        pe[:, 1::2] = torch.cos(phase[:, 1::2])
        self.register_buffer('pos_table', pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        if x.size(1) > self.pos_table.size(1):
            raise ValueError('Sequence length exceeds positional encoding length')
        return x + self.pos_table[:, :x.size(1)].to(dtype=x.dtype)


class _SAITSMultiHeadAttention(nn.Module):
    """Same Q/K/V parameterization and attention math as modeling/layers.py."""
    def __init__(self, d_model, n_heads, d_k, d_v, attention_dropout=0.0,
                 fast_attention=True):
        super().__init__()
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.attention_dropout = attention_dropout
        self.fast_attention = fast_attention
        self.w_qs = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.attention_drop = nn.Dropout(attention_dropout)

    def forward(self, x, diagonal_attention_mask, return_weights=False):
        B, T, _ = x.shape
        H = self.n_heads
        q = self.w_qs(x).reshape(B, T, H, self.d_k).transpose(1, 2)
        k = self.w_ks(x).reshape(B, T, H, self.d_k).transpose(1, 2)
        v = self.w_vs(x).reshape(B, T, H, self.d_v).transpose(1, 2)

        if self.fast_attention and not return_weights:
            allowed = None
            if diagonal_attention_mask:
                allowed = ~torch.eye(T, dtype=torch.bool, device=x.device)
            z = F.scaled_dot_product_attention(
                q, k, v, attn_mask=allowed,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,
            )
            weights = None
        else:
            logits = (q / math.sqrt(self.d_k)) @ k.transpose(-2, -1)
            if diagonal_attention_mask:
                diagonal = torch.eye(T, dtype=torch.bool, device=x.device)
                logits = logits.masked_fill(
                    diagonal[None, None],
                    -1e4 if logits.dtype == torch.float16 else -1e9,
                )
            weights = self.attention_drop(torch.softmax(logits, dim=-1))
            z = weights @ v

        z = z.transpose(1, 2).contiguous().reshape(B, T, H * self.d_v)
        return self.fc(z), weights


class _SAITSFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.fc2(F.relu(self.fc1(self.norm(x)))))


class _SAITSEncoderLayer(nn.Module):
    """Official pre-LN residual attention + pre-LN residual feedforward."""
    def __init__(self, d_model, d_ff, n_heads, d_k, d_v, dropout,
                 attn_dropout, diagonal_attention_mask, fast_attention):
        super().__init__()
        self.diagonal_attention_mask = diagonal_attention_mask
        self.norm = nn.LayerNorm(d_model)  # official default eps=1e-5
        self.attn = _SAITSMultiHeadAttention(
            d_model, n_heads, d_k, d_v, attn_dropout, fast_attention
        )
        self.drop = nn.Dropout(dropout)
        self.ffn = _SAITSFeedForward(d_model, d_ff, dropout)

    def forward(self, x, return_attention=False):
        y, weights = self.attn(
            self.norm(x), self.diagonal_attention_mask,
            return_weights=return_attention,
        )
        return self.ffn(x + self.drop(y)), weights


class SAITSImputerStandard(nn.Module):
    def __init__(
        self,
        seq_len=1080, d_in=2, d_model=128, n_heads=8, d_ff=256,
        n_groups=2, n_group_inner_layers=1,
        param_sharing_strategy='inner_group', d_k=None, d_v=None,
        dropout=0.1, attn_dropout=0.0, input_with_mask=True,
        diagonal_attention_mask=True, fast_attention=True,
    ):
        super().__init__()
        if n_groups < 1 or n_group_inner_layers < 1:
            raise ValueError('Both n_groups and n_group_inner_layers must be positive')
        if param_sharing_strategy not in ('inner_group', 'between_group'):
            raise ValueError("param_sharing_strategy must be 'inner_group' or 'between_group'")
        if seq_len < 2 and diagonal_attention_mask:
            raise ValueError('Diagonal masked attention requires at least 2 time steps')
        if d_k is None:
            if d_model % n_heads:
                raise ValueError('d_model must be divisible by n_heads when d_k unspecified')
            d_k = d_model // n_heads
        if d_v is None:
            d_v = d_k
        self.seq_len, self.d_in = int(seq_len), int(d_in)
        self.n_groups = int(n_groups)
        self.n_group_inner_layers = int(n_group_inner_layers)
        self.param_sharing_strategy = param_sharing_strategy
        self.input_with_mask = bool(input_with_mask)
        dim_input = 2 * d_in if input_with_mask else d_in
        layer_kwargs = dict(
            d_model=d_model, d_ff=d_ff, n_heads=n_heads, d_k=d_k, d_v=d_v,
            dropout=dropout, attn_dropout=attn_dropout,
            diagonal_attention_mask=diagonal_attention_mask,
            fast_attention=fast_attention,
        )
        count = n_group_inner_layers if param_sharing_strategy == 'between_group' else n_groups
        self.first_block = nn.ModuleList([_SAITSEncoderLayer(**layer_kwargs) for _ in range(count)])
        self.second_block = nn.ModuleList([_SAITSEncoderLayer(**layer_kwargs) for _ in range(count)])
        self.dropout = nn.Dropout(dropout)
        self.position_enc = _SAITSPositionEncoding(d_model, seq_len)
        self.embedding_1 = nn.Linear(dim_input, d_model)
        self.embedding_2 = nn.Linear(dim_input, d_model)
        self.reduce_dim_z = nn.Linear(d_model, d_in)
        self.reduce_dim_beta = nn.Linear(d_model, d_in)
        self.reduce_dim_gamma = nn.Linear(d_in, d_in)
        self.weight_combine = nn.Linear(d_in + seq_len, d_in)

    def _encode_block(self, x, layers, return_last_attention):
        # Both loops reproduce parameter-sharing semantics in official code.
        last_weights = None
        if self.param_sharing_strategy == 'between_group':
            for grp in range(self.n_groups):
                for j, layer in enumerate(layers):
                    last = (grp == self.n_groups - 1 and j == len(layers) - 1)
                    x, weights = layer(x, return_attention=return_last_attention and last)
                    if return_last_attention and last:
                        last_weights = weights
        else:
            for i, layer in enumerate(layers):
                for inner in range(self.n_group_inner_layers):
                    last = (i == len(layers) - 1 and inner == self.n_group_inner_layers - 1)
                    x, weights = layer(x, return_attention=return_last_attention and last)
                    if return_last_attention and last:
                        last_weights = weights
        return x, last_weights

    def forward(self, x_in, m_hat):
        if x_in.ndim != 3 or x_in.shape[1:] != (self.seq_len, self.d_in):
            raise ValueError(f'Expected (B,{self.seq_len},{self.d_in}), got {tuple(x_in.shape)}')
        if m_hat.shape != x_in.shape:
            raise ValueError('m_hat and x_in must have same shape')
        # Robust handling for NaNs in absent inputs, without using hidden targets.
        mask = ((m_hat > 0.5) & torch.isfinite(x_in)).to(x_in.dtype)
        observed_x = torch.where(mask.bool(), x_in, torch.zeros_like(x_in))
        first_input = torch.cat((observed_x, mask), dim=-1) if self.input_with_mask else observed_x
        z = self.dropout(self.position_enc(self.embedding_1(first_input)))
        z, _ = self._encode_block(z, self.first_block, return_last_attention=False)
        x_tilde_1 = self.reduce_dim_z(z)
        x_prime = torch.where(mask.bool(), observed_x, x_tilde_1)

        second_input = torch.cat((x_prime, mask), dim=-1) if self.input_with_mask else x_prime
        y = self.position_enc(self.embedding_2(second_input))
        y, attn = self._encode_block(y, self.second_block, return_last_attention=True)
        x_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(y)))

        # The official code averages last-layer attention over the head axis.
        attn_mean = attn.mean(dim=1)  # (B, T, T)
        eta = torch.sigmoid(self.weight_combine(torch.cat((mask, attn_mean), dim=-1)))
        x_tilde_3 = (1.0 - eta) * x_tilde_2 + eta * x_tilde_1
        x_c = torch.where(mask.bool(), observed_x, x_tilde_3)
        return x_tilde_1, x_tilde_2, x_tilde_3, x_c, attn_mean


def masked_mae(estimation, target, mask, eps=1e-9):
    if estimation.shape != target.shape or estimation.shape != mask.shape:
        raise ValueError('estimation, target and mask must have identical shapes')
    valid = (mask > 0.5) & torch.isfinite(target)
    target_safe = torch.where(valid, target, torch.zeros_like(target))
    errors = (estimation - target_safe).abs()
    errors = torch.where(valid, errors, torch.zeros_like(errors))
    return errors.sum() / valid.sum().clamp_min(1).to(errors.dtype)


def saits_losses_standard(x_tilde_1, x_tilde_2, x_tilde_3, x_c,
                          x_true, m_hat, i_mask, lam_ort=1.0, lam_mit=1.0):
    """Paper's ORT/MIT objective; no test ground truth is passed to the model.

    ORT: average MAE across all 3 reconstructions at observed input positions.
    MIT: MAE of final reconstruction x_tilde_3 at artificially hidden positions.
    M_hat is 1 for observed in x_in; I_mask is 1 for artificially withheld GT.
    Ground truth at genuine unavailable entries may be NaN.
    """
    if torch.any((m_hat > 0.5) & (i_mask > 0.5)):
        raise ValueError('m_hat and i_mask must be disjoint')
    observed = (m_hat > 0.5) & (i_mask <= 0.5)
    heldout = (i_mask > 0.5) & (m_hat <= 0.5)
    ort_1 = masked_mae(x_tilde_1, x_true, observed)
    ort_2 = masked_mae(x_tilde_2, x_true, observed)
    ort_3 = masked_mae(x_tilde_3, x_true, observed)
    ort = (ort_1 + ort_2 + ort_3) / 3.0
    mit = masked_mae(x_tilde_3, x_true, heldout)
    total = lam_ort * ort + lam_mit * mit
    return total, ort, mit



"""GP-VAE PyTorch port adapted to MAP/ScO2 (B,1080,2).
Source: https://github.com/ratschlab/GP-VAE (Fortuin et al., AISTATS 2020)
References: lib/models.py (JointEncoder / BandedJointEncoder, GaussianDecoder,
GP_VAE / HI_VAE), lib/nn_utils.py, lib/gp_kernel.py, train.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPVAEEncoder(nn.Module):
    """Official JointEncoder / BandedJointEncoder style."""
    def __init__(self, data_dim=2, latent_dim=35, hidden_sizes=(128, 128),
                 window_size=24, posterior='diagonal', data_type='physionet',
                 **_unused):
        super().__init__()
        if not hidden_sizes:
            raise ValueError('hidden_sizes must contain at least one value')
        if posterior not in ('diagonal', 'banded'):
            raise ValueError("posterior must be 'diagonal' or 'banded'")
        self.latent_dim = int(latent_dim)
        self.posterior = posterior
        self.data_type = data_type
        self.conv = nn.Conv1d(data_dim, hidden_sizes[0], kernel_size=window_size,
                              padding='same')
        dims = list(hidden_sizes)
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.to_params = nn.Linear(dims[-1], (3 if posterior == 'banded' else 2)*latent_dim)

    def forward(self, observed_normalized, m_hat=None):
        # Original network receives missing values as zeros, not a concatenated mask.
        h = self.conv(observed_normalized.transpose(1, 2)).transpose(1, 2)
        for layer in self.hidden:
            h = F.relu(layer(h))
        params = self.to_params(h)
        mu = params[..., :self.latent_dim]   # B,T,Z
        if self.posterior == 'diagonal':
            sd = F.softplus(params[..., self.latent_dim:]) + 1e-5
            return mu, {'std': sd, 'logvar': 2.0*torch.log(sd)}
        # Published banded encoder produces one positive diagonal and one
        # positive superdiagonal of an upper precision Cholesky factor.
        raw = params[..., self.latent_dim:]
        transform = torch.sigmoid if self.data_type == 'physionet' else F.softplus
        diag = 1.0 + transform(raw[..., :self.latent_dim])
        upper = transform(raw[..., self.latent_dim:])[:, :-1, :]
        return mu, {'diag': diag, 'upper': upper}


class GPVAEDecoder(nn.Module):
    def __init__(self, latent_dim=35, data_dim=2, hidden_sizes=(256, 256), **_unused):
        super().__init__()
        dims = [latent_dim] + list(hidden_sizes) + [data_dim]
        layers = []
        for a, b in zip(dims[:-2], dims[1:-1]):
            layers.extend([nn.Linear(a, b), nn.ReLU()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


def _cauchy_prior_matrix(T, sigma, length_scale, kernel, jitter):
    t = torch.arange(T, dtype=torch.float64)
    squared_dist = (t[:, None]-t[None, :]).square()
    if kernel == 'cauchy':
        k = sigma / (1.0 + squared_dist/(length_scale**2))
    elif kernel == 'rbf':
        k = torch.exp(-squared_dist/(length_scale**2))
    elif kernel == 'matern':
        k = torch.exp(-torch.sqrt(squared_dist)/math.sqrt(length_scale))
    elif kernel == 'diffusion':
        if not 0 < length_scale < 0.5:
            raise ValueError('diffusion kernel requires 0 < length_scale < 0.5')
        k = torch.eye(T, dtype=torch.float64)*(1.0-length_scale)
        k += torch.diag(torch.full((T-1,), length_scale, dtype=torch.float64), 1)
        k += torch.diag(torch.full((T-1,), length_scale, dtype=torch.float64), -1)
    else:
        raise ValueError(f'Unknown kernel: {kernel}')
    if kernel == 'cauchy':
        k += torch.eye(T, dtype=torch.float64)*jitter
    else:
        k += torch.eye(T, dtype=torch.float64)*jitter
    return k


class GPVAEImputer(nn.Module):
    def __init__(self, seq_len=1080, data_dim=2, latent_dim=35,
                 encoder_sizes=(128, 128), decoder_sizes=(256, 256),
                 window_size=24, kernel='cauchy', gp_length_scale=7.0,
                 gp_sigma=1.005, gp_jitter=0.001, kernel_scales=1,
                 beta=0.2, posterior='banded', data_type='physionet',
                 normalization='observed', eps=1e-6,
                 # Compatibility with earlier constructor:
                 hidden_dim=None, dropout=None, **kwargs):
        super().__init__()
        if kwargs:
            raise TypeError('Unexpected GPVAEImputer arguments: '+str(sorted(kwargs)))
        if hidden_dim is not None:
            encoder_sizes = (hidden_dim, hidden_dim)
            decoder_sizes = (hidden_dim, hidden_dim)
        self.seq_len = int(seq_len)
        self.data_dim = int(data_dim)
        self.latent_dim = int(latent_dim)
        self.posterior = posterior
        self.beta = float(beta)
        self.normalization = normalization
        self.eps = float(eps)
        if normalization not in ('observed', 'fold', 'none'):
            raise ValueError("normalization must be 'observed', 'fold', or 'none'")
        if not 1 <= kernel_scales <= latent_dim:
            raise ValueError('kernel_scales must be in [1,latent_dim]')
        self.encoder = GPVAEEncoder(data_dim, latent_dim, encoder_sizes, window_size,
                                     posterior=posterior, data_type=data_type)
        self.decoder = GPVAEDecoder(latent_dim, data_dim, decoder_sizes)
        self.register_buffer('fold_mu', torch.zeros(1, 1, data_dim))
        self.register_buffer('fold_sigma', torch.ones(1, 1, data_dim))
        self.register_buffer('fold_stats_ready', torch.tensor(False))

        # Store only one precision matrix per DISTINCT GP kernel scale.
        # Compute in float64 for reliable Cholesky; use stored float32 at runtime.
        precisions, diagonals, logdets = [], [], []
        for g in range(kernel_scales):
            length = gp_length_scale / 2**g
            K = _cauchy_prior_matrix(seq_len, gp_sigma, length, kernel, gp_jitter)
            L = torch.linalg.cholesky(K)
            inv = torch.cholesky_inverse(L)
            precisions.append(inv.float())
            diagonals.append(torch.diagonal(inv).float())
            logdets.append((2*torch.log(torch.diagonal(L)).sum()).float())
        self.register_buffer('gp_precision', torch.stack(precisions), persistent=False)
        self.register_buffer('gp_precision_diag', torch.stack(diagonals), persistent=False)
        self.register_buffer('gp_logdet', torch.stack(logdets), persistent=False)
        self.scale_ids = []
        n_assigned = 0
        for g in range(kernel_scales):
            n = latent_dim-n_assigned if g == kernel_scales-1 else math.ceil(latent_dim/kernel_scales)
            self.scale_ids.extend([g]*n)
            n_assigned += n
        assert len(self.scale_ids) == latent_dim

    @torch.no_grad()
    def set_standardization(self, mean, std):
        mean = torch.as_tensor(mean, dtype=self.fold_mu.dtype, device=self.fold_mu.device)
        std = torch.as_tensor(std, dtype=self.fold_sigma.dtype, device=self.fold_sigma.device)
        self.fold_mu.copy_(mean.reshape(1, 1, self.data_dim))
        self.fold_sigma.copy_(std.reshape(1, 1, self.data_dim).clamp_min(self.eps))
        self.fold_stats_ready.fill_(True)

    def observed_normalize(self, x_in, m_hat):
        valid = (m_hat > 0.5) & torch.isfinite(x_in)
        observed = torch.where(valid, x_in, torch.zeros_like(x_in))
        if self.normalization == 'observed':
            n = valid.sum(dim=1, keepdim=True).clamp_min(1)
            mu = observed.sum(1, keepdim=True)/n
            sigma = (((observed-mu).square()*valid).sum(1, keepdim=True)/n+self.eps).sqrt()
            mu = torch.where(n > 0, mu, torch.zeros_like(mu))
        elif self.normalization == 'fold':
            if not bool(self.fold_stats_ready):
                raise RuntimeError("Call model.set_standardization(train_mean,train_std) for normalization='fold'.")
            mu, sigma = self.fold_mu, self.fold_sigma
        else:
            mu = torch.zeros(1, 1, self.data_dim, device=x_in.device, dtype=x_in.dtype)
            sigma = torch.ones_like(mu)
        xnorm = torch.where(valid, (observed-mu)/sigma, torch.zeros_like(observed))
        return xnorm, mu, sigma, valid

    def _kl_diagonal(self, z_mu, logvar):
        # Analytic KL(N(mu,diag(var)) || N(0,K_GP)) for all (B,Z,T).
        z = z_mu.transpose(1, 2)
        v = logvar.exp().transpose(1, 2)
        logv = logvar.transpose(1, 2)
        B,Z,T = z.shape
        kl_per_batch = z.new_zeros(B)
        for g in set(self.scale_ids):
            indices = [i for i, v_ in enumerate(self.scale_ids) if v_ == g]
            mu_g, var_g, lv_g = z[:, indices], v[:, indices], logv[:, indices]
            p = self.gp_precision[g]
            quad = ((mu_g @ p)*mu_g).sum((-1,-2))
            trace = (var_g * self.gp_precision_diag[g]).sum((-1,-2))
            logdet = self.gp_logdet[g] * len(indices)
            kl_per_batch += 0.5*(quad+trace - len(indices)*T + logdet - lv_g.sum((-1,-2)))
        return kl_per_batch.mean()

    def _draw_banded(self, z_mu, diag, upper):
        eps = torch.randn_like(z_mu)
        values = [None]*self.seq_len
        values[0] = eps[:, 0, :]/diag[:, 0, :]
        for t in range(1, self.seq_len):
            values[t] = (eps[:, t, :] - upper[:, t-1, :]*values[t-1])/diag[:, t, :]
        z = z_mu + torch.stack(values, dim=1)
        return z, eps

    def _kl_banded_mc(self, z_mu, diag, upper, z, eps):
        # Unbiased MC estimator of KL for the ORIGINAL banded posterior.
        # Unlike author's analytic KL it has stochastic variance.
        B,T,Z = z_mu.shape
        logq = (-0.5*eps.square().sum((1,2)) + torch.log(diag).sum((1,2))
                -0.5*T*Z*math.log(2*math.pi))
        z_p = z.transpose(1, 2)
        logp = z.new_zeros(B)
        for g in set(self.scale_ids):
            indices = [i for i, v_ in enumerate(self.scale_ids) if v_ == g]
            z_g = z_p[:, indices]
            quadratic = ((z_g @ self.gp_precision[g])*z_g).sum((-1,-2))
            logp += -0.5*(quadratic + len(indices)*self.gp_logdet[g]
                          +len(indices)*T*math.log(2*math.pi))
        return (logq - logp).mean()

    def forward(self, x_in, m_hat, sample=None):
        if x_in.ndim != 3 or x_in.shape[1:] != (self.seq_len, self.data_dim):
            raise ValueError(f'Expected (B,{self.seq_len},{self.data_dim}), got {tuple(x_in.shape)}')
        if m_hat.shape != x_in.shape:
            raise ValueError('x_in and m_hat shapes do not match')
        # Use FP32 for the KL: dense GP inverses + autocast half can lose precision.
        with torch.autocast(device_type=x_in.device.type, enabled=False):
            x_norm, mu, sigma, obs = self.observed_normalize(x_in.float(), m_hat.float())
            z_mu, posterior_params = self.encoder(x_norm)
            sample = self.training if sample is None else bool(sample)
            if self.posterior == 'diagonal':
                z_logvar = posterior_params['logvar']
                z_sample = z_mu + torch.randn_like(z_mu)*posterior_params['std'] if sample else z_mu
                kl = self._kl_diagonal(z_mu, z_logvar)
            else:
                z_draw, eps = self._draw_banded(z_mu, posterior_params['diag'], posterior_params['upper'])
                z_sample = z_draw if sample else z_mu
                kl = self._kl_banded_mc(z_mu, posterior_params['diag'], posterior_params['upper'], z_draw, eps)
                diag, upper = posterior_params['diag'], posterior_params['upper']
                variances = [None]*self.seq_len
                variances[0] = diag[:, 0, :].square().reciprocal()
                for t in range(1, self.seq_len):
                    variances[t] = (1.0 + upper[:, t-1, :].square()*variances[t-1]) / diag[:, t, :].square()
                z_logvar = torch.stack(variances, dim=1).clamp_min(1e-30).log()
            x_norm_hat = self.decoder(z_sample)
            x_hat = x_norm_hat*sigma + mu
            x_c = torch.where(obs, x_in.float(), x_hat)
            return {'x_hat': x_hat, 'x_c': x_c, 'x_norm_hat': x_norm_hat,
                    'z_mu': z_mu, 'z_logvar': z_logvar, 'kl': kl,
                    'mu': mu, 'sigma': sigma, 'x_norm_in': x_norm,
                    'observed_mask': obs, 'beta': self.beta}


def gpvae_loss_same_I_mask(outputs, x_true, m_hat, I_mask,
                           lam_mit=None, beta_kl=None, loss_type=None):
    if loss_type is not None and str(loss_type).lower() not in ('nll', 'elbo', 'gaussian'):
        raise ValueError("GP-VAE original objective is Gaussian NLL, not loss_type='mae'/'mse'. Remove loss_type.")
    if lam_mit is not None:
        raise ValueError('Original GP-VAE does not use lam_mit; remove it from the loss call.')
    obs = outputs['observed_mask']
    x0 = outputs['x_norm_in']
    phat = outputs['x_norm_hat']
    # pyTorch Normal(loc,1).log_prob(x) = -.5*(x-loc)^2 -.5log(2pi)
    per_entry_nll = 0.5*((phat-x0).square() + math.log(2.0*math.pi))
    nll = torch.where(obs, per_entry_nll, torch.zeros_like(per_entry_nll)).sum((1,2)).mean()
    kl = outputs['kl']
    beta = float(outputs['beta'] if beta_kl is None else beta_kl)
    total = nll + beta*kl

    finite_gt = torch.isfinite(x_true)
    eval_mask = (I_mask > 0.5) & (~obs) & finite_gt
    safe_truth = torch.where(eval_mask, x_true, torch.zeros_like(x_true))
    safe_pred = torch.where(eval_mask, outputs['x_hat'], torch.zeros_like(outputs['x_hat']))
    mit_mae = (safe_pred-safe_truth).abs().sum()/eval_mask.sum().clamp_min(1)
    logs = {'loss': total.detach(), 'nll': nll.detach(), 'kl_loss': kl.detach(),
            'ort_loss': nll.detach(), 'mit_loss': mit_mae.detach(),
            'n_ort': obs.sum().detach(), 'n_mit': eval_mask.sum().detach(),
            'beta': torch.tensor(beta, device=total.device)}
    return total, logs


# MTSIT
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_timewise_mean_std(x: torch.Tensor, m_hat: torch.Tensor, eps: float = 1e-5):
    """
    Compute mean/std for each sample and variable across time only.

    x     : (B, T, D)
    m_hat : (B, T, D), 1=observed, 0=missing

    Returns:
      mu    : (B, 1, D)
      sigma : (B, 1, D)
    """
    m = m_hat.to(dtype=x.dtype)

    count = m.sum(dim=1, keepdim=True).clamp_min(1.0)          # (B,1,D)
    mu = (x * m).sum(dim=1, keepdim=True) / count              # (B,1,D)

    var = (((x - mu) * m) ** 2).sum(dim=1, keepdim=True) / count
    sigma = torch.sqrt(var + eps)
    return mu, sigma


class LearnablePositionalEncoding(nn.Module):
    """
    Fully learnable positional encoding E_pos in R^(T x L)
    """
    def __init__(self, seq_len: int, d_model: int, init_std: float = 0.02):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pe, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TimeBatchNorm(nn.Module):
    """
    BatchNorm over latent dimension L for sequences shaped (B,T,L).
    Implemented via BatchNorm1d(L) on permuted tensor (B,L,T).
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,L)
        x = x.transpose(1, 2)      # (B,L,T)
        x = self.bn(x)
        x = x.transpose(1, 2)      # (B,T,L)
        return x

class MTSITEncoderBlock(nn.Module):
    """
    Paper-style encoder block:
      X_bn  = BN(X + MHA(X))
      X_ffn = Dropout( GELU( X_bn W_h ) W_o )
      Y     = BN(X_bn + X_ffn)
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.bn1 = TimeBatchNorm(d_model)
        self.bn2 = TimeBatchNorm(d_model)

        self.ffn_fc1 = nn.Linear(d_model, d_ff)
        self.ffn_fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: (B,T,L)

        # MHA + residual + BN
        attn_out, attn_weights = self.mha(x, x, x, need_weights=True)
        x_bn = self.bn1(x + attn_out)

        # FFN(GELU) + residual + BN
        x_ffn = self.ffn_fc2(F.gelu(self.ffn_fc1(x_bn)))
        x_ffn = self.dropout(x_ffn)

        y = self.bn2(x_bn + x_ffn)
        return y, attn_weights


class MTSITImputerOriginalAligned(nn.Module):
    """
    Inputs:
      x_in  : (B,T,D)   missing values already prefilled (e.g. 0)
      m_hat : (B,T,D)   1=observed, 0=missing

    Outputs:
      x_hat      : (B,T,D)  reconstructed / predicted sequence
      x_c        : (B,T,D)  complemented imputed series
      x_norm_hat : (B,T,D)  predicted normalized output before de-normalization
      mu, sigma  : (B,1,D)
    """
    def __init__(
        self,
        seq_len: int = 1080,
        d_in: int = 2,
        d_model: int = 128,
        n_heads: int = 8,
        d_ff: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_in = d_in
        self.d_model = d_model

        # Eq. (1): X_p = X_norm W_p + B_p + E_pos
        self.in_proj = nn.Linear(d_in, d_model, bias=False)   # W_p
        self.input_bias = nn.Parameter(torch.zeros(1, seq_len, d_model))  # B_p
        nn.init.normal_(self.input_bias, mean=0.0, std=0.02)

        self.pos = LearnablePositionalEncoding(seq_len=seq_len, d_model=d_model)

        self.encoder_blocks = nn.ModuleList([
            MTSITEncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Eq. (10): X_hat = Z W_o + B_o
        self.out_proj = nn.Linear(d_model, d_in, bias=True)

    def normalize_inputs(self, x_in: torch.Tensor, m_hat: torch.Tensor, eps: float = 1e-5):
        """
        Per-sample, per-variable normalization across sequence length.
        Uses only observed values to estimate mean/std.
        """
        mu, sigma = masked_timewise_mean_std(x_in, m_hat, eps=eps)
        x_norm = (x_in - mu) / sigma

        # for missing positions, keep the provided filled value pathway stable
        # after normalization; simplest is zero them out
        x_norm = m_hat * x_norm
        return x_norm, mu, sigma

    def forward(self, x_in: torch.Tensor, m_hat: torch.Tensor, return_attn: bool = False):
        B, T, D = x_in.shape
        assert T == self.seq_len and D == self.d_in, \
            f"Expected (B,{self.seq_len},{self.d_in}), got {tuple(x_in.shape)}"
        assert m_hat.shape == x_in.shape, "m_hat must have same shape as x_in"

        # Normalize across time for each variable
        x_norm, mu, sigma = self.normalize_inputs(x_in, m_hat)

        # Input projection + learnable bias + learnable positional encoding
        h = self.in_proj(x_norm) + self.input_bias[:, :T, :]
        h = self.pos(h)

        attn_list = []
        for block in self.encoder_blocks:
            h, attn_w = block(h)
            if return_attn:
                attn_list.append(attn_w)

        # Linear decoder in normalized space
        x_norm_hat = self.out_proj(h)                  # (B,T,D)

        # De-normalize back to original scale
        x_hat = x_norm_hat * sigma + mu                # (B,T,D)

        # Complement observed values with predictions at missing positions
        x_c = m_hat * x_in + (1.0 - m_hat) * x_hat

        if return_attn:
            return x_hat, x_c, x_norm_hat, mu, sigma, attn_list
        return x_hat, x_c, x_norm_hat, mu, sigma



# CSDI 
import numpy as np
import torch
from torch import nn
from diff_models import diff_CSDI

class OfficialCSDIImputer(nn.Module):
    def __init__(self, seq_len=1080, d_in=2, channels=64, n_layers=4,
                 n_heads=8, diffusion_steps=50, beta_start=0.0001,
                 beta_end=0.5, schedule='quad', timeemb=128, featureemb=16,
                 diffusion_embedding_dim=128, normalize=True, dt=5.0):
        super().__init__()
        if schedule not in ('quad', 'linear'):
            raise ValueError('schedule must be quad or linear')
        self.seq_len, self.d_in, self.normalize, self.dt = seq_len, d_in, normalize, dt
        self.num_steps = diffusion_steps
        self.timeemb, self.featureemb = timeemb, featureemb
        self.feature_embedding = nn.Embedding(d_in, featureemb)
        cfg = dict(channels=channels, layers=n_layers, nheads=n_heads,
                   diffusion_embedding_dim=diffusion_embedding_dim,
                   num_steps=diffusion_steps,
                   side_dim=timeemb + featureemb + 1, is_linear=False)
        # Exact original network definition from ermongroup/CSDI/diff_models.py
        self.diffmodel = diff_CSDI(cfg, inputdim=2)
        if schedule == 'quad':
            betas = np.linspace(np.sqrt(beta_start), np.sqrt(beta_end), diffusion_steps)**2
        else:
            betas = np.linspace(beta_start, beta_end, diffusion_steps)
        betas = torch.tensor(betas, dtype=torch.float32)
        alpha_hat = 1 - betas
        alpha = alpha_hat.cumprod(0)
        self.register_buffer('beta', betas)
        self.register_buffer('alpha_hat', alpha_hat)
        self.register_buffer('alpha', alpha)

    def _check(self, x, mask):
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.d_in):
            raise ValueError(f'Expected (B,{self.seq_len},{self.d_in}), got {tuple(x.shape)}')
        if mask.shape != x.shape:
            raise ValueError('mask shape must equal data shape')

    def _normalize(self, x, mask):
        mask = mask.float()
        if not self.normalize:
            mu = torch.zeros(x.size(0), 1, x.size(2), dtype=x.dtype, device=x.device)
            sigma = torch.ones_like(mu)
        else:
            count = mask.sum(dim=1, keepdim=True).clamp_min(1)
            mu = (x * mask).sum(dim=1, keepdim=True) / count
            var = (((x - mu) * mask)**2).sum(dim=1, keepdim=True) / count
            sigma = (var + 1e-5).sqrt()
        return mu, sigma

    def _side_info(self, cond_mask, times=None):
        # cond_mask: (B,K,L) ; original CSDI Fourier time and learned feature embeddings
        B, K, L = cond_mask.shape
        if times is None:
            times = torch.arange(L, device=cond_mask.device, dtype=torch.float32)[None].expand(B,-1)
        if times.shape != (B,L):
            raise ValueError('times must have shape (B,T)')
        times = times.to(device=cond_mask.device, dtype=torch.float32)
        frequencies = torch.exp(-torch.log(torch.tensor(10000., device=times.device)) *
                                torch.arange(0,self.timeemb,2,device=times.device)/self.timeemb)
        phases = times.unsqueeze(-1)*frequencies
        time_embedding = torch.zeros(B,L,self.timeemb,device=times.device)
        time_embedding[...,0::2] = phases.sin()
        time_embedding[...,1::2] = phases.cos()
        te = time_embedding.unsqueeze(2).expand(B,L,K,self.timeemb)
        fe = self.feature_embedding(torch.arange(K,device=times.device))
        fe = fe.view(1,1,K,self.featureemb).expand(B,L,K,self.featureemb)
        side = torch.cat((te,fe),dim=-1).permute(0,3,2,1)
        return torch.cat((side, cond_mask.unsqueeze(1)),dim=1)

    @staticmethod
    def _network_input(noisy, clean, cond_mask):
        return torch.stack((cond_mask*clean, (1-cond_mask)*noisy),dim=1)

    def training_loss(self, x_true, m_hat, i_mask, times=None, fixed_step=None):
        """Fixed artificial holdout training. m_hat=visible, i_mask=hidden with GT.
        """
        self._check(x_true, m_hat)
        self._check(x_true, i_mask)
        mask = m_hat.float()
        target = i_mask.float()*(1-mask)
        if (mask*target).any():
            raise ValueError('overlapping conditioning and target positions')
        if target.sum() == 0:
            raise ValueError('no supervised target values')
        mu,sigma = self._normalize(x_true*mask,mask)
        normalized = (x_true-mu)/sigma
        normalized = torch.where((mask+target)>0, normalized, 0.)
        clean = normalized.permute(0,2,1)
        cm = mask.permute(0,2,1)
        tm = target.permute(0,2,1)
        B = x_true.size(0)
        if fixed_step is None:
            step = torch.randint(self.num_steps,(B,),device=x_true.device)
        else:
            step = torch.full((B,),int(fixed_step),device=x_true.device,dtype=torch.long)
        noise = torch.randn_like(clean)
        abar = self.alpha[step].view(B,1,1)
        noisy = abar.sqrt()*clean + (1-abar).sqrt()*noise
        side = self._side_info(cm,times)
        predicted = self.diffmodel(self._network_input(noisy,clean,cm),side,step)
        return (((predicted-noise)*tm)**2).sum()/tm.sum().clamp_min(1.)

    @torch.no_grad()
    def sample(self, x_in, m_hat, n_samples=1, times=None):
        self._check(x_in, m_hat)
        if n_samples < 1:
            raise ValueError("n_samples >= 1 required")

        mask = m_hat.float()
        mu, sigma = self._normalize(x_in * mask, mask)

        # clean, cm, side are the same as before
        clean = (((x_in - mu) / sigma) * mask).permute(0, 2, 1)   # (B, K, L)
        cm = mask.permute(0, 2, 1)                                # (B, K, L)
        side = self._side_info(cm, times)                         # (B, side_dim, K, L)

        B, K, L = clean.shape
        T = x_in.shape[1]
        D = x_in.shape[2]

        clean_rep = clean.repeat(n_samples, 1, 1)                 # (n_samples*B, K, L)
        cm_rep = cm.repeat(n_samples, 1, 1)                       # (n_samples*B, K, L)
        side_rep = side.repeat(n_samples, 1, 1, 1)                # (n_samples*B, side_dim, K, L)

        x_in_rep = x_in.repeat(n_samples, 1, 1)                   # (n_samples*B, T, D)
        mask_rep = mask.repeat(n_samples, 1, 1)                   # (n_samples*B, T, D)
        mu_rep = mu.repeat(n_samples, 1, 1)                       # (n_samples*B, 1, D)
        sigma_rep = sigma.repeat(n_samples, 1, 1)                 # (n_samples*B, 1, D)

        current = torch.randn_like(clean_rep)                     # (n_samples*B, K, L)

        for t in range(self.num_steps - 1, -1, -1):
            step = torch.full(
                (n_samples * B,),
                t,
                device=x_in.device,
                dtype=torch.long
            )

            eps_hat = self.diffmodel(
                self._network_input(current, clean_rep, cm_rep),
                side_rep,
                step
            )

            current = (
                current
                - (1 - self.alpha_hat[t]) / torch.sqrt(1 - self.alpha[t]) * eps_hat
            ) / torch.sqrt(self.alpha_hat[t])

            if t > 0:
                sigma_t = torch.sqrt(
                    (1 - self.alpha[t - 1]) / (1 - self.alpha[t]) * self.beta[t]
                )
                current = current + sigma_t * torch.randn_like(current)

        # Back to original physical units
        sample = current.permute(0, 2, 1) * sigma_rep + mu_rep    # (n_samples*B, T, D)

        # Keep observed values unchanged
        completed = mask_rep * x_in_rep + (1 - mask_rep) * sample # (n_samples*B, T, D)

        # Restore output shape: (n_samples, B, T, D)
        return completed.reshape(n_samples, B, T, D)

    @torch.no_grad()
    def forward(self, x_in, m_hat, n_samples=1, times=None):
        """MTSIT-compatible 5 outputs; n_samples>1 uses pointwise median."""
        samples = self.sample(x_in,m_hat,n_samples=n_samples,times=times)
        x_c = samples.median(dim=0).values if n_samples>1 else samples[0]
        mask = m_hat.float()
        mu,sigma = self._normalize(x_in*mask,mask)
        x_hat = x_c  # at observed positions x_c retains observed values
        x_norm_hat = (x_hat-mu)/sigma
        return x_hat,x_c,x_norm_hat,mu,sigma



"""TransMIT: PyTorch port of the authors' TensorFlow/Keras dual-attention model.
Reference: https://github.com/SJoeQin/TransMIT (TransMIT.py, utils.py, run.py)
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class _KerasStyleMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, d_q=64, dropout=0.1):
        super().__init__()
        hq = int(num_heads) * int(d_q)
        self.n_heads = int(num_heads)
        self.d_q = int(d_q)
        self.query = nn.Linear(embed_dim, hq)
        self.key = nn.Linear(embed_dim, hq)
        self.value = nn.Linear(embed_dim, hq)
        self.output = nn.Linear(hq, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, _ = x.shape
        q = self.query(x).reshape(b, n, self.n_heads, self.d_q).transpose(1, 2)
        k = self.key(x).reshape(b, n, self.n_heads, self.d_q).transpose(1, 2)
        v = self.value(x).reshape(b, n, self.n_heads, self.d_q).transpose(1, 2)
        weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_q)
        weights = self.attn_dropout(weights.softmax(dim=-1))
        values = weights @ v
        values = values.transpose(1, 2).contiguous().reshape(b, n, -1)
        return self.output(values)


class _TransMITBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_q=64, dropout=0.1):
        super().__init__()
        self.attn = _KerasStyleMultiHeadAttention(d_model, n_heads, d_q, dropout)
        self.ln1 = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn = nn.Linear(d_model, d_model)  # Keras Dense(d_model), default linear
        self.ln2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        return self.ln2(x + self.ffn(x))


class TransMITImputer(nn.Module):
    def __init__(
        self,
        seq_len=1080,
        d_in=2,
        s=16,
        d_model=128,
        d_q=64,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        alpha=0.5,
        max_train_windows=1024,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.d_in = int(d_in)
        self.s = int(s)
        self.window_len = self.s + 1
        self.alpha = float(alpha)
        self.max_train_windows = max_train_windows
        if self.s < 1 or self.seq_len <= self.s:
            raise ValueError('Need 1 <= s < seq_len')
        if not 0 <= self.alpha <= 1:
            raise ValueError('alpha must be in [0,1]')

        # Official: Dense(d_model) on (B,W,D), plus Dense(d_model) on (B,D,W)
        self.input_temporal = nn.Linear(d_in, d_model)
        self.input_feature = nn.Linear(self.window_len, d_model)
        self.temporal_blocks = nn.ModuleList([
            _TransMITBlock(d_model, num_heads, d_q, dropout)
            for _ in range(num_layers)
        ])
        self.feature_blocks = nn.ModuleList([
            _TransMITBlock(d_model, num_heads, d_q, dropout)
            for _ in range(num_layers)
        ])

        self.temporal_output = nn.Linear(d_model, d_in)
        self.feature_output = nn.Linear(d_model, self.window_len)
        self.fusion_output = nn.Linear(2 * d_in, d_in)

        # Identity until set_standardization(): the authors use StandardScaler
        # fitted on training data outside the Keras model.
        self.register_buffer('data_mean', torch.zeros(1, 1, d_in))
        self.register_buffer('data_std', torch.ones(1, 1, d_in))
        self.register_buffer('standardization_fitted', torch.tensor(False))

    @torch.no_grad()
    def set_standardization(self, mean, std):
        mean = torch.as_tensor(mean, dtype=self.data_mean.dtype, device=self.data_mean.device).reshape(1, 1, self.d_in)
        std = torch.as_tensor(std, dtype=self.data_std.dtype, device=self.data_std.device).reshape(1, 1, self.d_in)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError('mean and std must be finite; std strictly positive')
        self.data_mean.copy_(mean)
        self.data_std.copy_(std)
        self.standardization_fitted.fill_(True)

    def predict_window(self, window):
        if window.ndim != 3 or window.shape[1:] != (self.window_len, self.d_in):
            raise ValueError(f'Expected (N,{self.window_len},{self.d_in}); got {tuple(window.shape)}')
        x = self.input_temporal(window)
        xt = self.input_feature(window.transpose(1, 2))
        for block in self.temporal_blocks:
            x = block(x)
        for block in self.feature_blocks:
            xt = block(xt)
        x = self.temporal_output(x)
        xt = self.feature_output(xt).transpose(1, 2)
        return self.fusion_output(torch.cat((x, xt), dim=-1).mean(dim=1))

    def _check_inputs(self, x_in, m_hat):
        if x_in.ndim != 3 or x_in.shape[1:] != (self.seq_len, self.d_in):
            raise ValueError(f'Expected (B,{self.seq_len},{self.d_in}), got {tuple(x_in.shape)}')
        if m_hat.shape != x_in.shape:
            raise ValueError('m_hat and x_in must have same shape')

    @torch.no_grad()
    def forward(self, x_in, m_hat, predict_observed=False):
        self._check_inputs(x_in, m_hat)
        B, T, D = x_in.shape
        mask = (m_hat > 0.5) & torch.isfinite(x_in)
        observed_norm = torch.where(mask, (x_in - self.data_mean) / self.data_std, 0.)
        completed_norm = observed_norm.clone()
        predicted_norm = observed_norm.clone()
        for t in range(T):
            active = (~mask[:, t, :]).any(dim=-1)
            if predict_observed:
                active = torch.ones_like(active)
            if not bool(active.any()):
                continue
            # Independently batch segments that need predictions at time t.
            indices = torch.nonzero(active, as_tuple=False).flatten()
            past = completed_norm[indices, max(0, t-self.s):t, :]
            if past.size(1) < self.s:
                past = F.pad(past, (0, 0, self.s-past.size(1), 0))
            current = observed_norm[indices, t:t+1, :]
            window = torch.cat((past, current), dim=1)
            pred = self.predict_window(window)
            predicted_norm[indices, t, :] = pred
            completed_norm[indices, t, :] = torch.where(
                mask[indices, t, :], observed_norm[indices, t, :], pred
            )

        x_hat = predicted_norm * self.data_std + self.data_mean
        x_c = torch.where(mask, x_in, completed_norm * self.data_std + self.data_mean)
        return x_hat, x_c, predicted_norm, self.data_mean.expand(B, -1, -1), self.data_std.expand(B, -1, -1)

    @staticmethod
    def _weighted_rmse(y_pred, y_true, weight):
        n = weight.sum()
        if not bool(n > 0):
            return y_pred.sum() * 0.0
        return torch.sqrt(((y_pred - y_true).square() * weight).sum() / n + 1e-12)

    def training_loss(self, x_in, m_hat, x_true, i_mask, alpha=None, max_train_windows=None):
        self._check_inputs(x_in, m_hat)
        if x_true.shape != x_in.shape or i_mask.shape != x_in.shape:
            raise ValueError('x_true and i_mask must have same shape as x_in')
        a = self.alpha if alpha is None else float(alpha)
        B, T, D = x_in.shape
        valid = torch.isfinite(x_true)
        observed = (m_hat > 0.5) & valid
        targets = (i_mask > 0.5) & (~observed) & valid
        available = observed | targets
        choose = torch.nonzero(available.any(dim=-1).reshape(-1), as_tuple=False).flatten()
        if choose.numel() == 0:
            raise ValueError('No finite observed or artificially withheld ground-truth values')
        max_w = self.max_train_windows if max_train_windows is None else max_train_windows
        if max_w is not None and choose.numel() > max_w:
            choose = choose[torch.randperm(choose.numel(), device=choose.device)[:max_w]]
        # Normalize true history and observed current values separately.
        history_norm = torch.where(valid, (x_true - self.data_mean) / self.data_std, 0.)
        current_norm = torch.where(observed, (x_in - self.data_mean) / self.data_std, 0.)
        # Windows ending at every t: (B,T,W,D); s left-padding for t=0...s-1.
        all_windows = F.pad(history_norm.transpose(1, 2), (self.s, 0)).unfold(-1, self.window_len, 1)
        all_windows = all_windows.permute(0, 2, 3, 1).contiguous().reshape(B*T, self.window_len, D)
        windows = all_windows.index_select(0, choose).clone()
        flat_current = current_norm.reshape(B*T, D)
        windows[:, -1, :] = flat_current.index_select(0, choose)
        y_pred = self.predict_window(windows)
        y_true = history_norm.reshape(B*T, D).index_select(0, choose)
        obs_weight = observed.reshape(B*T, D).float().index_select(0, choose)
        imp_weight = targets.reshape(B*T, D).float().index_select(0, choose)
        recon = self._weighted_rmse(y_pred, y_true, obs_weight)
        impute = self._weighted_rmse(y_pred, y_true, imp_weight)
        total = a * recon + (1-a) * impute
        return total, recon, impute

    @torch.no_grad()
    def validation_loss(self, x_in, m_hat, x_true, i_mask, alpha=None):
        predicted, completed, _, _, _ = self.forward(x_in, m_hat, predict_observed=True)
        a = self.alpha if alpha is None else float(alpha)
        valid = torch.isfinite(x_true)
        obs = ((m_hat > 0.5) & valid).float()
        targets = ((i_mask > 0.5) & (m_hat <= 0.5) & valid).float()
        y_gt = torch.where(valid, (x_true - self.data_mean) / self.data_std, 0.)
        y_hat = (predicted - self.data_mean) / self.data_std
        recon = self._weighted_rmse(y_hat, y_gt, obs)
        impute = self._weighted_rmse(y_hat, y_gt, targets)
        total = a*recon + (1-a)*impute
        return total, recon, impute





