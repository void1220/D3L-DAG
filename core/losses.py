"""Discrete and continuous denoising objectives."""
import math
import torch
from torch.nn import functional as F
from .schedules import mask_probability

def corrupt(clean, t, steps, kernel, m, schedule='linear', generator=None):
    chance = mask_probability(t.float() / steps, schedule)[:, None]
    replace = torch.rand(clean.shape, device=clean.device, generator=generator) < chance
    if kernel == 'absorbing':
        return torch.where(replace, -1, clean)
    if kernel != 'uniform':
        raise ValueError(kernel)
    n = clean.shape[-1] // 2
    sizes = torch.tensor([m] * n + [n] * n, device=clean.device)
    noise = (torch.rand(clean.shape, device=clean.device, generator=generator) * sizes).long()
    return torch.where(replace, noise, clean)

def alpha_bar(t, steps):
    y = torch.cos(math.pi * t.double() / steps / 2).square()
    return torch.where(t >= steps, 0.0, torch.where(t <= 0, 1.0, y)).float()

def continuous_prediction(logits):
    a, o = logits
    return torch.cat([a.float().softmax(-1) * math.sqrt(a.shape[-1]), o.float().softmax(-1) * math.sqrt(o.shape[-1])], -1)

def training_loss(model, batch, cfg):
    b, n, m = batch['ct'].shape
    clean = batch['tokens']
    device = clean.device
    if cfg['decoder'] == 'ar':
        pos = torch.randint(0, 2 * n, (b,), device=device)
        index = torch.arange(n, device=device)[None]
        visible = torch.cat([2 * index < pos[:, None], 2 * index + 1 < pos[:, None]], -1)
        state = torch.where(visible, clean, -1)
        al, ol = model(batch, state, 2 * n - pos)
        row = torch.arange(b, device=device)
        task = pos // 2
        la = F.cross_entropy(al[row, task].float(), clean[row, task], reduction='none')
        logit = ol[row, task].float()
        match = batch['groups'][row, task] & (logit > -100000000.0)
        lo = torch.logsumexp(logit, -1) - torch.logsumexp(logit.masked_fill(~match, -1000000000.0), -1)
        return torch.where(pos % 2 == 0, la, lo).mean()
    t = torch.randint(1, model.steps + 1, (b,), device=device)
    t = torch.where(torch.rand(b, device=device) < 0.25, model.steps, t)
    if cfg['decoder'] == 'continuous':
        a = F.one_hot(clean[:, :n], m).float() * math.sqrt(m)
        o = batch['groups'].float()
        o = o / o.sum(-1, keepdim=True) * math.sqrt(n)
        target = torch.cat([a, o], -1)
        ab = alpha_bar(t, model.steps)[:, None, None]
        state = ab.sqrt() * target + (1 - ab).sqrt() * torch.randn_like(target)
        pred = continuous_prediction(model(batch, state, t))
        return 0.5 * ((pred[..., :m] - target[..., :m]).square().mean() + (pred[..., m:] - target[..., m:]).square().mean())
    state = corrupt(clean, t, model.steps, cfg['kernel'], m, cfg['noise_schedule'])
    al, ol = (v.float() for v in model(batch, state, t))
    la = F.cross_entropy(al.transpose(1, 2), clean[:, :n])
    lo = (torch.logsumexp(ol, -1) - torch.logsumexp(ol.masked_fill(~batch['groups'], -1000000000.0), -1)).mean()
    return 0.5 * (la + lo)
