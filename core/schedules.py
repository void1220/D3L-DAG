"""Diffusion noise schedules."""
import math
import torch
NAMES = ('linear', 'cosine', 'quadratic', 'sigmoid', 'piecewise')

def mask_probability(u, name='linear'):
    if name not in NAMES:
        raise ValueError(name)
    u = u.clamp(0, 1)
    if name == 'linear':
        y = u
    elif name == 'cosine':
        y = 0.5 * (1 - torch.cos(math.pi * u))
    elif name == 'quadratic':
        y = u.square()
    elif name == 'sigmoid':
        lo = 1 / (1 + math.exp(6))
        hi = 1 / (1 + math.exp(-6))
        y = (torch.sigmoid(12 * u - 6) - lo) / (hi - lo)
    else:
        y = torch.where(u <= 0.5, 0.5 * u, 1.5 * u - 0.5)
    return torch.where(u <= 0, 0.0, torch.where(u >= 1, 1.0, y))

def reveal_probability(t, steps, name='linear'):
    if not 1 <= t <= steps:
        raise ValueError((t, steps))
    if name == 'linear':
        return 1 / t
    f = mask_probability(torch.tensor([t - 1, t], dtype=torch.float64) / steps, name)
    return float((f[1] - f[0]) / f[1])
