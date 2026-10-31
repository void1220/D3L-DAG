"""Joint task and processor permutation."""
import torch

def permute(batch, tp=None, pp=None):
    b, n, m = batch['ct'].shape
    device = batch['ct'].device
    if tp is None:
        tp = torch.rand(b, n, device=device).argsort(-1)
    if pp is None:
        pp = torch.rand(b, m, device=device).argsort(-1)
    pi = pp.argsort(-1)
    bi = torch.arange(b, device=device)[:, None]
    out = {}
    out['ct'] = batch['ct'][bi, tp].gather(2, pp[:, None].expand(-1, n, -1))
    out['eligible'] = batch['eligible'][bi, tp].gather(2, pp[:, None].expand(-1, n, -1))
    out['extra'] = batch['extra'][bi, tp]
    out['path'] = batch['path'][bi, tp].gather(2, pp[:, None, :, None].expand(-1, n, -1, 3))
    out['reach'] = batch['reach'][bi, tp].gather(2, tp[:, None, :, None].expand(-1, n, -1, 2))
    for k in ['comm', 'edge']:
        out[k] = batch[k][bi, tp].gather(2, tp[:, None].expand(-1, n, -1))
    if 'tokens' in batch:
        aa = pi.gather(1, batch['tokens'][:, :n].gather(1, tp))
        oo = batch['tokens'][:, n:].gather(1, tp)
        out['tokens'] = torch.cat([aa, oo], -1)
        out['groups'] = batch['groups'][bi, tp]
    return (out, tp, pp)
