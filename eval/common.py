"""Baseline instance and schedule adapters."""
from types import SimpleNamespace
import numpy as np
import torch

def sanitize(inst):
    n, m = (inst.num_tasks, inst.num_procs)
    ct = np.array(inst.comp_time, dtype=np.float64, copy=True)
    comm = np.array(inst.comm_time, dtype=np.float64, copy=True)
    eligible = np.array(getattr(inst, 'eligibility_mask', np.ones((n, m), bool)), copy=True)
    if ct.shape != (n, m) or comm.shape != (n, n) or eligible.shape != (n, m):
        raise ValueError('invalid instance dimensions')
    if eligible.dtype != bool or not eligible.any(1).all():
        raise ValueError('invalid eligibility')
    if not np.isfinite(ct).all() or (ct <= 0).any() or (not np.isfinite(comm).all()) or (comm < 0).any():
        raise ValueError('invalid costs')
    preds = {i: list(inst.predecessors.get(i, [])) for i in range(n)}
    succ = {i: [] for i in range(n)}
    for i, ps in preds.items():
        if len(set(ps)) != len(ps) or any((p < 0 or p >= n for p in ps)):
            raise ValueError('invalid graph')
        for p in ps:
            succ[p].append(i)
    seen = set()
    while len(seen) < n:
        ready = [i for i in range(n) if i not in seen and set(preds[i]) <= seen]
        if not ready:
            raise ValueError('cyclic graph')
        seen.update(ready)
    return SimpleNamespace(num_tasks=n, num_procs=m, comp_time=ct, comm_time=comm, eligibility_mask=eligible, predecessors=preds, successors=succ)

def condition(inst, device='cpu'):
    inst = sanitize(inst)
    n, m = (inst.num_tasks, inst.num_procs)
    edge = np.zeros((n, n))
    for i, ps in inst.predecessors.items():
        edge[ps, i] = 1
    scale = max(float(inst.comp_time.mean()), 1e-08)
    feat = np.concatenate([np.log1p(inst.comp_time / scale), inst.eligibility_mask, np.log1p(inst.comm_time / scale), np.log1p(inst.comm_time.T / scale), edge, edge.T], axis=1)
    return dict(features=torch.tensor(feat, dtype=torch.float32, device=device)[None], eligible=torch.tensor(inst.eligibility_mask, device=device)[None])

def decode(inst, tokens):
    from core.decoding import decode_priority_schedule
    n = inst.num_tasks
    return decode_priority_schedule(inst, tokens[:n], tokens[n:])

def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)

def schedule_json(a, sched):
    return dict(assignment=np.asarray(a).tolist(), order=sched.order.tolist(), start=sched.start.tolist(), finish=sched.finish.tolist(), makespan=float(sched.makespan))
