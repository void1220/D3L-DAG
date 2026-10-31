"""GS and FFT dataset access."""
from pathlib import Path
import json
import numpy as np
import torch
from core.decoding import edge_matrix, decode_priority_schedule
from core.features import structural_features
from .ptr_features import _topo_levels_from_preds, _critical_path_tlevel_blevel
SCALES = {'GS01': ('gs', 5, 5), 'GS02': ('gs', 8, 9), 'GS03': ('gs', 12, 11), 'GS04': ('gs', 16, 14), 'FFT01': ('fft', 4, 5), 'FFT02': ('fft', 8, 9), 'FFT03': ('fft', 16, 11), 'FFT04': ('fft', 26, 14)}

def generate_instance(scale, seed):
    family, rho, m = SCALES[scale]
    if family == 'gs':
        from env.hcps_env_gs import generate_random_hcps_instance
    else:
        from env.hcps_env_fft import generate_random_hcps_instance
    inst = generate_random_hcps_instance(seed=int(seed), rho=rho, num_procs=m)
    inst.eligibility_mask = np.ones_like(inst.comp_time, dtype=bool)
    return inst

def load(folder, scale, split, limit=None):
    if split not in ('train', 'validation', 'test'):
        raise ValueError(split)
    with np.load(Path(folder) / scale / (split + '.npz'), allow_pickle=False) as f:
        raw = {k: f[k][:limit].copy() if limit is not None else f[k].copy() for k in f.files}
    if not len(raw['seeds']):
        raise ValueError('Empty split')
    return raw

def reconstruct(scale, raw, i):
    inst = generate_instance(scale, int(raw['seeds'][i]))
    inst.comp_time = raw['comp_time'][i].astype(np.float64)
    inst.comm_time = raw['comm_time'][i].astype(np.float64)
    return inst

def instance_features(inst, tokens=None):
    n, m = (inst.num_tasks, inst.num_procs)
    level = _topo_levels_from_preds(n, inst.predecessors)
    tl, bl = _critical_path_tlevel_blevel(n, inst.predecessors, inst.successors, inst.comp_time.mean(1), inst.comm_time)
    ex, path, reach = structural_features(inst)
    extra = np.concatenate([np.stack([level / max(1, level.max()), tl / max(1e-06, tl.max()), bl / max(1e-06, bl.max())], -1), ex], -1)
    d = dict(ct=inst.comp_time.astype(np.float32), comm=inst.comm_time.astype(np.float32), edge=edge_matrix(inst).astype(np.float32), eligible=inst.eligibility_mask, extra=extra.astype(np.float32), path=path, reach=reach)
    if tokens is not None:
        start = decode_priority_schedule(inst, tokens[:n], tokens[n:]).start
        d['tokens'] = np.asarray(tokens, dtype=np.int64)
        d['groups'] = np.isclose(start[:, None], start[np.argsort(tokens[n:])][None, :], atol=1e-09, rtol=0)
    return d

def _prepare(args):
    scale, seed, ct, comm, tokens = args
    inst = generate_instance(scale, seed)
    inst.comp_time = ct
    inst.comm_time = comm
    return instance_features(inst, tokens)

def prepare(raw, scale, labels=False, workers=0):
    jobs = ((scale, int(s), raw['comp_time'][i], raw['comm_time'][i], raw['tokens'][i] if labels else None) for i, s in enumerate(raw['seeds']))
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            items = list(pool.map(_prepare, jobs, chunksize=16))
    else:
        items = list(map(_prepare, jobs))
    return {k: torch.as_tensor(np.stack([x[k] for x in items])) for k in items[0]}

def take(data, ids, device=None):
    return {k: v[ids].to(device) if device is not None else v[ids] for k, v in data.items()}

def save_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
