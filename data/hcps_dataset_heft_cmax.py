"""Generate labeled GS and FFT datasets."""
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .dataset import SCALES, generate_instance, save_json
from core.decoding import edge_matrix, heft

def generate_one(args):
    scale, seed = args
    inst = generate_instance(scale, seed)
    a, p, st, ft, mk = heft(inst.comp_time, inst.comm_time, edge_matrix(inst))
    order = np.lexsort((np.arange(inst.num_tasks), st))
    ranks = np.empty(inst.num_tasks, dtype=np.int64)
    ranks[order] = np.arange(inst.num_tasks)
    return dict(seeds=seed, comp_time=inst.comp_time, comm_time=inst.comm_time, tokens=np.concatenate([a, ranks]), expert_makespan=mk)

def main(family=None):
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='datasets/dts5000')
    p.add_argument('--scales', nargs='+', choices=list(SCALES), default=[s for s in SCALES if family is None or s.lower().startswith(family)])
    p.add_argument('--instances', type=int, default=5000)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if a.instances < 10 or a.instances % 10:
        raise ValueError('instances must be a positive multiple of 10')
    for scale in a.scales:
        dest = Path(a.out) / scale
        if any(((dest / (s + '.npz')).exists() for s in ('train', 'validation', 'test'))):
            raise FileExistsError(dest)
        offset = list(SCALES).index(scale) * 10000000
        seeds = np.arange(a.instances, dtype=np.int64) + a.seed + offset
        jobs = [(scale, int(s)) for s in seeds]
        if a.workers > 1:
            with ProcessPoolExecutor(max_workers=a.workers) as pool:
                rows = list(pool.map(generate_one, jobs, chunksize=16))
        else:
            rows = list(map(generate_one, jobs))
        arrays = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
        order = np.random.default_rng(a.seed + offset).permutation(a.instances)
        cut = a.instances // 10
        parts = {'train': order[:8 * cut], 'validation': order[8 * cut:9 * cut], 'test': order[9 * cut:]}
        dest.mkdir(parents=True, exist_ok=True)
        for split, ids in parts.items():
            np.savez_compressed(dest / (split + '.npz'), **{k: v[ids] for k, v in arrays.items()})
        save_json(dest / 'dataset.json', dict(scale=scale, instances=a.instances, split={k: len(v) for k, v in parts.items()}, seed=a.seed, generator=SCALES[scale], label_method='HEFT'))
        print(f'{scale}: ' + ', '.join((f'{k}={len(v)}' for k, v in parts.items())), flush=True)
if __name__ == '__main__':
    main()
