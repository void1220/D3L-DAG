"""Evaluate D3L schedules."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from core.decoding import evaluate_population, edge_matrix
from core.registry import get_d3l_variant
from core.sampling import solve
from data.dataset import SCALES, load, prepare, take, reconstruct, save_json

def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)

def summarize(rows):
    v = np.array([r['makespan'] for r in rows])
    h = np.array([r['heft'] for r in rows])
    return dict(instances=len(rows), mean_makespan=float(v.mean()), std_makespan=float(v.std(ddof=1)) if len(v) > 1 else 0.0, mean_seconds=float(np.mean([r['seconds'] for r in rows])), heft_mean=float(h.mean()), improvement_percent=float(100 * (h.mean() - v.mean()) / h.mean()))

@torch.no_grad()
def evaluate_d3l(model, data, raw, cfg, device, report=True):
    rows = []
    model.eval()
    comp = raw['comp_time'][0]
    comm = raw['comm_time'][0]
    edge = data['edge'][0].numpy().astype(bool)
    n, m = comp.shape
    evaluate_population(comp, comm, edge, np.zeros((1, n), dtype=np.int64), np.arange(n, dtype=float)[None])
    amp = cfg['amp'] and torch.device(device).type == 'cuda' and torch.cuda.is_bf16_supported()
    for i, seed in enumerate(raw['seeds']):
        batch = take(data, [i], device)
        sync(device)
        begin = time.perf_counter()
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=amp):
            result = solve(model, batch, raw['comp_time'][i], raw['comm_time'][i], data['edge'][i].numpy().astype(bool), cfg, int(seed) + cfg['seed'])
        sync(device)
        result.update(index=i, seed=int(seed), seconds=time.perf_counter() - begin, heft=float(raw['expert_makespan'][i]))
        rows.append(result)
        if report and ((i + 1) % 10 == 0 or i + 1 == len(raw['seeds'])):
            print(f"Evaluation {i + 1}/{len(raw['seeds'])} mean={np.mean([x['makespan'] for x in rows]):.4f}", flush=True)
    return (summarize(rows), rows)

def save_results(out, summary, rows):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'summary.json', summary)
    save_json(out / 'schedules.json', rows)
    with (out / 'results.csv').open('w', encoding='utf-8', newline='') as f:
        keys = ['index', 'seed', 'heft', 'makespan', 'seconds']
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', default='datasets/dts5000')
    p.add_argument('--split', choices=['validation', 'test'], default='test')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--limit', type=int)
    p.add_argument('--candidates', type=int)
    p.add_argument('--candidate-chunk', type=int)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--assignment-temperature', type=float)
    p.add_argument('--priority-temperature', type=float)
    p.add_argument('--resampling', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--resampling-rounds', type=int)
    p.add_argument('--restart-step', type=int)
    p.add_argument('--parents', type=int)
    a = p.parse_args()
    from core.train import configure_runtime
    ck = torch.load(a.checkpoint, map_location=a.device, weights_only=False)
    cfg = ck['config'].copy()
    for k in ('candidates', 'candidate_chunk', 'assignment_temperature', 'priority_temperature', 'resampling', 'resampling_rounds', 'restart_step', 'parents'):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    configure_runtime(cfg['seed'], a.threads)
    model = get_d3l_variant(ck['n'], ck['m'], cfg).to(a.device)
    model.load_state_dict(ck['model'])
    raw = load(a.data, ck['scale'], a.split, a.limit)
    data = prepare(raw, ck['scale'])
    summary, rows = evaluate_d3l(model, data, raw, cfg, a.device)
    summary.update(scale=ck['scale'], split=a.split, epoch=ck['epoch'], weights=ck['weights'], sampling={k: cfg[k] for k in ('candidates', 'candidate_chunk', 'resampling', 'resampling_rounds', 'restart_step', 'parents', 'assignment_temperature', 'priority_temperature')})
    save_results(a.out, summary, rows)
    print(json.dumps(summary), flush=True)
if __name__ == '__main__':
    main()
