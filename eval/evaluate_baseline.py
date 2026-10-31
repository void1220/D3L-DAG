"""Evaluate scheduling baselines."""
import argparse
import json
import time
import numpy as np
import torch
from core.decoding import heft, edge_matrix, evaluate_population
from core.train import configure_runtime
from data.dataset import SCALES, load
from .baseline_models import instances, build, solve_model
from .evaluate import summarize, save_results, sync

def solve_ails(inst, args):
    from .hcps_AILS_cmax import AILSCfg, load_lstm_bundle, ails_solve_with_sim, _build_cond_vec_for_instance
    from data.ptr_features import extract_instance_features_simple
    if args.scale.startswith('GS'):
        from env.hcps_env_gs import HCPSSimulator
    else:
        from env.hcps_env_fft import HCPSSimulator
    if not args.assignment_checkpoint or not args.order_checkpoint:
        raise ValueError('AILS requires both pretrained denoiser checkpoints')
    if not hasattr(args, '_bundles'):
        args._bundles = (load_lstm_bundle(args.assignment_checkpoint, args.device), load_lstm_bundle(args.order_checkpoint, args.device))
    a, o = args._bundles
    cond = _build_cond_vec_for_instance(inst, sample=None, assign_bundle=a, order_bundle=o, extract_instance_features_simple=extract_instance_features_simple)
    cfg = AILSCfg(seed=args.seed)
    if args.seconds is not None:
        cfg.runtime_max_s = args.seconds
    sim = HCPSSimulator(inst)
    alloc, order, _, _ = ails_solve_with_sim(inst, sim, cfg, assign_bundle=a, order_bundle=o, cond_vec=cond)
    r = sim.evaluate_schedule(alloc, order)
    return dict(makespan=float(r.makespan), assignment=alloc, order=order)

@torch.no_grad()
def evaluate_models(method, model, insts, raw, cfg, aux, device, args=None):
    rows = []
    first = insts[0]
    n = first.num_tasks
    evaluate_population(first.comp_time, first.comm_time, edge_matrix(first), np.zeros((1, n), np.int64), np.arange(n, dtype=float)[None])
    for i, inst in enumerate(insts):
        seed = int(raw['seeds'][i]) + cfg['seed']
        sync(device)
        begin = time.perf_counter()
        if method == 'heft':
            a, p, st, ft, mk = heft(inst.comp_time, inst.comm_time, edge_matrix(inst))
            r = dict(makespan=float(mk), assignment=a.tolist(), start=st.tolist(), finish=ft.tolist())
        elif method == 'ga':
            from .hcps_ga_cmax import genetic
            r = genetic(inst.comp_time, inst.comm_time, edge_matrix(inst), seed, seconds=args.seconds if args.seconds is not None else 120.0, max_generations=args.generations, patience=args.ga_patience)
        elif method == 'ails':
            r = solve_ails(inst, args)
        else:
            r = solve_model(method, model, inst, cfg, aux, seed)
        sync(device)
        r.update(index=i, seed=int(raw['seeds'][i]), heft=float(raw['expert_makespan'][i]), seconds=time.perf_counter() - begin)
        rows.append(r)
        if (i + 1) % 10 == 0 or i + 1 == len(insts):
            print(f"{method}: {i + 1}/{len(insts)} mean={np.mean([x['makespan'] for x in rows]):.4f}", flush=True)
    return (summarize(rows), rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['heft', 'ga', 'ails', 'transrl', 'ptrtasking', 'cvae-opt'], required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--scale', choices=list(SCALES))
    p.add_argument('--data', default='datasets/dts5000')
    p.add_argument('--out', required=True)
    p.add_argument('--split', choices=['validation', 'test'], default='test')
    p.add_argument('--limit', type=int)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--candidates', type=int)
    p.add_argument('--candidate-chunk', type=int)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--seconds', type=float)
    p.add_argument('--generations', type=int, default=5000)
    p.add_argument('--ga-patience', type=int, default=500)
    p.add_argument('--assignment-checkpoint')
    p.add_argument('--order-checkpoint')
    a = p.parse_args()
    configure_runtime(a.seed, a.threads)
    model = None
    aux = {}
    overrides = dict(seed=a.seed)
    if a.candidates is not None:
        overrides['candidates'] = a.candidates
    if a.candidate_chunk is not None:
        overrides['candidate_chunk'] = a.candidate_chunk
    cfg = dict(candidates=256, candidate_chunk=64) | overrides
    if a.method in ('transrl', 'ptrtasking', 'cvae-opt'):
        if not a.checkpoint:
            raise ValueError('--checkpoint is required')
        ck = torch.load(a.checkpoint, map_location=a.device, weights_only=False)
        if ck['method'] != a.method:
            raise ValueError('Checkpoint method mismatch')
        if a.scale and a.scale != ck['scale']:
            raise ValueError('Checkpoint scale mismatch')
        a.scale = ck['scale']
        cfg = ck['config'] | overrides
        aux = ck['aux']
        model = build(a.method, ck['n'], ck['m'], cfg).to(a.device)
        model.load_state_dict(ck['model'])
    if not a.scale:
        raise ValueError('--scale is required')
    raw = load(a.data, a.scale, a.split, a.limit)
    insts = instances(raw, a.scale)
    summary, rows = evaluate_models(a.method, model, insts, raw, cfg, aux, a.device, a)
    summary.update(method=a.method, scale=a.scale, split=a.split)
    save_results(a.out, summary, rows)
    print(json.dumps(summary), flush=True)
if __name__ == '__main__':
    main()
