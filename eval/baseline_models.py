"""Shared baseline training and inference interfaces."""
import numpy as np
import torch
from core.decoding import edge_matrix, decode, decode_priority_schedule, evaluate_population
from data.dataset import reconstruct
from data.ptr_features import extract_instance_features_simple
from . import hcps_PtrTasking_cmax as ptr
from . import hcps_cvae_opt_cmax as cvae
from .hcps_trans_RL_cmax import MultiAgentTransformer
DEFAULTS = {'ptrtasking': dict(width=256, layers=2, dropout=0.05, epochs=800, batch_size=256, learning_rate=0.0005, weight_decay=0.0001, warmup_epochs=20, minimum_lr_ratio=0.1, validation_every=50, patience=3, ema=True, ema_decay=0.999, label_smoothing=0.01, lambda_task=1.0, lambda_proc=2.0, lambda_proc_time=0.2, sample_temperature=0.9, sample_task_topk=6, sample_proc_topk=3), 'cvae-opt': dict(width=128, latent=100, epochs=800, batch_size=8, learning_rate=0.001, weight_decay=0.0, validation_every=10, patience=15, ema=False, beta=0.0001), 'transrl': dict(width=128, layers=2, heads=4, epochs=20, batch_size=8, learning_rate=0.0003, weight_decay=0.01, validation_every=5, patience=2, ema=True, ema_decay=0.99, rollouts=1, ppo_epochs=4, ppo_minibatch=128, candidate_chunk=128)}

def configuration(method):
    return dict(seed=42, candidates=256, candidate_chunk=64, validation_limit=200, min_improvement=0.001, threads=4) | DEFAULTS[method]

def build(method, n, m, cfg):
    if method == 'ptrtasking':
        return ptr.PtrTaskingFast(120, cfg['width'], cfg['layers'], cfg['dropout'])
    if method == 'cvae-opt':
        return cvae.CVAEOptDTS(n, m, cfg['width'], cfg['latent'])
    if method == 'transrl':
        return MultiAgentTransformer(n, m, cfg['width'], cfg['layers'], cfg['heads'])
    raise ValueError(method)

def instances(raw, scale):
    return [reconstruct(scale, raw, i) for i in range(len(raw['seeds']))]

def ptr_features(inst):
    inst.deadline = 0.0
    f = extract_instance_features_simple(inst, max_tasks=inst.num_tasks, max_procs=inst.num_procs, topk_ct=14)
    f[0, 2] = 0.0
    return f

def ptr_training(insts, raw):
    features = []
    orders = []
    ready = []
    for i, inst in enumerate(insts):
        n = inst.num_tasks
        token = raw['tokens'][i]
        order = decode_priority_schedule(inst, token[:n], token[n:]).order
        e = edge_matrix(inst)
        deg = e.sum(0)
        done = np.zeros(n, bool)
        masks = []
        for t in order:
            masks.append((deg == 0) & ~done)
            done[t] = True
            deg = deg - e[t]
        features.append(ptr_features(inst))
        orders.append(order)
        ready.append(masks)
    return {k: torch.as_tensor(v) for k, v in dict(features=np.stack(features), comp=raw['comp_time'].astype(np.float32), order=np.stack(orders), assignment=raw['tokens'][:, :insts[0].num_tasks], ready=np.asarray(ready)).items()}

def cvae_batch(insts, raw, ids, device):
    items = [cvae.prepare(insts[int(i)]) for i in ids]
    targets = torch.stack([cvae.target(insts[int(i)], raw['tokens'][int(i)]) for i in ids]).to(device)
    return (cvae.batch(items, device), targets)

@torch.no_grad()
def calibrate_cvae(model, insts, raw, device):
    model.eval()
    samples = []
    for i in range(min(32, len(insts))):
        data, y = cvae_batch(insts, raw, [i], device)
        _, mu, lv = model.posterior(data, y)
        samples.append((mu + torch.randn((256, model.latent), device=device) * (lv * 0.5).exp()).abs().flatten())
    return max(0.001, float(torch.quantile(torch.cat(samples), 0.99)))

@torch.no_grad()
def candidates(method, model, inst, cfg, aux, seed):
    device = next(model.parameters()).device
    model.eval()
    n, m = (inst.num_tasks, inst.num_procs)
    if method == 'ptrtasking':
        return ptr.sample(model, ptr_features(inst), edge_matrix(inst), aux['mean'].to(device), aux['std'].to(device), k=cfg['candidates'], chunk=cfg['candidate_chunk'], temperature=cfg['sample_temperature'], task_topk=cfg['sample_task_topk'], proc_topk=cfg['sample_proc_topk'], seed=seed)
    if method != 'cvae-opt':
        raise ValueError(method)
    data = cvae.batch([cvae.prepare(inst)], device)
    hidden = model.instance_embedding(data['features'])
    gen = torch.Generator(device=device).manual_seed(seed)
    tokens = []
    for left in range(0, cfg['candidates'], cfg['candidate_chunk']):
        k = min(cfg['candidate_chunk'], cfg['candidates'] - left)
        z = (torch.rand((k, model.latent), device=device, generator=gen) * 2 - 1) * aux['latent_bound']
        pairs, _ = model.rollout(data, z, hidden=hidden)
        tokens.extend((cvae.to_tokens(p, n, m) for p in pairs.cpu().numpy()))
    y = np.stack(tokens)
    return (y[:, :n].copy(), y[:, n:].astype(np.float64))

def solve_model(method, model, inst, cfg, aux, seed):
    if method == 'transrl':
        from .transrl_batch import solve_batch
        result = solve_batch(model, inst, cfg['candidates'], seed, cfg['candidate_chunk'])
        return result['schedule'] | {'candidates': cfg['candidates']}
    a, p = candidates(method, model, inst, cfg, aux, seed)
    e = edge_matrix(inst)
    values = evaluate_population(inst.comp_time, inst.comm_time, e, a, p)
    j = int(values.argmin())
    mk, st, ft, order = decode(inst.comp_time, inst.comm_time, e, a[j], p[j])
    return dict(makespan=float(mk), assignment=a[j].tolist(), priorities=p[j].tolist(), order=order.tolist(), start=st.tolist(), finish=ft.tolist(), candidates=cfg['candidates'])
