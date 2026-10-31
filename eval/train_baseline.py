"""Train scheduling baselines."""
import argparse
import copy
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from core.train import configure_runtime
from data.dataset import SCALES, load, save_json
from .baseline_models import DEFAULTS, configuration, build, instances, ptr_training, cvae_batch, calibrate_cvae
from . import hcps_PtrTasking_cmax as ptr

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=list(DEFAULTS), required=True)
    p.add_argument('--scale', choices=list(SCALES), required=True)
    p.add_argument('--data', default='datasets/dts5000')
    p.add_argument('--out', required=True)
    p.add_argument('--config')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    for k in ('epochs', 'batch_size', 'validation_every', 'validation_limit', 'patience', 'candidates', 'candidate_chunk', 'threads', 'seed'):
        p.add_argument('--' + k.replace('_', '-'), type=int)
    args = p.parse_args()
    cfg = configuration(args.method)
    if args.config:
        values = json.loads(Path(args.config).read_text())
        if set(values) - set(cfg):
            raise ValueError('Unknown baseline configuration key')
        cfg.update(values)
    for k in cfg:
        if getattr(args, k, None) is not None:
            cfg[k] = getattr(args, k)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'last.pt').exists() and (not args.resume):
        raise FileExistsError('Use a new output directory or --resume')
    configure_runtime(cfg['seed'], cfg['threads'])
    device = torch.device(args.device)
    raw = load(args.data, args.scale, 'train')
    val = load(args.data, args.scale, 'validation', cfg['validation_limit'])
    insts = instances(raw, args.scale)
    vinsts = instances(val, args.scale)
    n, m = (insts[0].num_tasks, insts[0].num_procs)
    model = build(args.method, n, m, cfg).to(device)
    ema = copy.deepcopy(model).eval() if cfg['ema'] else None
    if ema is not None:
        for p in ema.parameters():
            p.requires_grad_(False)
    optimizer_class = torch.optim.Adam if args.method == 'cvae-opt' else torch.optim.AdamW
    opt = optimizer_class(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    aux = {}
    if args.method == 'ptrtasking':
        data = ptr_training(insts, raw)
        f = data['features']
        std = f.std((0, 1), unbiased=False)
        aux = {'mean': f.mean((0, 1)), 'std': torch.where(std < 0.0001, 1.0, std)}
    epoch0 = 0
    best = float('inf')
    reference = float('inf')
    stale = 0
    best_epoch = 0
    if args.resume:
        ck = torch.load(out / 'last.pt', map_location=device, weights_only=False)
        if ck['config'] != cfg:
            raise ValueError('Resume configuration mismatch')
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        aux = ck['aux']
        if ema is not None:
            ema.load_state_dict(ck['ema'])
        epoch0 = ck['epoch']
        best = ck['best']
        reference = ck['reference']
        stale = ck['stale']
        best_epoch = ck['best_epoch']
    save_json(out / 'config.json', cfg)
    print(json.dumps(dict(method=args.method, scale=args.scale, parameters=sum((p.numel() for p in model.parameters())), train=len(insts), validation=len(vinsts))), flush=True)
    from .evaluate_baseline import evaluate_models
    for epoch in range(epoch0 + 1, cfg['epochs'] + 1):
        if cfg['patience'] > 0 and stale >= cfg['patience']:
            break
        begin = time.perf_counter()
        model.train()
        losses = []
        torch.manual_seed(2000000 + epoch)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(2000000 + epoch)
        if args.method == 'ptrtasking':
            if epoch <= cfg['warmup_epochs']:
                lr = cfg['learning_rate'] * epoch / max(1, cfg['warmup_epochs'])
            else:
                progress = (epoch - cfg['warmup_epochs']) / max(1, cfg['epochs'] - cfg['warmup_epochs'])
                lr = cfg['learning_rate'] * (cfg['minimum_lr_ratio'] + (1 - cfg['minimum_lr_ratio']) * 0.5 * (1 + math.cos(math.pi * progress)))
            for group in opt.param_groups:
                group['lr'] = lr
        for ids in torch.randperm(len(insts)).split(cfg['batch_size']):
            if args.method == 'transrl':
                from .transrl_batch import rollouts
                from .hcps_trans_RL_cmax import ppo_update
                for rep in range(cfg['rollouts']):
                    _, rows = rollouts(model, [insts[i] for i in ids.tolist()], [cfg['seed'] + epoch * 1000000 + int(i) * cfg['rollouts'] + rep for i in ids])
                    metrics = ppo_update(model, opt, rows, epochs=cfg['ppo_epochs'], minibatch=cfg['ppo_minibatch'])
                    losses.append(metrics['loss'])
                    if ema is not None:
                        with torch.no_grad():
                            for ep, p in zip(ema.parameters(), model.parameters()):
                                ep.lerp_(p, 1 - cfg['ema_decay'])
                continue
            opt.zero_grad(set_to_none=True)
            if args.method == 'ptrtasking':
                batch = {k: v[ids].to(device) for k, v in data.items()}
                loss = ptr.loss(model, batch, torch.arange(len(ids), device=device), aux['mean'].to(device), aux['std'].to(device), cfg)
            else:
                batch, y = cvae_batch(insts, raw, ids.tolist(), device)
                loss, _ = model.loss(batch, y, cfg['beta'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0 if args.method == 'cvae-opt' else 1.0, error_if_nonfinite=True)
            opt.step()
            losses.append(float(loss.detach()))
            if ema is not None:
                with torch.no_grad():
                    for ep, p in zip(ema.parameters(), model.parameters()):
                        ep.lerp_(p, 1 - cfg['ema_decay'])
        record = dict(epoch=epoch, loss=float(np.mean(losses)), training_seconds=time.perf_counter() - begin)
        if epoch % cfg['validation_every'] == 0 or epoch == cfg['epochs']:
            begin = time.perf_counter()
            variants = {'raw': model}
            if ema is not None:
                variants['ema'] = ema
            scores = {}
            for weight, selected in variants.items():
                if args.method == 'cvae-opt':
                    aux['latent_bound'] = calibrate_cvae(selected, insts, raw, device)
                summary, _ = evaluate_models(args.method, selected, vinsts, val, cfg, aux, device)
                scores[weight] = summary['mean_makespan']
                if scores[weight] < best:
                    best = scores[weight]
                    best_epoch = epoch
                    torch.save(dict(method=args.method, model=selected.state_dict(), config=cfg, aux=aux, epoch=epoch, scale=args.scale, n=n, m=m, validation_makespan=best, weights=weight), out / 'best.pt')
            value = min(scores.values())
            record.update(validation_makespan=value, validation_scores=scores, validation_seconds=time.perf_counter() - begin)
            if value < reference * (1 - cfg['min_improvement']):
                reference = value
                stale = 0
            else:
                stale += 1
        record.update(best_epoch=best_epoch, stale_validations=stale)
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict() if ema is not None else None, optimizer=opt.state_dict(), config=cfg, aux=aux, epoch=epoch, best=best, reference=reference, stale=stale, best_epoch=best_epoch), out / 'last.pt')
        with (out / 'training.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if cfg['patience'] > 0 and stale >= cfg['patience']:
            break
if __name__ == '__main__':
    main()
