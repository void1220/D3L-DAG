"""D3L training and checkpoint selection."""
import argparse
import copy
import json
import random
import math
import time
from pathlib import Path
import numpy as np
import torch
from numba import set_num_threads
from .config import D3LConfig
from .registry import get_d3l_variant
from .losses import training_loss
from .augmentation import permute
from data.dataset import SCALES, load, prepare, take, save_json

def configure_runtime(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_config(args):
    cfg = D3LConfig().to_dict()
    if args.config:
        values = json.loads(Path(args.config).read_text())
        unknown = set(values) - set(cfg)
        if unknown:
            raise ValueError(f'Unknown configuration: {unknown}')
        cfg.update(values)
    for k in cfg:
        value = getattr(args, k, None)
        if value is not None:
            cfg[k] = value
    if cfg['decoder'] not in ('d3l', 'ar', 'continuous'):
        raise ValueError('decoder')
    if cfg['kernel'] not in ('absorbing', 'uniform'):
        raise ValueError('kernel')
    if cfg['feature_mode'] not in ('full', 'base', 'position', 'gcn'):
        raise ValueError('feature_mode')
    if cfg['candidates'] < 1 or cfg['epochs'] < 1 or cfg['validation_every'] < 1:
        raise ValueError('Invalid training budget')
    return cfg

def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='datasets/dts5000')
    p.add_argument('--scale', choices=list(SCALES), required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--config')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    for k, v in D3LConfig().to_dict().items():
        if isinstance(v, bool):
            p.add_argument('--' + k.replace('_', '-'), action=argparse.BooleanOptionalAction, default=None)
        else:
            p.add_argument('--' + k.replace('_', '-'), type=type(v), default=None)
    return p

def run(args):
    from eval.evaluate import evaluate_d3l
    cfg = read_config(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'last.pt').exists() and (not args.resume):
        raise FileExistsError('Use a new output directory or --resume')
    configure_runtime(cfg['seed'], cfg['threads'])
    device = torch.device(args.device)
    print(f'{args.scale}: preparing training and validation data', flush=True)
    raw = load(args.data, args.scale, 'train')
    val = load(args.data, args.scale, 'validation', cfg['validation_limit'])
    data = prepare(raw, args.scale, True, args.workers)
    vdata = prepare(val, args.scale, False, args.workers)
    n, m = raw['comp_time'].shape[1:]
    model = get_d3l_variant(n, m, cfg).to(device)
    ema = copy.deepcopy(model).eval() if cfg['ema'] else None
    if ema is not None:
        for p in ema.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW([dict(params=[p for p in model.parameters() if p.ndim >= 2], weight_decay=cfg['weight_decay']), dict(params=[p for p in model.parameters() if p.ndim < 2], weight_decay=0.0)], lr=cfg['learning_rate'])
    start_epoch = 0
    best = float('inf')
    significant = float('inf')
    stale = 0
    best_epoch = 0
    if args.resume:
        ck = torch.load(out / 'last.pt', map_location=device, weights_only=False)
        if ck['config'] != cfg:
            raise ValueError('Resume configuration must match the saved configuration')
        model.load_state_dict(ck['raw_model'])
        opt.load_state_dict(ck['optimizer'])
        if ema is not None:
            ema.load_state_dict(ck['ema_model'])
        start_epoch = ck['epoch']
        best = ck['best']
        significant = ck['significant']
        stale = ck['stale']
        best_epoch = ck['best_epoch']
        torch.set_rng_state(ck['rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']])
    save_json(out / 'config.json', cfg)
    print(json.dumps(dict(scale=args.scale, parameters=sum((p.numel() for p in model.parameters())), train=len(raw['seeds']), validation=len(val['seeds']), config=cfg)), flush=True)
    amp = cfg['amp'] and device.type == 'cuda' and torch.cuda.is_bf16_supported()
    batches = math.ceil(len(raw['seeds']) / cfg['batch_size'])
    for epoch in range(start_epoch + 1, cfg['epochs'] + 1):
        if cfg['patience'] > 0 and stale >= cfg['patience']:
            break
        model.train()
        begin = time.perf_counter()
        losses = []
        torch.manual_seed(2000000 + epoch)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(2000000 + epoch)
        for bi, ids in enumerate(torch.randperm(len(raw['seeds'])).split(cfg['batch_size'])):
            step = (epoch - 1) * batches + bi
            warmup = cfg['warmup_epochs'] * batches
            hold = cfg['hold_epochs'] * batches
            total = cfg['schedule_epochs'] * batches
            if step < warmup:
                lr = cfg['learning_rate'] * (step + 1) / max(1, warmup)
            elif step < hold:
                lr = cfg['learning_rate']
            else:
                lr = cfg['learning_rate_floor'] + (cfg['learning_rate'] - cfg['learning_rate_floor']) * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - hold) / max(1, total - hold - 1))))
            for group in opt.param_groups:
                group['lr'] = lr
            torch.manual_seed(1000000 + step)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(1000000 + step)
            batch = take(data, ids, device)
            if cfg['permutation_augmentation']:
                batch, _, _ = permute(batch)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                loss = training_loss(model, batch, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            if ema is not None:
                with torch.no_grad():
                    for ep, p in zip(ema.parameters(), model.parameters()):
                        ep.lerp_(p, 1 - cfg['ema_decay'])
            losses.append((float(loss.detach()), len(ids)))
        record = dict(epoch=epoch, loss=sum((x * k for x, k in losses)) / sum((k for _, k in losses)), learning_rate=lr, training_seconds=time.perf_counter() - begin)
        selected = ema if ema is not None else model
        if epoch % cfg['validation_every'] == 0 or epoch == cfg['epochs']:
            begin = time.perf_counter()
            summary, _ = evaluate_d3l(selected, vdata, val, cfg, device, report=True)
            value = summary['mean_makespan']
            record.update(validation_makespan=value, validation_seconds=time.perf_counter() - begin)
            if value < best:
                best = value
                best_epoch = epoch
                torch.save(dict(model=selected.state_dict(), config=cfg, epoch=epoch, scale=args.scale, n=n, m=m, weights='ema' if ema is not None else 'raw', validation_makespan=value), out / 'best.pt')
            if value < significant * (1 - cfg['min_improvement']):
                significant = value
                stale = 0
            else:
                stale += 1
        record.update(best_epoch=best_epoch, stale_validations=stale)
        torch.save(dict(raw_model=model.state_dict(), ema_model=ema.state_dict() if ema is not None else None, optimizer=opt.state_dict(), config=cfg, epoch=epoch, best=best, significant=significant, stale=stale, best_epoch=best_epoch, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else []), out / 'last.pt')
        with (out / 'training.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if cfg['patience'] > 0 and stale >= cfg['patience']:
            break
    return out / 'best.pt'
if __name__ == '__main__':
    run(parser().parse_args())
