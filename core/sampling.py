"""Candidate generation and schedule selection."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from .losses import corrupt, alpha_bar, continuous_prediction
from .schedules import mask_probability, reveal_probability
from .decoding import evaluate_population, decode

def draw(logits, temperature, generator):
    if temperature == 0:
        return logits.argmax(-1)
    u = torch.rand(logits.shape, device=logits.device, generator=generator).clamp(1e-07, 1 - 1e-07)
    return (logits / temperature - torch.log(-torch.log(u))).argmax(-1)

def project_priority(scores, order, temperature, generator):
    rows, n, _ = scores.shape
    out = torch.empty((rows, n), device=scores.device, dtype=torch.long)
    used = torch.zeros((rows, n), device=scores.device, dtype=torch.bool)
    rr = torch.arange(rows, device=scores.device)
    for j in range(n):
        task = order[:, j]
        v = draw(scores[rr, task].masked_fill(used, -1000000000.0), temperature, generator)
        out[rr, task] = v
        used[rr, v] = True
    return out

def uniform_posterior(p0, xt, t, steps, schedule):
    if t == 1:
        return p0
    fs = mask_probability(torch.tensor([t - 1, t], device=p0.device, dtype=torch.float64) / steps, schedule)
    ap = 1 - fs[0]
    at = 1 - fs[1]
    beta = 1 - at / ap
    k = p0.shape[-1]
    one = F.one_hot(xt, k).to(p0.dtype)
    px = p0.gather(-1, xt.unsqueeze(-1))
    ud = (1 - at) / k
    us = at + ud
    common = (1 - ap) / k * ((1 - px) / ud + px / us)
    score = torch.where(one.bool(), common + ap * px / us, common + ap * p0 / ud)
    result = score * (beta / k + (1 - beta) * one)
    return (result / result.sum(-1, keepdim=True)).float()

@torch.no_grad()
def sample(model, batch, count, seed, cfg, initial=None, start_t=None):
    model.eval()
    b, n, m = batch['ct'].shape
    device = batch['ct'].device
    gen = torch.Generator(device=device).manual_seed(int(seed))
    expanded = {k: v.repeat_interleave(count, 0) for k, v in batch.items() if k not in ('tokens', 'groups')}
    rows = b * count
    rr = torch.arange(rows, device=device)
    kind = cfg['decoder']
    state = torch.full((rows, 2 * n), -1, device=device, dtype=torch.long)
    if kind == 'ar':
        for step in range(2 * n):
            al, ol = model(expanded, state, torch.full((rows,), 2 * n - step, device=device))
            task = step // 2
            logits = (al[:, task] if step % 2 == 0 else ol[:, task]).float()
            state[:, task + (n if step % 2 else 0)] = draw(logits, 1.0, gen)
        return state.reshape(b, count, 2 * n)
    if kind == 'continuous':
        z = torch.randn((rows, n, m + n), device=device, generator=gen)
        for t in range(model.steps, 0, -1):
            pred = continuous_prediction(model(expanded, z, torch.full((rows,), t, device=device)))
            prev, cur = alpha_bar(torch.tensor([t - 1, t]), model.steps).double().tolist()
            a = cur / prev
            beta = 1 - a
            z = math.sqrt(prev) * beta / (1 - cur) * pred + math.sqrt(a) * (1 - prev) / (1 - cur) * z
            if t > 1:
                z += math.sqrt(max(0, beta * (1 - prev) / (1 - cur))) * torch.randn(z.shape, device=device, generator=gen)
        aa = z[..., :m].masked_fill(~expanded['eligible'], -1000000000.0).argmax(-1)
        order = torch.rand((rows, n), device=device, generator=gen).argsort(-1)
        oo = project_priority(z[..., m:], order, 0.0, gen)
        return torch.cat([aa, oo], -1).reshape(b, count, 2 * n)
    t0 = model.steps if start_t is None else start_t
    if not 1 <= t0 <= model.steps:
        raise ValueError('Invalid restart step')
    if initial is None and t0 != model.steps:
        raise ValueError('Initial solution required for partial restart')
    if initial is not None:
        state = corrupt(initial.repeat_interleave(count, 0), torch.full((rows,), t0, device=device), model.steps, cfg['kernel'], m, cfg['noise_schedule'], gen)
    elif cfg['kernel'] == 'uniform':
        state = torch.cat([torch.randint(m, (rows, n), device=device, generator=gen), torch.randint(n, (rows, n), device=device, generator=gen)], -1)
    ta = cfg['assignment_temperature']
    to = cfg['priority_temperature']
    uniform_order = torch.rand((rows, n), device=device, generator=gen).argsort(-1)
    for t in range(t0, 0, -1):
        al, ol = (v.float() for v in model(expanded, state, torch.full((rows,), t, device=device)))
        if cfg['kernel'] == 'uniform':
            pa = uniform_posterior(al.softmax(-1), state[:, :n], t, model.steps, cfg['noise_schedule'])
            po = uniform_posterior(ol.softmax(-1), state[:, n:], t, model.steps, cfg['noise_schedule'])
            aa = draw(pa.clamp_min(1e-30).log().masked_fill(~expanded['eligible'], -1000000000.0), ta, gen)
            oo = project_priority(po.clamp_min(1e-30).log(), uniform_order, to, gen)
            state = torch.cat([aa, oo], -1)
            continue
        reveal = (torch.rand(state.shape, device=device, generator=gen) < reveal_probability(t, model.steps, cfg['noise_schedule'])) & (state < 0)
        aa = draw(al, ta, gen)
        state[:, :n] = torch.where(reveal[:, :n], aa, state[:, :n])
        targets = reveal[:, n:]
        order = torch.rand(targets.shape, device=device, generator=gen).masked_fill(~targets, 2.0).argsort(-1)
        occupied = F.one_hot(state[:, n:].clamp_min(0), n).bool() & (state[:, n:] >= 0).unsqueeze(-1)
        used = occupied.any(1)
        for j in range(int(targets.sum(-1).max().item())):
            task = order[:, j]
            active = targets[rr, task]
            v = draw(ol[rr, task].masked_fill(used, -1000000000.0), to, gen)
            r = rr[active]
            v = v[active]
            state[r, n + task[active]] = v
            used[r, v] = True
    return state.reshape(b, count, 2 * n)

@torch.no_grad()
def solve(model, batch, comp, comm, edges, cfg, seed=42):
    if len(batch['ct']) != 1:
        raise ValueError('solve expects one instance')
    count = int(cfg['candidates'])
    chunk = int(cfg['candidate_chunk'])
    rounds = int(cfg['resampling_rounds']) if cfg['resampling'] else 1
    if min(count, chunk, rounds) < 1 or rounds > count:
        raise ValueError('Invalid candidate budget')
    if cfg['resampling'] and cfg['decoder'] != 'd3l':
        raise ValueError('Resampling is a discrete diffusion option')
    tokens = []
    scores = []
    generated = 0
    device = batch['ct'].device
    n = comp.shape[0]
    for round_id in range(rounds):
        budget = count // rounds + (round_id < count % rounds)
        parents = None
        if round_id:
            pool = np.concatenate(tokens)
            values = np.concatenate(scores)
            parents = pool[np.argsort(values, kind='stable')[:max(1, cfg['parents'])]]
        for left in range(0, budget, chunk):
            k = min(chunk, budget - left)
            if parents is None:
                y = sample(model, batch, k, seed + generated, cfg)[0]
            else:
                init = torch.as_tensor(parents[np.arange(left, left + k) % len(parents)], device=device)
                expanded = {name: value.expand(k, *value.shape[1:]) for name, value in batch.items()}
                y = sample(model, expanded, 1, seed + generated, cfg, initial=init, start_t=min(cfg['restart_step'], model.steps))[:, 0]
            arr = y.cpu().numpy()
            values = evaluate_population(comp, comm, edges, arr[:, :n].copy(), arr[:, n:].astype(np.float64))
            tokens.append(arr)
            scores.append(values)
            generated += k
    all_tokens = np.concatenate(tokens)
    all_scores = np.concatenate(scores)
    j = int(all_scores.argmin())
    token = all_tokens[j]
    mk, st, ft, order = decode(comp, comm, edges, token[:n].copy(), token[n:].astype(np.float64))
    return dict(makespan=float(mk), assignment=token[:n].tolist(), priorities=token[n:].tolist(), order=order.tolist(), start=st.tolist(), finish=ft.tolist(), candidates=generated)
