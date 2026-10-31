"""Batched Transformer-policy rollouts."""
import time
import numpy as np
import torch
from .transrl_env import FastSchedulingEnv as SchedulingEnv, prepare_template
from .transrl_policy import FastPolicy
import weakref
_POLICIES = weakref.WeakKeyDictionary()
from .common import schedule_json, sync

@torch.no_grad()
def rollouts(model, instances, seeds, store=True):
    model.eval()
    device = next(model.parameters()).device
    templates = {}
    for x in instances:
        if id(x) not in templates:
            templates[id(x)] = prepare_template(x)
    envs = [SchedulingEnv.from_template(templates[id(x)]) for x in instances]
    generators = [torch.Generator(device=device).manual_seed(int(s)) for s in seeds]
    rows = [[] for _ in envs]
    for step in range(envs[0].n):
        states = [e.observation_numpy() for e in envs]
        obs = torch.as_tensor(np.stack([x[0] for x in states]), device=device)
        loads = torch.as_tensor(np.stack([x[1] for x in states]), device=device)
        cache = _POLICIES.setdefault(model, {})
        if len(envs) not in cache:
            cache[len(envs)] = FastPolicy(model, len(envs), obs.shape[-1])
        actions, logps, value = cache[len(envs)].act(obs, loads, generators)
        acts = actions.cpu().tolist()
        for j, e in enumerate(envs):
            reward, done = e.step(acts[j])
            if store:
                rows[j].append((obs[j], loads[j], actions[j], logps[j], value[j], reward, done))
    return (envs, [x for trajectory in rows for x in trajectory])

def solve_batch(model, inst, budget=256, seed=42, chunk=32):
    device = next(model.parameters()).device
    sync(device)
    start = time.perf_counter()
    best = None
    seen = set()
    for left in range(0, budget, chunk):
        count = min(chunk, budget - left)
        envs, _ = rollouts(model, [inst] * count, list(range(seed + left, seed + left + count)), store=False)
        for env in envs:
            sched = env.schedule()
            seen.add((tuple(env.a), tuple(env.order)))
            if best is None or sched.makespan < best[1].makespan:
                best = (env.a.copy(), sched)
    sync(device)
    elapsed = time.perf_counter() - start
    return dict(makespan=best[1].makespan, online_seconds=elapsed, feasible=True, candidates=budget, distinct_candidates=len(seen), schedule=schedule_json(*best), seed=seed)
