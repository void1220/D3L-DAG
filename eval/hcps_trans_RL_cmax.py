"""Transformer policy and PPO scheduling."""
from __future__ import annotations
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .common import sanitize, sync, schedule_json

class SchedulingEnv:
    routing_rules = ('earliest_finish', 'shortest_processing', 'earliest_available', 'minimum_incoming_communication')
    sequencing_rules = ('shortest_processing', 'longest_processing', 'upward_rank', 'most_descendants', 'lowest_id')

    def __init__(self, instance):
        self.inst = sanitize(instance)
        self.n, self.m = (self.inst.num_tasks, self.inst.num_procs)
        self.a = np.full(self.n, -1, dtype=np.int64)
        self.start = np.zeros(self.n)
        self.finish = np.zeros(self.n)
        self.avail = np.zeros(self.m)
        self.order = []
        self.scale = max(float(self.inst.comp_time.mean()), 1e-08)
        self.reward_scale = max(float(self.inst.comp_time.min(1).sum() / self.m), 1.0)
        self.rank = np.zeros(self.n)
        descendants = [set() for _ in range(self.n)]
        pending = set(range(self.n))
        while pending:
            ready = [t for t in sorted(pending) if all((s not in pending for s in self.inst.successors[t]))]
            for t in ready:
                ss = self.inst.successors[t]
                self.rank[t] = self.inst.comp_time[t].mean() + max((self.inst.comm_time[t, s] + self.rank[s] for s in ss), default=0.0)
                for s in ss:
                    descendants[t].update({s} | descendants[s])
                pending.remove(t)
        self.desc = np.array([len(s) for s in descendants])

    def ready(self):
        return [t for t in range(self.n) if self.a[t] < 0 and all((self.a[p] >= 0 for p in self.inst.predecessors[t]))]

    def release(self, t, p):
        return max((self.finish[q] + (self.inst.comm_time[q, t] if self.a[q] != p else 0.0) for q in self.inst.predecessors[t]), default=0.0)

    def observation(self, device):
        ready = self.ready()
        e = self.inst.eligibility_mask
        edge = np.zeros((self.n, self.n))
        for t, ps in self.inst.predecessors.items():
            edge[ps, t] = 1
        costs = np.concatenate([np.log1p(self.inst.comp_time / self.scale), e, np.log1p(self.inst.comm_time / self.scale), np.log1p(self.inst.comm_time.T / self.scale), edge, edge.T], axis=1)
        dynamic = np.stack([self.a >= 0, np.isin(np.arange(self.n), ready), self.start / self.scale, self.finish / self.scale, (self.a + 1) / self.m, self.rank / max(self.rank.max(), 1), self.desc / max(self.n, 1)], 1)
        features = np.concatenate([costs, dynamic], 1)
        return (torch.tensor(features, dtype=torch.float32, device=device)[None], torch.tensor(self.avail / self.scale, dtype=torch.float32, device=device)[None])

    def step(self, actions):
        route, sequence = map(int, actions)
        if not 0 <= route < 4 or not 0 <= sequence < 5:
            raise ValueError('rule action')
        ready = self.ready()
        if not ready:
            raise ValueError('no schedulable task')
        procs = {}
        for t in ready:
            choices = []
            for p in np.flatnonzero(self.inst.eligibility_mask[t]):
                release = self.release(t, p)
                end = max(self.avail[p], release) + self.inst.comp_time[t, p]
                incoming = sum((self.inst.comm_time[q, t] for q in self.inst.predecessors[t] if self.a[q] != p))
                score = (end, self.inst.comp_time[t, p], self.avail[p], incoming)[route]
                choices.append((score, end, int(p)))
            procs[t] = min(choices)[2]

        def key(t):
            dur = self.inst.comp_time[t, procs[t]]
            return ((dur, -dur, -self.rank[t], -self.desc[t], t)[sequence], t)
        t = min(ready, key=key)
        p = procs[t]
        old = float(self.finish.max())
        self.a[t] = p
        self.start[t] = max(self.avail[p], self.release(t, p))
        self.finish[t] = self.start[t] + self.inst.comp_time[t, p]
        self.avail[p] = self.finish[t]
        self.order.append(t)
        reward = -(float(self.finish.max()) - old) / self.reward_scale
        return (reward, len(self.order) == self.n)

    def schedule(self):
        from core.decoding import DecodedSchedule
        if len(self.order) != self.n:
            raise ValueError('unfinished schedule')
        return DecodedSchedule(np.array(self.order), self.start.copy(), self.finish.copy(), float(self.finish.max()))

class MultiAgentTransformer(nn.Module):

    def __init__(self, n, m, width=128, layers=2, heads=4):
        super().__init__()
        self.spec = dict(n=n, m=m, width=width, layers=layers, heads=heads)
        self.n, self.m = (n, m)
        self.input = nn.Linear(2 * m + 4 * n + 7, width)
        self.load = nn.Linear(m, width)
        self.roles = nn.Parameter(torch.randn(2, width) * 0.02)
        enc = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.action_embed = nn.Embedding(6, width)
        dec = nn.TransformerDecoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec, layers)
        self.reasoning = nn.Linear(width, width)
        self.actors = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, k)) for k in (4, 5)])
        self.critic = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, obs, loads, actions):
        tasks = self.input(obs)
        role = self.roles[None] + self.load(loads)[:, None]
        memory = self.encoder(torch.cat([tasks, role], 1))
        role_mem = memory[:, -2:]
        shifted = torch.full((len(obs), 2), 5, dtype=torch.long, device=obs.device)
        shifted[:, 1] = actions[:, 0]
        q = self.action_embed(shifted) + role_mem
        mask = torch.triu(torch.ones(2, 2, device=obs.device, dtype=torch.bool), diagonal=1)
        reasoning = self.reasoning(self.decoder(q, memory, tgt_mask=mask))
        logits = [head(reasoning[:, i]) for i, head in enumerate(self.actors)]
        value = self.critic(role_mem.mean(1)).squeeze(-1)
        return (logits, value)

    @torch.no_grad()
    def act(self, obs, loads, generator, greedy=False):
        actions = torch.zeros((len(obs), 2), dtype=torch.long, device=obs.device)
        lps = []
        for role in range(2):
            logits, value = self(obs, loads, actions)
            lp = logits[role].log_softmax(-1)
            a = lp.argmax(-1) if greedy else torch.multinomial(lp.exp(), 1, generator=generator).squeeze(-1)
            actions[:, role] = a
            lps.append(lp.gather(1, a[:, None]).squeeze(1))
        return (actions, torch.stack(lps, 1), value)

    def evaluate_actions(self, obs, loads, actions):
        logits, values = self(obs, loads, actions)
        lps = []
        ent = []
        for i, l in enumerate(logits):
            p = l.log_softmax(-1)
            lps.append(p.gather(1, actions[:, i:i + 1]).squeeze(1))
            ent.append(-(p.exp() * p).sum(-1))
        return (torch.stack(lps, 1), values, torch.stack(ent, 1).mean(1))

def rollout(model, inst, seed, greedy=False):
    model.eval()
    device = next(model.parameters()).device
    env = SchedulingEnv(inst)
    rng = torch.Generator(device=device).manual_seed(seed)
    rows = []
    while len(env.order) < env.n:
        obs, loads = env.observation(device)
        actions, lp, value = model.act(obs, loads, rng, greedy)
        reward, done = env.step(actions[0].tolist())
        rows.append((obs[0], loads[0], actions[0], lp[0], value[0], reward, done))
    return (env, rows)

def gae(rewards, values, dones, gamma=1.0, lam=0.95):
    adv = torch.zeros_like(rewards)
    last = torch.zeros((), device=rewards.device)
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        live = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * live - values[t]
        last = delta + gamma * lam * live * last
        adv[t] = last
    return (adv, adv + values)

def ppo_update(model, optimizer, rows, epochs=4, minibatch=128, clip=0.2, entropy=0.01):
    if not rows:
        raise ValueError('empty rollout')
    device = next(model.parameters()).device
    obs, loads, act, old_lp, values = [torch.stack([r[i] for r in rows]).detach() for i in range(5)]
    rewards = torch.tensor([r[5] for r in rows], device=device)
    dones = torch.tensor([r[6] for r in rows], device=device, dtype=torch.float32)
    advantages, returns = gae(rewards, values, dones)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-06)
    model.train()
    stats = []
    for _ in range(epochs):
        ids = torch.randperm(len(rows), device=device)
        for idx in ids.split(minibatch):
            lp, v, en = model.evaluate_actions(obs[idx], loads[idx], act[idx])
            ratio = (lp - old_lp[idx]).exp()
            adv = advantages[idx, None]
            policy = -torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
            vclip = values[idx] + (v - values[idx]).clamp(-clip, clip)
            value = 0.5 * torch.maximum((v - returns[idx]).square(), (vclip - returns[idx]).square()).mean()
            loss = policy + 0.5 * value - entropy * en.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            stats.append(dict(loss=float(loss.detach()), policy=float(policy.detach()), value=float(value.detach()), grad_norm=float(norm)))
    return stats[-1]
