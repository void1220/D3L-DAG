"""CVAE-Opt adaptation for DAG scheduling."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .common import condition, sanitize, decode
from .cvae_layers import Encoder, Pointer, Attention, Embedding

def prepare(inst):
    inst = sanitize(inst)
    n, m = (inst.num_tasks, inst.num_procs)
    task = condition(inst)['features'][0]
    task = task.repeat_interleave(m, dim=0)
    proc = torch.eye(m).repeat(n, 1)
    cost = torch.tensor(np.log1p(inst.comp_time / inst.comp_time.mean()), dtype=torch.float32).reshape(-1, 1)
    feat = torch.cat((task, proc, cost, torch.zeros(n * m, 1)), dim=1)
    start = torch.zeros(1, feat.shape[1])
    start[0, -1] = 1
    pred = torch.zeros(n, n, dtype=torch.bool)
    for t, ps in inst.predecessors.items():
        pred[t, ps] = True
    return dict(features=torch.cat((start, feat))[None], pred=pred[None], eligible=torch.tensor(inst.eligibility_mask)[None])

def target(inst, tokens):
    a, sched = (np.asarray(tokens[:inst.num_tasks]), decode(inst, tokens))
    return torch.tensor([0] + [1 + int(t) * inst.num_procs + int(a[t]) for t in sched.order], dtype=torch.long)

def to_tokens(pairs, n, m):
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.shape != (n,) or (pairs < 1).any() or (pairs > n * m).any():
        raise ValueError('invalid pair sequence')
    tasks, processors = ((pairs - 1) // m, (pairs - 1) % m)
    if not np.array_equal(np.sort(tasks), np.arange(n)):
        raise ValueError('duplicate/missing task')
    assignment = np.empty(n, dtype=np.int64)
    ranks = np.empty(n, dtype=np.int64)
    assignment[tasks] = processors
    ranks[tasks] = np.arange(n)
    return np.concatenate((assignment, ranks))

def batch(items, device):
    return {k: torch.cat([x[k] for x in items], 0).to(device) for k in items[0]}

class CVAEBase(nn.Module):

    def __init__(self, n, m, width=128, latent=100):
        super().__init__()
        self.spec = dict(n=n, m=m, width=width, latent=latent)
        self.n, self.m, self.latent = (n, m, latent)
        self.instance_embedding = Embedding(3 * m + 4 * n + 2, width)
        reference = Embedding(3 * m + 4 * n + 2, width)
        attention = Attention(width)
        rnn = nn.GRU(width, width, 1, batch_first=True, dropout=0)
        self.encoder = Encoder(self.instance_embedding, reference, attention, rnn, None, latent, width)
        self.pointer = Pointer(attention, rnn, width, latent)
        self.reference_embedding = reference
        for p in self.parameters():
            if p.ndim > 1:
                nn.init.xavier_uniform_(p)

    def posterior(self, data, solution):
        return self.encoder(data['features'], solution, self.instance_embedding(data['features']), SimpleNamespace())

    def rollout(self, data, z, solution=None, hidden=None):
        features = data['features']
        b = z.shape[0]
        if hidden is None:
            hidden = self.instance_embedding(features)
        features = features.expand(b, -1, -1)
        hidden = hidden.expand(b, -1, -1)
        pred = data['pred'].expand(b, -1, -1)
        eligible = data['eligible'].expand(b, -1, -1)
        visited = torch.zeros(b, self.n, dtype=torch.bool, device=z.device)
        ptr = torch.zeros(b, dtype=torch.long, device=z.device)
        hh = None
        actions = []
        logps = []
        rows = torch.arange(b, device=z.device)
        for step in range(self.n):
            reference = self.reference_embedding(features[rows, ptr][:, None])
            logits, hh = self.pointer(hidden, reference, z, hh)
            ready = ~visited & ~(pred & ~visited[:, None, :]).any(-1)
            legal = (ready[:, :, None] & eligible).flatten(1)
            mask = torch.cat((torch.zeros(b, 1, dtype=torch.bool, device=z.device), legal), 1)
            logits = logits.masked_fill(~mask, -torch.inf)
            if solution is None:
                ptr = logits.argmax(-1)
            else:
                ptr = solution[:, step + 1]
                if not mask[rows, ptr].all():
                    raise ValueError('teacher sequence violates DTS feasibility')
                logps.append(F.log_softmax(logits, -1)[rows, ptr])
            actions.append(ptr)
            visited = visited.scatter(1, ((ptr - 1) // self.m)[:, None], True)
        return (torch.stack(actions, 1), torch.stack(logps, 1) if logps else None)

    def loss(self, data, solution, beta):
        z, mu, logvar = self.posterior(data, solution)
        _, logp = self.rollout(data, z, solution)
        rc = -logp.sum()
        kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum()
        return (rc + beta * kl, dict(rc=float(rc.detach()) / len(solution), kl=float(kl.detach()) / len(solution)))

class HistoryCounts:

    def __init__(self, data, b, n, m):
        self.n, self.m = (n, m)
        self.pred = data['pred'].expand(b, -1, -1)
        self.degree = self.pred.sum(-1).clamp_min(1).float()
        self.count = torch.zeros(b, m, device=self.pred.device)
        self.parents = torch.zeros(b, n, m, device=self.pred.device)
        self.rows = torch.arange(b, device=self.pred.device)

    @torch.no_grad()
    def features(self):
        return torch.stack((self.count[:, None, :].expand(-1, self.n, -1) / self.n, self.parents / self.degree[:, :, None]), -1)

    @torch.no_grad()
    def update(self, ptr):
        task, proc = ((ptr - 1) // self.m, (ptr - 1) % self.m)
        selected = F.one_hot(proc, self.m).float()
        edge = self.pred[self.rows, :, task].float()
        self.count = self.count + selected
        self.parents = self.parents + edge[:, :, None] * selected[:, None, :]

class CVAEOptDTS(CVAEBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_head = nn.Sequential(nn.Linear(2, self.spec['width']), nn.ReLU(), nn.Linear(self.spec['width'], 1))
        nn.init.zeros_(self.history_head[-1].weight)
        nn.init.zeros_(self.history_head[-1].bias)

    def rollout(self, data, z, solution=None, hidden=None):
        features = data['features']
        b = z.shape[0]
        if hidden is None:
            hidden = self.instance_embedding(features)
        features = features.expand(b, -1, -1)
        hidden = hidden.expand(b, -1, -1)
        pred = data['pred'].expand(b, -1, -1)
        eligible = data['eligible'].expand(b, -1, -1)
        visited = torch.zeros(b, self.n, dtype=torch.bool, device=z.device)
        ptr = torch.zeros(b, dtype=torch.long, device=z.device)
        rows = torch.arange(b, device=z.device)
        hh = None
        actions = []
        logps = []
        state = HistoryCounts(data, b, self.n, self.m)
        for step in range(self.n):
            reference = self.reference_embedding(features[rows, ptr][:, None])
            logits, hh = self.pointer(hidden, reference, z, hh)
            summary = state.features()
            extra = self.history_head(summary).squeeze(-1).flatten(1)
            logits = logits + torch.cat((extra.new_zeros(b, 1), extra), 1)
            ready = ~visited & ~(pred & ~visited[:, None, :]).any(-1)
            legal = (ready[:, :, None] & eligible).flatten(1)
            mask = torch.cat((torch.zeros(b, 1, dtype=torch.bool, device=z.device), legal), 1)
            logits = logits.masked_fill(~mask, -torch.inf)
            if solution is None:
                ptr = logits.argmax(-1)
            else:
                ptr = solution[:, step + 1]
                if not mask[rows, ptr].all():
                    raise ValueError('teacher sequence violates DTS feasibility')
                selected = F.log_softmax(logits, -1)[rows, ptr]
                logps.append(selected)
            state.update(ptr)
            actions.append(ptr)
            visited = visited.scatter(1, ((ptr - 1) // self.m)[:, None], True)
        return (torch.stack(actions, 1), torch.stack(logps, 1) if logps else None)
