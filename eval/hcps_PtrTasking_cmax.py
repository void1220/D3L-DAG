"""Pointer-network task and processor scheduling."""
from typing import Tuple
import torch
from torch import nn

class MLP(nn.Module):

    def __init__(self, in_dim: int, hidden: int, depth: int, dropout: float):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(max(0, depth - 1)):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, hidden))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class PtrTaskingFast(nn.Module):

    def __init__(self, feat_dim: int, hidden: int, mlp_depth: int, dropout: float):
        super().__init__()
        self.hidden = hidden
        self.task_enc = MLP(feat_dim, hidden, mlp_depth, dropout)
        self.proc_enc = MLP(feat_dim, hidden, mlp_depth, dropout)
        self.dec = nn.LSTMCell(hidden, hidden)
        self.start = nn.Parameter(torch.zeros(hidden))
        self.task_Wx = nn.Linear(hidden, hidden, bias=False)
        self.task_Wh = nn.Linear(hidden, hidden, bias=False)
        self.task_v = nn.Linear(hidden, 1, bias=False)
        self.proc_Wp = nn.Linear(hidden, hidden, bias=False)
        self.proc_Wh = nn.Linear(hidden, hidden, bias=False)
        self.proc_Wt = nn.Linear(hidden, hidden, bias=False)
        self.proc_v = nn.Linear(hidden, 1, bias=False)

    def encode(self, feats: torch.Tensor, N: int, P: int) -> Tuple[torch.Tensor, torch.Tensor]:
        task_x = feats[:, 1:1 + N, :]
        proc_x = feats[:, 1 + N:1 + N + P, :]
        B = feats.size(0)
        T = self.task_enc(task_x.reshape(B * N, -1)).reshape(B, N, -1)
        Pm = self.proc_enc(proc_x.reshape(B * P, -1)).reshape(B, P, -1)
        return (T, Pm)

    def init_state(self, B: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = self.start.unsqueeze(0).expand(B, -1).contiguous().to(device)
        c0 = torch.zeros_like(h0)
        return (h0, c0)

    def task_logits(self, T: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        u = torch.tanh(self.task_Wx(T) + self.task_Wh(h).unsqueeze(1))
        logits = self.task_v(u).squeeze(-1)
        return logits

    def proc_logits(self, Pm: torch.Tensor, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        u = torch.tanh(self.proc_Wp(Pm) + self.proc_Wh(h).unsqueeze(1) + self.proc_Wt(t_emb).unsqueeze(1))
        logits = self.proc_v(u).squeeze(-1)
        return logits

    def update(self, h: torch.Tensor, c: torch.Tensor, inp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dec(inp, (h, c))
import numpy as np
from torch.nn import functional as F

def smooth_ce(logits, target, mask=None, smoothing=0.01):
    logits = logits.float()
    if mask is not None:
        logits = logits.masked_fill(~mask, -1000000000.0)
    logp = F.log_softmax(logits, -1)
    nll = -logp.gather(1, target[:, None]).squeeze(1)
    if mask is None:
        smooth = -logp.mean(-1)
    else:
        smooth = -(logp * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    return ((1 - smoothing) * nll + smoothing * smooth).mean()

def loss(model, data, ids, mean, std, cfg):
    feats = (data['features'][ids] - mean) / std
    comp = data['comp'][ids]
    order = data['order'][ids]
    assign = data['assignment'][ids]
    ready = data['ready'][ids]
    b, n, m = comp.shape
    T, P = model.encode(feats, n, m)
    tx = model.task_Wx(T)
    px = model.proc_Wp(P)
    h, c = model.init_state(b, feats.device)
    batch = torch.arange(b, device=feats.device)
    lt = 0.0
    lp = 0.0
    lc = 0.0
    for step in range(n):
        task = order[:, step]
        proc = assign[batch, task]
        logits = model.task_v(torch.tanh(tx + model.task_Wh(h)[:, None])).squeeze(-1)
        lt = lt + smooth_ce(logits, task, ready[:, step], cfg['label_smoothing'])
        t = T[batch, task]
        pl = model.proc_v(torch.tanh(px + model.proc_Wh(h)[:, None] + model.proc_Wt(t)[:, None])).squeeze(-1)
        lp = lp + smooth_ce(pl, proc, smoothing=cfg['label_smoothing'])
        cost = comp[batch, task].float()
        lc = lc + (F.softmax(pl.float(), -1) * (cost / cost.mean(-1, keepdim=True))).sum(-1).mean()
        h, c = model.update(h, c, t + P[batch, proc])
    total = (cfg['lambda_task'] * lt + cfg['lambda_proc'] * lp + cfg['lambda_proc_time'] * lc) / n
    return total

@torch.no_grad()
def sample(model, features, edges, mean, std, k=256, chunk=64, temperature=0.9, task_topk=6, proc_topk=3, seed=42):
    device = next(model.parameters()).device
    model.eval()
    n = len(edges)
    m = len(features) - 1 - n
    f = torch.as_tensor(features, device=device, dtype=torch.float32)[None]
    T0, P0 = model.encode((f - mean) / std, n, m)
    tx0 = model.task_Wx(T0)
    px0 = model.proc_Wp(P0)
    e = torch.as_tensor(edges.astype(np.int64), device=device)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    allocations = []
    priorities = []
    sequences = []
    for left in range(0, k, chunk):
        b = min(chunk, k - left)
        T = T0.expand(b, -1, -1)
        P = P0.expand(b, -1, -1)
        tx = tx0.expand(b, -1, -1)
        px = px0.expand(b, -1, -1)
        h, c = model.init_state(b, device)
        deg = e.sum(0)[None].expand(b, -1).clone()
        done = torch.zeros((b, n), dtype=torch.bool, device=device)
        alloc = torch.empty((b, n), dtype=torch.int64, device=device)
        priority = torch.empty_like(alloc)
        sequence = torch.empty_like(alloc)
        bi = torch.arange(b, device=device)
        for step in range(n):
            ready = (deg == 0) & ~done
            tl = model.task_v(torch.tanh(tx + model.task_Wh(h)[:, None])).squeeze(-1).float().masked_fill(~ready, -1000000000.0)
            tv, ti = tl.topk(min(task_topk, n), dim=-1)
            task = ti.gather(1, torch.multinomial(F.softmax(tv / temperature, -1), 1, generator=rng)).squeeze(1)
            if left == 0:
                task[0] = tl[0].argmax()
            t = T[bi, task]
            pl = model.proc_v(torch.tanh(px + model.proc_Wh(h)[:, None] + model.proc_Wt(t)[:, None])).squeeze(-1).float()
            pv, pi = pl.topk(min(proc_topk, m), dim=-1)
            proc = pi.gather(1, torch.multinomial(F.softmax(pv / temperature, -1), 1, generator=rng)).squeeze(1)
            if left == 0:
                proc[0] = pl[0].argmax()
            alloc[bi, task] = proc
            priority[bi, task] = step
            sequence[:, step] = task
            done[bi, task] = True
            deg -= e[task]
            h, c = model.update(h, c, t + P[bi, proc])
        allocations.append(alloc)
        priorities.append(priority)
        sequences.append(sequence)
    a = torch.cat(allocations).cpu().numpy()
    p = torch.cat(priorities).cpu().numpy().astype(np.float64)
    s = torch.cat(sequences).cpu().numpy()
    i, j = np.nonzero(edges)
    return (a, p)
