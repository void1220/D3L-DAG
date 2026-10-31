"""Relation-biased diffusion Transformer and optional condition encoders."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class GraphBlock(nn.Module):

    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads, self.dim, self.dropout = (heads, width // heads, dropout)
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.rel = nn.Parameter(torch.randn(heads, 10) * 0.1)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4 * width), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * width, width))

    def attend(self, h, rel):
        b, n, d = h.shape
        q, k, v = self.qkv(self.norm(h)).reshape(b, n, 3, self.heads, self.dim).permute(2, 0, 3, 1, 4)
        q = F.normalize(q, dim=-1) * math.sqrt(self.dim)
        k = F.normalize(k, dim=-1) * math.sqrt(self.dim)
        bias = torch.einsum('bijn,hn->bhij', rel, self.rel).to(q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0.0)
        h = h + F.dropout(self.out(out.transpose(1, 2).reshape(b, n, d)), self.dropout, self.training)
        return h

    def forward(self, h, rel):
        h = self.attend(h, rel)
        return h + F.dropout(self.ff(h), self.dropout, self.training)

class GraphMaskModel(nn.Module):

    def __init__(self, n, m, width=256, layers=6, heads=8, dropout=0.05, steps=30, noise_schedule='linear', feature_dim=17, feature_mode='full', cross_attention=False):
        super().__init__()
        self.n, self.m, self.width, self.steps = (n, m, width, steps)
        self.noise_schedule = noise_schedule
        self.spec = dict(n=n, m=m, width=width, layers=layers, heads=heads, dropout=dropout, steps=steps, noise_schedule=noise_schedule, feature_dim=feature_dim, feature_mode=feature_mode, cross_attention=cross_attention)
        self.feature_dim = feature_dim
        self.feature_mode = feature_mode
        self.cross_attention = cross_attention
        self.unmasked_priority_logits = True
        if feature_dim not in (13, 17):
            raise ValueError(feature_dim)
        self.task = nn.Sequential(nn.Linear(feature_dim, width), nn.GELU(), nn.Linear(width, width))
        self.proc = nn.Sequential(nn.Linear(6, width), nn.GELU(), nn.Linear(width, width))
        self.pair = nn.Linear(14, 64)
        self.pair_task, self.pair_proc = (nn.Linear(width, 64), nn.Linear(width, 64))
        self.pair_out = nn.Sequential(nn.GELU(), nn.Linear(64, 1))
        self.proc_type = nn.Parameter(torch.randn(width) * 0.02)
        self.rank = nn.Embedding(n + 1, width)
        self.mask_a = nn.Parameter(torch.randn(width) * 0.02)
        self.time = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([GraphBlock(width, heads, dropout) for _ in range(layers)])
        self.final = nn.LayerNorm(width)
        self.aquery, self.pkey = (nn.Linear(width, width), nn.Linear(width, width))
        self.priority = nn.Linear(width, n)
        if feature_mode == 'gcn':
            del self.task, self.proc
            self.raw_encoder = RawGCN(width)
            self.pair = nn.Linear(1, 64)
        if cross_attention:
            self.condition_norm = nn.LayerNorm(width)
            self.condition_global = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))
            self.cross_norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
            self.cross_layers = nn.ModuleList([nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True) for _ in range(layers)])

    def forward(self, batch, state, t):
        if self.feature_mode == 'gcn':
            batch = {k: batch[k] for k in ('ct', 'comm', 'edge', 'eligible')}
        ct, comm, edge = (batch['ct'], batch['comm'], batch['edge'])
        b, n, m = ct.shape
        scale = ct.mean((1, 2), keepdim=True).clamp_min(1e-06)
        cost = torch.log1p(ct / scale)
        delay = torch.log1p(comm / scale)
        mode = getattr(self, 'feature_mode', 'full')
        if mode != 'gcn':
            indeg = edge.sum(1) / max(1, n - 1)
            outdeg = edge.sum(2) / max(1, n - 1)
            task = torch.stack([cost.mean(2), cost.amin(2), cost.amax(2), cost.std(2, unbiased=False), indeg, outdeg, delay.sum(1) / n, delay.sum(2) / n], -1)
            proc = torch.stack([cost.mean(1), cost.amin(1), cost.amax(1), cost.std(1, unbiased=False)], -1)
            task = torch.cat([task, batch['extra']], -1)
        continuous = state.ndim == 3
        if continuous:
            za, zo = (state[..., :m], state[..., m:])
            known = torch.softmax(za.float(), -1)
        else:
            a, o = (state[..., :n], state[..., n:])
            known = F.one_hot(a.clamp_min(0), m).float() * (a >= 0).unsqueeze(-1)
        if mode != 'gcn':
            work = (known * ct / scale).sum(1) / n
            proc = torch.cat([proc, torch.log1p(work).unsqueeze(-1), (known.sum(1) / n).unsqueeze(-1)], -1)
        mode = getattr(self, 'feature_mode', 'full')
        if mode == 'gcn':
            task_h, ph = self.raw_encoder(cost, delay, edge, batch['eligible'])
        else:
            if mode != 'full':
                task = task.clone()
                task[..., 8:] = 0
                if mode == 'position':
                    task[..., 8:9] = batch['extra'][..., 0:1]
                    task[..., 14:17] = batch['extra'][..., 6:9]
                proc = proc.clone()
                proc[..., 4:] = 0
            ph = self.proc(proc)
            task_input = task if self.feature_dim == 17 else task[..., [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13]]
            task_h = self.task(task_input)
        if continuous:
            ah = (za / math.sqrt(m)).to(ph.dtype) @ ph
            rank_h = (zo / math.sqrt(n)).to(self.rank.weight.dtype) @ self.rank.weight[:n]
        else:
            ah = ph.gather(1, a.clamp_min(0).unsqueeze(-1).expand(-1, -1, self.width))
            ah = torch.where((a < 0).unsqueeze(-1), self.mask_a, ah)
            rank_h = self.rank(torch.where(o < 0, n, o))
        tt = t.float() / self.steps
        te = self.time(torch.stack([tt, torch.sin(math.pi * tt), torch.cos(math.pi * tt)], -1))
        h = task_h + ah + rank_h + te[:, None]
        rel = ct.new_zeros((b, n + m, n + m, 10))
        if mode in ('full', 'position'):
            rel[:, :n, :n, 8:] = batch['reach']
        rel[:, :n, :n, :4] = torch.stack([edge, edge.transpose(1, 2), delay, delay.transpose(1, 2)], -1)
        rel[:, :n, n:, 4] = cost
        rel[:, n:, :n, 5] = cost.transpose(1, 2)
        rel[:, :n, n:, 6] = known
        rel[:, n:, :n, 7] = known.transpose(1, 2)
        h = torch.cat([h, ph + self.proc_type + te[:, None]], 1)
        condition = None
        if self.cross_attention:
            global_features = torch.stack([torch.log1p(ct.mean((1, 2))), torch.log1p(comm.mean((1, 2))), torch.full_like(ct[:, 0, 0], math.log1p(n)), torch.full_like(ct[:, 0, 0], math.log1p(m))], -1)
            condition = self.condition_norm(torch.cat([task_h, ph, self.condition_global(global_features)[:, None]], 1))
        for i, block in enumerate(self.blocks):
            if not self.cross_attention:
                h = block(h, rel)
            else:
                h = block.attend(h, rel)
                q = self.cross_norms[i](h)
                h = h + self.cross_layers[i](q, condition, condition, need_weights=False)[0]
                h = h + F.dropout(block.ff(h), block.dropout, self.training)
        h = self.final(h)
        h, ph = (h[:, :n], h[:, n:])
        with torch.autocast(device_type=ct.device.type, enabled=False):
            h, ph = (h.float(), ph.float())
            if mode == 'gcn':
                pair = cost.unsqueeze(-1)
            else:
                incoming = comm.transpose(1, 2) / scale
                outgoing = comm / scale
                kin = known.sum(1)[:, None, :].expand(-1, n, -1) / n
                same_in = incoming @ known
                same_out = outgoing @ known
                pred_cost = edge.transpose(1, 2) @ cost / edge.sum(1).clamp_min(1).unsqueeze(-1)
                succ_cost = edge @ cost / edge.sum(2).clamp_min(1).unsqueeze(-1)
                pair = torch.stack([cost, cost - cost.amin(2, keepdim=True), cost - cost.mean(2, keepdim=True), torch.log1p(same_in), torch.log1p(same_out), kin, pred_cost, succ_cost], -1)
                cross_in = (incoming.sum(-1, keepdim=True) - same_in).clamp_min(0)
                cross_out = (outgoing.sum(-1, keepdim=True) - same_out).clamp_min(0)
                dynamic = torch.stack([torch.log1p(work)[:, None, :].expand(-1, n, -1), torch.log1p(cross_in), torch.log1p(cross_out)], -1)
                pair = torch.cat([pair, batch['path'], dynamic], -1) if mode == 'full' else torch.cat([pair, ct.new_zeros((b, n, m, 6))], -1)
                if mode != 'full':
                    pair = pair.clone()
                    pair[..., 3:] = 0
            q = F.normalize(self.aquery(h), dim=-1)
            key = F.normalize(self.pkey(ph), dim=-1)
            allocation = 16 * torch.einsum('bnd,bmd->bnm', q, key)
            pair_h = self.pair(pair) + self.pair_task(h)[:, :, None] + self.pair_proc(ph)[:, None]
            allocation = allocation + 8 * torch.tanh(self.pair_out(pair_h).squeeze(-1) / 8)
            allocation = allocation - allocation.mean(-1, keepdim=True)
            priority = self.priority(h)
        allocation = allocation.masked_fill(~batch['eligible'], -1000000000.0)
        if not continuous and (not getattr(self, 'unmasked_priority_logits', False)):
            occupied = F.one_hot(o.clamp_min(0), n).bool() & (o >= 0).unsqueeze(-1)
            priority = priority.masked_fill(occupied.any(1)[:, None], -1000000000.0)
        return (allocation, priority)

class RawGCN(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.raw = nn.Linear(2, width)
        self.layers = nn.ModuleList([nn.Linear(width, width), nn.Linear(width, width)])

    def forward(self, cost, delay, edge, eligible):
        z = torch.relu(self.raw(torch.stack([cost, eligible.to(cost.dtype)], -1)))
        task, proc = (z.mean(2), z.mean(1))
        a = edge * (1 + delay)
        a = a + a.transpose(1, 2) + torch.eye(a.shape[-1], device=a.device)[None]
        d = a.sum(-1).clamp_min(1e-06).rsqrt()
        a = a * d[:, :, None] * d[:, None, :]
        for layer in self.layers:
            task = torch.relu(layer(a.to(task.dtype) @ task))
            proc = torch.relu(layer(proc))
        return (task, proc)
