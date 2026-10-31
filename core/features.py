"""Processor-conditioned structural features."""
import numpy as np

def structural_features(inst):
    n = inst.num_tasks
    m = inst.num_procs
    ct = np.asarray(inst.comp_time)
    comm = np.asarray(inst.comm_time)
    deg = np.array([len(inst.predecessors[i]) for i in range(n)])
    ready = list(np.flatnonzero(deg == 0))
    order = []
    level = np.zeros(n)
    while ready:
        t = ready.pop()
        order.append(t)
        for c in inst.successors[t]:
            level[c] = max(level[c], level[t] + 1)
            deg[c] -= 1
            if deg[c] == 0:
                ready.append(c)
    different = 1 - np.eye(m)
    up = ct.copy()
    down = ct.copy()
    reach = np.zeros((n, n), bool)
    for t in order:
        for p in inst.predecessors[t]:
            up[t] = np.maximum(up[t], ct[t] + (up[p, :, None] + comm[p, t] * different).min(0))
            reach[:, t] |= reach[:, p]
            reach[p, t] = True
    for t in reversed(order):
        for c in inst.successors[t]:
            down[t] = np.maximum(down[t], ct[t] + (down[c, None, :] + comm[t, c] * different).min(1))
    scale = max(ct.mean(), 1e-06)
    path = up + down - ct
    pair = np.stack([np.log1p(up / scale), np.log1p(down / scale), np.log1p((path - path.min(1, keepdims=True)) / scale)], -1)
    u = up.min(1)
    d = down.min(1)
    lb = max(u.max(), 1e-06)
    width = np.array([(level == x).sum() / n for x in level])
    extra = np.stack([u / lb, d / lb, np.maximum(0, lb - path.min(1)) / lb, width, reach.sum(0) / n, reach.sum(1) / n], -1)
    return (extra.astype(np.float32), pair.astype(np.float32), np.stack([reach, reach.T], -1).astype(np.float32))
