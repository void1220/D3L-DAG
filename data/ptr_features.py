"""Pointer-scheduling instance features."""
from typing import Dict
import numpy as np

def _topo_levels_from_preds(num_tasks: int, predecessors: Dict[int, list]) -> np.ndarray:
    indeg = np.zeros(num_tasks, dtype=np.int32)
    succ = [[] for _ in range(num_tasks)]
    for v in range(num_tasks):
        preds = predecessors.get(v, [])
        indeg[v] = len(preds)
        for u in preds:
            if 0 <= u < num_tasks:
                succ[u].append(v)
    from collections import deque
    q = deque([i for i in range(num_tasks) if indeg[i] == 0])
    level = np.zeros(num_tasks, dtype=np.float32)
    seen = 0
    while q:
        u = q.popleft()
        seen += 1
        for v in succ[u]:
            lv = level[u] + 1.0
            if lv > level[v]:
                level[v] = lv
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if seen < num_tasks:
        for i in range(num_tasks):
            if indeg[i] > 0:
                level[i] = float(indeg[i])
    return level

def _fourier_id_embed(x_norm: np.ndarray, n_freq: int=2) -> np.ndarray:
    x = np.asarray(x_norm, dtype=np.float32)
    feats = []
    for k in range(1, n_freq + 1):
        ang = 2.0 * np.pi * k * x
        feats.append(np.sin(ang))
        feats.append(np.cos(ang))
    return np.stack(feats, axis=-1).astype(np.float32)

def _critical_path_tlevel_blevel(N: int, predecessors: Dict[int, list], successors: Dict[int, list], node_w: np.ndarray, comm_w: np.ndarray):
    indeg = np.zeros(N, dtype=np.int32)
    for v in range(N):
        indeg[v] = len(predecessors.get(v, []))
    from collections import deque
    q = deque([i for i in range(N) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in successors.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) < N:
        order = list(range(N))
    tlevel = np.zeros(N, dtype=np.float32)
    for u in order:
        best = 0.0
        for p in predecessors.get(u, []):
            if 0 <= p < N:
                best = max(best, float(tlevel[p] + comm_w[p, u]))
        tlevel[u] = float(node_w[u] + best)
    blevel = np.zeros(N, dtype=np.float32)
    for u in reversed(order):
        best = 0.0
        for v in successors.get(u, []):
            if 0 <= v < N:
                best = max(best, float(comm_w[u, v] + blevel[v]))
        blevel[u] = float(node_w[u] + best)
    return (tlevel, blevel)

def extract_instance_features_simple(inst, *, max_tasks: int, max_procs: int, topk_ct: int=14) -> np.ndarray:
    N, M = (int(inst.num_tasks), int(inst.num_procs))
    max_tasks = N
    max_procs = M
    ct = np.asarray(inst.comp_time, dtype=np.float32)
    comm = np.asarray(inst.comm_time, dtype=np.float32)
    ct_log = np.log1p(np.maximum(ct, 0.0))
    comm_log = np.log1p(np.maximum(comm, 0.0))
    indeg = np.array([len(inst.predecessors.get(i, [])) for i in range(N)], dtype=np.float32)
    outdeg = np.array([len(inst.successors.get(i, [])) for i in range(N)], dtype=np.float32)
    level = _topo_levels_from_preds(N, inst.predecessors)
    succ = {i: list(inst.successors.get(i, [])) for i in range(N)}
    preds = {i: list(inst.predecessors.get(i, [])) for i in range(N)}
    num_edges = int((comm > 0).sum())
    edge_den = float(num_edges) / float(max(1, N * (N - 1)))
    static_p = np.array([p.static_power for p in inst.processors], dtype=np.float32) if M > 0 else np.zeros((0,), np.float32)
    dyn_c = np.array([p.dyn_coeff for p in inst.processors], dtype=np.float32) if M > 0 else np.zeros((0,), np.float32)
    fmax = np.array([p.f_levels[-1] for p in inst.processors], dtype=np.float32) if M > 0 else np.zeros((0,), np.float32)
    static_log = np.log1p(np.maximum(static_p, 0.0))
    dyn_log = np.log1p(np.maximum(dyn_c, 0.0))
    fmax_log = np.log1p(np.maximum(fmax, 0.0))
    if M <= 1:
        proc_id_norm = np.zeros((max(M, 1),), dtype=np.float32)
    else:
        proc_id_norm = np.arange(M, dtype=np.float32) / float(M - 1)
    proc_id_fourier = _fourier_id_embed(proc_id_norm, n_freq=2)
    if N > 0:
        cm = comm_log[:N, :N]
        out_sum = cm.sum(axis=1)
        in_sum = cm.sum(axis=0)
        out_max = cm.max(axis=1)
        in_max = cm.max(axis=0)
        out_deg_comm = (comm[:N, :N] > 0).sum(axis=1).astype(np.float32)
        in_deg_comm = (comm[:N, :N] > 0).sum(axis=0).astype(np.float32)
        deg_norm = max(1.0, float(N - 1))
        out_deg_norm = out_deg_comm / deg_norm
        in_deg_norm = in_deg_comm / deg_norm
    else:
        out_sum = in_sum = out_max = in_max = out_deg_norm = in_deg_norm = np.zeros((0,), np.float32)
    if N > 0 and M > 0:
        node_w = ct_log.mean(axis=1)
    else:
        node_w = np.zeros((N,), dtype=np.float32)
    tlevel, blevel = _critical_path_tlevel_blevel(N=N, predecessors=preds, successors=succ, node_w=node_w.astype(np.float32), comm_w=comm_log[:N, :N].astype(np.float32) if N > 0 else np.zeros((0, 0), np.float32))
    tmax = float(np.max(tlevel)) if tlevel.size else 1.0
    bmax = float(np.max(blevel)) if blevel.size else 1.0
    tlevel_norm = (tlevel / max(1e-06, tmax)).astype(np.float32)
    blevel_norm = (blevel / max(1e-06, bmax)).astype(np.float32)
    crit_norm = (blevel / max(1e-06, float(np.max(blevel)))).astype(np.float32) if blevel.size else np.zeros((N,), np.float32)
    Dg = 8
    task_base = 22
    per_cand = 7
    Dt = task_base + per_cand * topk_ct
    Dp = 12
    D = max(Dg, Dt, Dp)
    L = 1 + max_tasks + max_procs
    feats = np.zeros((L, D), dtype=np.float32)
    feats[0, 0] = float(N)
    feats[0, 1] = float(M)
    feats[0, 2] = float(np.log1p(max(0.0, float(getattr(inst, 'deadline', 0.0)))))
    feats[0, 3] = float(edge_den)
    if N > 0 and M > 0:
        ct_all = ct_log.reshape(-1)
        feats[0, 4] = float(ct_all.mean())
        feats[0, 5] = float(ct_all.std(ddof=0))
    else:
        feats[0, 4] = 0.0
        feats[0, 5] = 0.0
    if N > 0:
        cm_all = comm_log.reshape(-1)
        feats[0, 6] = float(cm_all.mean())
        feats[0, 7] = float(cm_all.sum())
    else:
        feats[0, 6] = 0.0
        feats[0, 7] = 0.0
    T = min(N, max_tasks)
    if T > 0:
        ct_row = ct_log[:T, :M] if ct_log.ndim == 2 and M > 0 else np.zeros((T, 0), np.float32)
        if ct_row.shape[1] > 0:
            ct_mean = ct_row.mean(axis=1)
            ct_std = ct_row.std(axis=1, ddof=0)
            ct_min = ct_row.min(axis=1)
            ct_max = ct_row.max(axis=1)
        else:
            ct_mean = np.zeros((T,), np.float32)
            ct_std = np.zeros((T,), np.float32)
            ct_min = np.zeros((T,), np.float32)
            ct_max = np.zeros((T,), np.float32)
        indeg_norm = indeg[:T] / max(1.0, float(N - 1))
        outdeg_norm = outdeg[:T] / max(1.0, float(N - 1))
        level_norm = level[:T] / max(1.0, float(np.max(level)) if level.size else 1.0)
        if N <= 1:
            task_id_norm = np.zeros((T,), np.float32)
        else:
            task_id_norm = np.arange(T, dtype=np.float32) / float(N - 1)
        task_id_fourier = _fourier_id_embed(task_id_norm, n_freq=2)
        base = 1
        feats[base:base + T, 0] = ct_mean
        feats[base:base + T, 1] = ct_std
        feats[base:base + T, 2] = ct_min
        feats[base:base + T, 3] = ct_max
        feats[base:base + T, 4] = indeg_norm
        feats[base:base + T, 5] = outdeg_norm
        feats[base:base + T, 6] = level_norm
        feats[base:base + T, 7] = (indeg[:T] > 0).astype(np.float32)
        feats[base:base + T, 8] = (outdeg[:T] > 0).astype(np.float32)
        feats[base:base + T, 9:13] = task_id_fourier
        feats[base:base + T, 13] = out_sum[:T]
        feats[base:base + T, 14] = in_sum[:T]
        feats[base:base + T, 15] = out_max[:T]
        feats[base:base + T, 16] = in_max[:T]
        feats[base:base + T, 17] = out_deg_norm[:T]
        feats[base:base + T, 18] = in_deg_norm[:T]
        feats[base:base + T, 19] = tlevel_norm[:T]
        feats[base:base + T, 20] = blevel_norm[:T]
        feats[base:base + T, 21] = crit_norm[:T]
        if M > 0 and ct_row.shape[1] > 0 and (topk_ct > 0):
            k = min(topk_ct, M)
            idx = np.argpartition(ct_row, kth=k - 1, axis=1)[:, :k]
            row_vals = np.take_along_axis(ct_row, idx, axis=1)
            ordk = np.argsort(row_vals, axis=1)
            idx = np.take_along_axis(idx, ordk, axis=1)
            row_vals = np.take_along_axis(ct_row, idx, axis=1)
            best = row_vals[:, [0]]
            denom = np.maximum(1e-06, ct_max.reshape(-1, 1))
            ct_norm = row_vals / denom
            delta_norm = (row_vals - best) / denom
            if k <= 1:
                rank_norm = np.zeros((T, k), dtype=np.float32)
            else:
                rank_norm = (np.arange(k, dtype=np.float32)[None, :] / float(k - 1)).repeat(T, axis=0)
            cand_id_norm = idx.astype(np.float32) / float(max(1, M - 1)) if M > 1 else np.zeros_like(idx, dtype=np.float32)
            cand_fourier = _fourier_id_embed(cand_id_norm, n_freq=2)
            off = task_base
            for j in range(k):
                feats[base:base + T, off + j * per_cand + 0] = ct_norm[:, j]
                feats[base:base + T, off + j * per_cand + 1] = delta_norm[:, j]
                feats[base:base + T, off + j * per_cand + 2] = rank_norm[:, j]
                feats[base:base + T, off + j * per_cand + 3:off + j * per_cand + 7] = cand_fourier[:, j, :]
    P = min(M, max_procs)
    if P > 0:
        ct_col = ct_log[:N, :P] if ct_log.ndim == 2 and N > 0 else np.zeros((0, P), np.float32)
        if ct_col.shape[0] > 0:
            col_mean = ct_col.mean(axis=0)
            col_std = ct_col.std(axis=0, ddof=0)
            col_min = ct_col.min(axis=0)
            col_max = ct_col.max(axis=0)
        else:
            col_mean = np.zeros((P,), np.float32)
            col_std = np.zeros((P,), np.float32)
            col_min = np.zeros((P,), np.float32)
            col_max = np.zeros((P,), np.float32)
        base = 1 + max_tasks
        feats[base:base + P, 0] = static_log[:P] if static_log.size else 0.0
        feats[base:base + P, 1] = dyn_log[:P] if dyn_log.size else 0.0
        feats[base:base + P, 2] = fmax_log[:P] if fmax_log.size else 0.0
        feats[base:base + P, 3] = col_mean
        feats[base:base + P, 4] = col_std
        feats[base:base + P, 5] = col_min
        feats[base:base + P, 6] = col_max
        feats[base:base + P, 7:11] = proc_id_fourier[:P, :]
        feats[base:base + P, 11] = proc_id_norm[:P] if M > 1 else 0.0
    return feats
