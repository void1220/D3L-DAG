"""Legacy HEFT scheduling routines."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os
import time
import pickle
import numpy as np

def _build_succ_from_preds(n: int, preds: Dict[int, List[int]]) -> Dict[int, List[int]]:
    succ = {i: [] for i in range(n)}
    for v, ps in preds.items():
        for u in ps:
            if 0 <= u < n:
                succ[u].append(v)
    return succ

def _compute_upward_ranks(inst) -> np.ndarray:
    N = int(inst.num_tasks)
    M = int(inst.num_procs)
    ct = np.asarray(inst.comp_time, dtype=np.float64)
    comm = np.asarray(inst.comm_time, dtype=np.float64)
    preds: Dict[int, List[int]] = inst.predecessors
    succ: Dict[int, List[int]] = getattr(inst, 'successors', None)
    if succ is None or len(succ) == 0:
        succ = _build_succ_from_preds(N, preds)
    avg_comp = ct.mean(axis=1) if N > 0 and M > 0 else np.zeros((N,), dtype=np.float64)
    indeg = np.zeros(N, dtype=np.int32)
    for v in range(N):
        indeg[v] = len(preds.get(v, []))
    from collections import deque
    q = deque([i for i in range(N) if indeg[i] == 0])
    topo: List[int] = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succ.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(topo) < N:
        topo = list(range(N))
    rank_u = np.zeros(N, dtype=np.float64)
    for i in reversed(topo):
        best = 0.0
        for j in succ.get(i, []):
            val = float(comm[i, j]) + float(rank_u[j])
            if val > best:
                best = val
        rank_u[i] = float(avg_comp[i] + best)
    return rank_u

def heft_solve(inst) -> Tuple[List[int], List[int]]:
    N = int(inst.num_tasks)
    M = int(inst.num_procs)
    ct = np.asarray(inst.comp_time, dtype=np.float64)
    comm = np.asarray(inst.comm_time, dtype=np.float64)
    preds: Dict[int, List[int]] = inst.predecessors
    succ: Dict[int, List[int]] = getattr(inst, 'successors', None)
    if succ is None or len(succ) == 0:
        succ = _build_succ_from_preds(N, preds)
    rank_u = _compute_upward_ranks(inst)
    order = np.lexsort((np.arange(N, dtype=np.int64), -rank_u)).astype(np.int64).tolist()
    indeg = np.zeros(N, dtype=np.int32)
    for v in range(N):
        indeg[v] = len(preds.get(v, []))
    ready = indeg == 0
    scheduled = np.zeros(N, dtype=bool)
    pos = np.empty(N, dtype=np.int32)
    pos[np.array(order, dtype=np.int64)] = np.arange(N, dtype=np.int32)
    proc_ready = np.zeros(M, dtype=np.float64)
    finish_t = np.zeros(N, dtype=np.float64)
    assign = np.full(N, -1, dtype=np.int64)
    done = 0
    while done < N:
        cand = np.where(ready & ~scheduled)[0]
        if cand.size == 0:
            cand = np.where(~scheduled)[0]
        t = int(cand[np.argmin(pos[cand])])
        best_p = 0
        best_eft = float('inf')
        for p in range(M):
            est = float(proc_ready[p])
            for u in preds.get(t, []):
                pu = int(assign[u])
                base = float(finish_t[u])
                if pu != p:
                    d = float(comm[u, t])
                    if d > 0.0:
                        base += d
                if base > est:
                    est = base
            eft = est + float(ct[t, p])
            if eft < best_eft - 1e-12:
                best_eft = eft
                best_p = p
        assign[t] = int(best_p)
        finish_t[t] = float(best_eft)
        proc_ready[best_p] = float(best_eft)
        scheduled[t] = True
        ready[t] = False
        done += 1
        for v in succ.get(t, []):
            indeg[v] -= 1
            if indeg[v] <= 0:
                ready[v] = True
    return (assign.astype(int).tolist(), order)
