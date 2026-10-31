"""Precedence-aware insertion scheduling."""
import heapq
import numpy as np
from numba import njit, prange
from dataclasses import dataclass

def topology(edges):
    deg = edges.sum(axis=0).astype(int)
    ready = list(np.flatnonzero(deg == 0))
    heapq.heapify(ready)
    order = []
    while ready:
        i = heapq.heappop(ready)
        order.append(i)
        for j in np.flatnonzero(edges[i]):
            deg[j] -= 1
            if deg[j] == 0:
                heapq.heappush(ready, int(j))
    if len(order) != len(edges):
        raise ValueError('Cyclic input graph')
    return order

def heft(comp, comm, edges):
    n, m = comp.shape
    order = topology(edges)
    rank = comp.mean(axis=1).copy()
    for i in reversed(order):
        succ = np.flatnonzero(edges[i])
        if len(succ):
            rank[i] += max((float(comm[i, j] + rank[j]) for j in succ))
    a = np.full(n, -1, dtype=np.int64)
    st = np.zeros(n)
    ft = np.zeros(n)
    cal = [[] for _ in range(m)]
    deg = edges.sum(axis=0).astype(int)
    ready = [(-rank[i], i) for i in range(n) if deg[i] == 0]
    heapq.heapify(ready)
    while ready:
        _, i = heapq.heappop(ready)
        pred = np.flatnonzero(edges[:, i])
        choices = []
        for p in range(m):
            s = max((ft[j] + (comm[j, i] if a[j] != p else 0) for j in pred), default=0.0)
            for left, right in cal[p]:
                if s + comp[i, p] <= left:
                    break
                s = max(s, right)
            choices.append((s + comp[i, p], p, s))
        end, p, s = min(choices)
        a[i] = p
        st[i] = s
        ft[i] = end
        cal[p].append((s, end))
        cal[p].sort()
        for j in np.flatnonzero(edges[i]):
            deg[j] -= 1
            if deg[j] == 0:
                heapq.heappush(ready, (-rank[j], int(j)))
    priority = np.empty(n)
    priority[np.lexsort((np.arange(n), st))] = np.arange(n) / n
    return (a, priority, st, ft, float(ft.max()))

@njit(cache=True)
def decode(comp, comm, edges, a, priority):
    n, m = comp.shape
    deg = np.zeros(n, np.int64)
    for i in range(n):
        for j in range(n):
            if edges[i, j]:
                deg[j] += 1
    st = np.zeros(n)
    ft = np.zeros(n)
    done = np.zeros(n, np.bool_)
    cal = np.zeros((m, n), np.int64)
    count = np.zeros(m, np.int64)
    order = np.zeros(n, np.int64)
    for step in range(n):
        i = -1
        key = 1e+300
        for k in range(n):
            if not done[k] and deg[k] == 0 and (priority[k] < key):
                i = k
                key = priority[k]
        if i < 0:
            raise ValueError('Cycle or invalid priority')
        p = a[i]
        s = 0.0
        for j in range(n):
            if edges[j, i]:
                s = max(s, ft[j] + (comm[j, i] if a[j] != p else 0.0))
        dur = comp[i, p]
        pos = count[p]
        for k in range(count[p]):
            j = cal[p, k]
            if s + dur <= st[j]:
                pos = k
                break
            s = max(s, ft[j])
        st[i] = s
        ft[i] = s + dur
        for k in range(count[p], pos, -1):
            cal[p, k] = cal[p, k - 1]
        cal[p, pos] = i
        count[p] += 1
        done[i] = True
        order[step] = i
        for j in range(n):
            if edges[i, j]:
                deg[j] -= 1
    return (float(ft.max()), st, ft, order)

@njit(cache=True, parallel=True)
def evaluate_population(comp, comm, edges, alloc, priorities):
    values = np.empty(len(alloc))
    for k in prange(len(alloc)):
        values[k] = decode(comp, comm, edges, alloc[k], priorities[k])[0]
    return values

@dataclass
class DecodedSchedule:
    order: np.ndarray
    start: np.ndarray
    finish: np.ndarray
    makespan: float

def edge_matrix(inst):
    e = np.zeros((inst.num_tasks, inst.num_tasks), dtype=np.bool_)
    for i, ps in inst.predecessors.items():
        e[ps, i] = True
    return e

def decode_priority_schedule(inst, assignment, priorities):
    a = np.asarray(assignment, dtype=np.int64)
    p = np.asarray(priorities, dtype=np.float64)
    if a.shape != (inst.num_tasks,) or (a < 0).any() or (a >= inst.num_procs).any():
        raise ValueError('Invalid allocation')
    mk, st, ft, order = decode(np.asarray(inst.comp_time, dtype=np.float64), np.asarray(inst.comm_time, dtype=np.float64), edge_matrix(inst), a, p)
    return DecodedSchedule(order, st, ft, mk)
