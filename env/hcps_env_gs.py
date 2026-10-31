"""Gaussian-elimination scheduling environment."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import heapq

@dataclass
class Task:
    id: int
    base_work: float
    energy_coeff: float

@dataclass
class Processor:
    id: int
    f_levels: np.ndarray
    static_power: float
    dyn_coeff: float

@dataclass
class TaskRunInfo:
    task_id: int
    proc_id: int
    freq_level: int
    freq_value: float
    start_time: float
    finish_time: float
    exec_time: float
    energy_dynamic: float

@dataclass
class SimulationResult:
    makespan: float
    energy_static: float
    energy_dynamic: float
    energy_total: float
    feasible: bool
    task_runs: List[TaskRunInfo]

@dataclass
class HCPSInstance:
    num_tasks: int
    num_procs: int
    tasks: List[Task]
    processors: List[Processor]
    predecessors: Dict[int, List[int]]
    successors: Dict[int, List[int]]
    comm_time: np.ndarray
    comp_time: np.ndarray
    deadline: float
    xh: Optional[np.ndarray] = None
    hx: Optional[np.ndarray] = None

def _num_tasks_from_rho(rho: int) -> int:
    level = 2 * rho - 2
    start = 0
    prev_num = 1
    for i in range(2, level + 1):
        if i % 2 == 0:
            num = rho - i // 2
        else:
            num = 1
        start = start + prev_num
        prev_num = num
    return start + 1

def _build_task_level_array(rng: np.random.Generator, rho: int) -> np.ndarray:
    level = 2 * rho - 2
    tla = np.zeros((level, 2), dtype=np.int64)
    tla[0, :] = np.array([0, 1], dtype=np.int64)
    for i in range(2, level + 1):
        prev_start, prev_num = (tla[i - 2, 0], tla[i - 2, 1])
        new_start = prev_start + prev_num
        if i % 2 == 0:
            new_num = rho - i // 2
        else:
            new_num = 1
        tla[i - 1, :] = np.array([new_start, new_num], dtype=np.int64)
    return tla

def _matlab_randi(rng: np.random.Generator, low: int, high: int, size=None) -> np.ndarray:
    return rng.integers(low, high + 1, size=size, endpoint=False)

def _generate_matlab_flag1_gaussian(rng: np.random.Generator, rho: int, num_server: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    task_level = _build_task_level_array(rng, rho)
    level = task_level.shape[0]
    num_task = int(task_level[-1, 0] + 1)
    dag = np.zeros((num_task, num_task), dtype=np.int64) - 1
    for i in range(1, level):
        if i % 2 == 1:
            src = int(task_level[i - 1, 0] + 1)
            next_start = int(task_level[i, 0])
            next_num = int(task_level[i, 1])
            for j in range(1, next_num + 1):
                dst = next_start + j
                dag[src - 1, dst - 1] = 10 + int(_matlab_randi(rng, 1, 90))
        else:
            src = int(task_level[i - 1, 0] + 1)
            dst = int(task_level[i, 0] + 1)
            dag[src - 1, dst - 1] = 10 + int(_matlab_randi(rng, 1, 90))
            cur_num = int(task_level[i - 1, 1])
            for j in range(1, cur_num):
                dag[src - 1, dst - 1] = 10 + int(_matlab_randi(rng, 1, 90))
                if i + 1 < level:
                    src2 = int(task_level[i - 1, 0] + j + 1)
                    dst2 = int(task_level[i + 1, 0] + j)
                    if 1 <= src2 <= num_task and 1 <= dst2 <= num_task:
                        dag[src2 - 1, dst2 - 1] = 0 + int(_matlab_randi(rng, 1, 20))
    aft = np.zeros((num_task, level), dtype=np.int64)
    pre = np.zeros((num_task, level), dtype=np.int64)
    for i in range(num_task):
        cnt = 0
        for j in range(num_task):
            if dag[i, j] > 0:
                aft[i, cnt] = j + 1
                cnt += 1
    for i in range(num_task):
        cnt = 0
        for j in range(num_task):
            if dag[j, i] > 0:
                pre[i, cnt] = j + 1
                cnt += 1
    return (dag, pre, aft, task_level)

class HCPSSimulator:

    def __init__(self, instance: HCPSInstance, comm_jitter_std: float=0.0, exec_jitter_std: float=0.0, rng: Optional[np.random.Generator]=None):
        self.inst = instance
        self.rng = rng

    @staticmethod
    def _validate_assignment_order(n: int, m: int, assignment: List[int], order: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        a = np.asarray(assignment, dtype=np.int64)
        o = np.asarray(order, dtype=np.int64)
        if a.shape != (n,):
            raise ValueError(f'assignment shape must be (N,), got {a.shape}')
        if o.shape != (n,):
            raise ValueError(f'order shape must be (N,), got {o.shape}')
        if (a < 0).any() or (a >= m).any():
            a = np.clip(a, 0, m - 1)
        if o.min(initial=0) < 0 or o.max(initial=-1) >= n:
            need_fix = True
        else:
            cnt = np.bincount(o, minlength=n)
            need_fix = (cnt != 1).any()
        if need_fix:
            seen = np.zeros(n, dtype=np.bool_)
            uniq = np.empty(n, dtype=np.int64)
            k = 0
            for x in o:
                xi = int(x)
                if 0 <= xi < n and (not seen[xi]):
                    uniq[k] = xi
                    seen[xi] = True
                    k += 1
                    if k == n:
                        break
            if k < n:
                missing = np.nonzero(~seen)[0]
                need = n - k
                uniq[k:n] = missing[:need]
            o = uniq
        return (a, o)

    def evaluate_schedule(self, assignment: List[int], order: List[int]):
        inst = self.inst
        n, m = (inst.num_tasks, inst.num_procs)
        a, o = self._validate_assignment_order(n, m, assignment, order)
        preds = inst.predecessors
        succs = inst.successors
        comm = inst.comm_time
        ct = inst.comp_time
        procs = inst.processors
        proc_ready = np.zeros(m, dtype=np.float64)
        start_t = np.zeros(n, dtype=np.float64)
        finish_t = np.zeros(n, dtype=np.float64)
        done = np.zeros(n, dtype=np.bool_)
        pos = np.empty(n, dtype=np.int32)
        for i, tid in enumerate(o):
            pos[int(tid)] = i
        rem = np.zeros(n, dtype=np.int32)
        for t in range(n):
            rem[t] = len(preds.get(t, []))
        heap = []
        for t in range(n):
            if rem[t] == 0:
                heapq.heappush(heap, (int(pos[t]), t))
        task_runs: List[TaskRunInfo] = []
        dyn_energy = np.zeros(n, dtype=np.float64)
        static_p = float(sum((p.static_power for p in procs)))
        p_f_idx = np.array([len(p.f_levels) - 1 for p in procs], dtype=np.int64)
        p_f_val = np.array([float(p.f_levels[-1]) for p in procs], dtype=np.float64)
        p_dyn_c = np.array([float(p.dyn_coeff) for p in procs], dtype=np.float64)
        scheduled_cnt = 0
        while heap:
            _, t_id = heapq.heappop(heap)
            if done[t_id]:
                continue
            p_id = int(a[t_id])
            est = proc_ready[p_id]
            pt = preds.get(t_id, [])
            if pt:
                max_pred = 0.0
                for p in pt:
                    p = int(p)
                    base = float(finish_t[p])
                    if int(a[p]) != p_id:
                        dly = float(comm[p, t_id])
                        if dly > 0.0:
                            base += dly
                    if base > max_pred:
                        max_pred = base
                if max_pred > est:
                    est = max_pred
            exec_time = float(ct[t_id, p_id])
            st = est
            ft = est + exec_time
            start_t[t_id] = st
            finish_t[t_id] = ft
            proc_ready[p_id] = ft
            f_idx = int(p_f_idx[p_id])
            f_val = float(p_f_val[p_id])
            dyn = float(p_dyn_c[p_id] * inst.tasks[t_id].energy_coeff * exec_time * f_val ** 2)
            dyn_energy[t_id] = dyn
            task_runs.append(TaskRunInfo(task_id=int(t_id), proc_id=int(p_id), freq_level=f_idx, freq_value=f_val, start_time=float(st), finish_time=float(ft), exec_time=float(exec_time), energy_dynamic=float(dyn)))
            done[t_id] = True
            scheduled_cnt += 1
            for v in succs.get(t_id, []):
                v = int(v)
                rem[v] -= 1
                if rem[v] == 0:
                    heapq.heappush(heap, (int(pos[v]), v))
        if scheduled_cnt != n:
            raise RuntimeError('No schedulable task found; DAG/order inconsistent.')
        makespan = float(finish_t.max()) if n > 0 else 0.0
        energy_static = static_p * makespan
        energy_dynamic = float(dyn_energy.sum())
        energy_total = energy_static + energy_dynamic
        feasible = makespan <= float(inst.deadline)
        return SimulationResult(makespan=makespan, energy_static=float(energy_static), energy_dynamic=float(energy_dynamic), energy_total=float(energy_total), feasible=bool(feasible), task_runs=task_runs)

def _greedy_lb_makespan(inst: HCPSInstance) -> float:
    n, m = (inst.num_tasks, inst.num_procs)
    preds = inst.predecessors
    comm = inst.comm_time
    ct = inst.comp_time
    proc_ready = np.zeros(m, dtype=np.float64)
    finish_t = np.zeros(n, dtype=np.float64)
    assign = np.full(n, -1, dtype=np.int64)
    topo = list(range(n))
    for t in topo:
        best_finish = float('inf')
        best_p = 0
        for p in range(m):
            est = float(proc_ready[p])
            if preds.get(t):
                pr = []
                for pre in preds[t]:
                    p_pre = int(assign[pre])
                    if p_pre < 0:
                        raise RuntimeError('Invalid topo order in greedy LB.')
                    d = float(comm[pre, t])
                    if p_pre != p and d > 0.0:
                        pr.append(float(finish_t[pre] + d))
                    else:
                        pr.append(float(finish_t[pre]))
                if pr:
                    est = max(est, max(pr))
            ft = est + float(ct[t, p])
            if ft < best_finish:
                best_finish = ft
                best_p = p
        assign[t] = best_p
        finish_t[t] = best_finish
        proc_ready[best_p] = best_finish
    return float(finish_t.max()) if n > 0 else 0.0

def generate_random_hcps_instance(seed: int=0, rho: Optional[float]=None, num_tasks: int=0, num_procs: int=0, edge_prob: float=0.3, deadline_tightness: float=1.1) -> HCPSInstance:
    rng = np.random.default_rng(int(seed))
    if num_procs is None or int(num_procs) <= 0:
        m = int(rng.integers(3, 11))
    else:
        m = int(num_procs)
    if rho is not None:
        rho = int(rho)
        if rho < 2:
            raise ValueError('rho must be >= 2')
    else:
        candidates = [4, 8, 16]
        if num_tasks is None or int(num_tasks) <= 0:
            rho = int(rng.choice([4, 8]))
        else:
            target = int(num_tasks)
            best = None
            for r in candidates:
                n_r = _num_tasks_from_rho(r)
                score = abs(n_r - target)
                if best is None or score < best[0]:
                    best = (score, r, n_r)
            rho = int(best[1])
    dag_mat, pre_arr, aft_arr, task_level = _generate_matlab_flag1_gaussian(rng, rho=rho, num_server=m)
    n = dag_mat.shape[0]
    comm_time = np.where(dag_mat > 0, dag_mat, 0).astype(np.float64)
    predecessors: Dict[int, List[int]] = {i: [] for i in range(n)}
    successors: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if dag_mat[i, j] > 0:
                successors[i].append(j)
                predecessors[j].append(i)
    comp_time = _matlab_randi(rng, 1, 30, size=(n, m)).astype(np.float64)
    pks = 0.01 * np.ones((m,), dtype=np.float64)
    mk = 0.1 * _matlab_randi(rng, 25, 30, size=(m,)).astype(np.float64)
    p = 0.01 * _matlab_randi(rng, 3, 7, size=(m,)).astype(np.float64)
    c = 0.1 * _matlab_randi(rng, 8, 12, size=(m,)).astype(np.float64)
    fmin = _matlab_randi(rng, 1, 49, size=(m,)).astype(np.float64) / 100.0
    processors: List[Processor] = []
    for pid in range(m):
        f_low = float(min(fmin[pid], mk[pid] * 0.9))
        f_high = float(mk[pid])
        f_mid = float(0.5 * (f_low + f_high))
        f_levels = np.array([f_low, f_mid, f_high], dtype=np.float64)
        processors.append(Processor(id=int(pid), f_levels=f_levels, static_power=float(p[pid]), dyn_coeff=float(c[pid])))
    base_work = comp_time.mean(axis=1).astype(np.float64)
    tasks: List[Task] = [Task(id=int(t), base_work=float(base_work[t]), energy_coeff=1.0) for t in range(n)]
    tmp_inst = HCPSInstance(num_tasks=n, num_procs=m, tasks=tasks, processors=processors, predecessors=predecessors, successors=successors, comm_time=comm_time, comp_time=comp_time, deadline=1e+18, xh=pre_arr.astype(np.int64), hx=aft_arr.astype(np.int64))
    lb = _greedy_lb_makespan(tmp_inst)
    deadline = float(lb * float(deadline_tightness))
    return HCPSInstance(num_tasks=n, num_procs=m, tasks=tasks, processors=processors, predecessors=predecessors, successors=successors, comm_time=comm_time, comp_time=comp_time, deadline=deadline, xh=pre_arr.astype(np.int64), hx=aft_arr.astype(np.int64))

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

def extract_instance_features_simple(inst, *, max_tasks: int, max_procs: int, topk_ct: int=11) -> np.ndarray:
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
        out_max = cm.max(axis=1) if N > 0 else np.zeros((N,), np.float32)
        in_max = cm.max(axis=0) if N > 0 else np.zeros((N,), np.float32)
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
    feats[0, 2] = float(np.log1p(max(0.0, float(inst.deadline))))
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
        feats[0, 7] = float(cm_all.std(ddof=0))
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
