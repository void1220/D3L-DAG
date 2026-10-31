"""Accelerated scheduling environment."""
from dataclasses import dataclass
import numpy as np
import torch
from .hcps_trans_RL_cmax import SchedulingEnv
try:
    from numba import njit
except ImportError:

    def njit(*args, **kwargs):
        return lambda function: function

@dataclass(frozen=True)
class EnvTemplate:
    inst: object
    n: int
    m: int
    scale: float
    reward_scale: float
    rank: np.ndarray
    desc: np.ndarray
    static_features: np.ndarray
    rank_feature: np.ndarray
    desc_feature: np.ndarray
    pred: np.ndarray
    pred_count: np.ndarray
    succ: np.ndarray
    succ_count: np.ndarray

def prepare_template(instance):
    reference = SchedulingEnv(instance)
    inst, n, m = (reference.inst, reference.n, reference.m)
    edge = np.zeros((n, n))
    pred = np.zeros((n, n), dtype=np.int64)
    succ = np.zeros((n, n), dtype=np.int64)
    pred_count = np.zeros(n, dtype=np.int64)
    succ_count = np.zeros(n, dtype=np.int64)
    for t in range(n):
        ps, ss = (inst.predecessors[t], inst.successors[t])
        pred_count[t], succ_count[t] = (len(ps), len(ss))
        pred[t, :len(ps)] = ps
        succ[t, :len(ss)] = ss
        edge[ps, t] = 1
    costs = np.concatenate([np.log1p(inst.comp_time / reference.scale), inst.eligibility_mask, np.log1p(inst.comm_time / reference.scale), np.log1p(inst.comm_time.T / reference.scale), edge, edge.T], axis=1)
    return EnvTemplate(inst, n, m, reference.scale, reference.reward_scale, reference.rank, reference.desc, costs.astype(np.float32), (reference.rank / max(reference.rank.max(), 1)).astype(np.float32), (reference.desc / max(n, 1)).astype(np.float32), pred, pred_count, succ, succ_count)

@njit(cache=True, fastmath=False)
def _step(ct, comm, eligible, pred, pred_count, succ, succ_count, rank, desc, a, start, finish, avail, remaining_pred, route, sequence, reward_scale):
    n, m = ct.shape
    best_task = -1
    best_proc = -1
    best_task_score = 0.0
    for t in range(n):
        if a[t] >= 0 or remaining_pred[t] != 0:
            continue
        selected_proc = -1
        selected_score = 0.0
        selected_end = 0.0
        for p in range(m):
            if not eligible[t, p]:
                continue
            release, incoming = (0.0, 0.0)
            for j in range(pred_count[t]):
                q = pred[t, j]
                cross = a[q] != p
                value = finish[q] + (comm[q, t] if cross else 0.0)
                if value > release:
                    release = value
                if cross:
                    incoming += comm[q, t]
            end = max(avail[p], release) + ct[t, p]
            if route == 0:
                score = end
            elif route == 1:
                score = ct[t, p]
            elif route == 2:
                score = avail[p]
            else:
                score = incoming
            if selected_proc < 0 or score < selected_score or (score == selected_score and (end < selected_end or (end == selected_end and p < selected_proc))):
                selected_proc, selected_score, selected_end = (p, score, end)
        if selected_proc < 0:
            raise ValueError('no eligible processor')
        if sequence == 0:
            task_score = ct[t, selected_proc]
        elif sequence == 1:
            task_score = -ct[t, selected_proc]
        elif sequence == 2:
            task_score = -rank[t]
        elif sequence == 3:
            task_score = -desc[t]
        else:
            task_score = float(t)
        if best_task < 0 or task_score < best_task_score or (task_score == best_task_score and t < best_task):
            best_task, best_proc, best_task_score = (t, selected_proc, task_score)
    if best_task < 0:
        raise ValueError('no schedulable task')
    old = np.max(finish)
    t, p = (best_task, best_proc)
    a[t] = p
    release = 0.0
    for j in range(pred_count[t]):
        q = pred[t, j]
        value = finish[q] + (comm[q, t] if a[q] != p else 0.0)
        if value > release:
            release = value
    start[t] = max(avail[p], release)
    finish[t] = start[t] + ct[t, p]
    avail[p] = finish[t]
    for j in range(succ_count[t]):
        remaining_pred[succ[t, j]] -= 1
    reward = -(np.max(finish) - old) / reward_scale
    return (t, reward)

class FastSchedulingEnv(SchedulingEnv):

    def __init__(self, instance, template=None):
        self._initialize(prepare_template(instance) if template is None else template)

    @classmethod
    def from_template(cls, template):
        obj = cls.__new__(cls)
        obj._initialize(template)
        return obj

    def _initialize(self, template):
        self.template = template
        self.inst = template.inst
        self.n, self.m = (template.n, template.m)
        self.scale, self.reward_scale = (template.scale, template.reward_scale)
        self.rank, self.desc = (template.rank, template.desc)
        self.a = np.full(self.n, -1, dtype=np.int64)
        self.start = np.zeros(self.n)
        self.finish = np.zeros(self.n)
        self.avail = np.zeros(self.m)
        self.order = []
        self.remaining_pred = template.pred_count.copy()

    def ready(self):
        return np.flatnonzero((self.a < 0) & (self.remaining_pred == 0)).tolist()

    def observation_numpy(self):
        t = self.template
        d = t.static_features.shape[1]
        features = np.empty((self.n, d + 7), dtype=np.float32)
        features[:, :d] = t.static_features
        features[:, d] = self.a >= 0
        features[:, d + 1] = (self.a < 0) & (self.remaining_pred == 0)
        features[:, d + 2] = self.start / self.scale
        features[:, d + 3] = self.finish / self.scale
        features[:, d + 4] = (self.a + 1) / self.m
        features[:, d + 5] = t.rank_feature
        features[:, d + 6] = t.desc_feature
        loads = (self.avail / self.scale).astype(np.float32)
        return (features, loads)

    def observation(self, device):
        features, loads = self.observation_numpy()
        return (torch.from_numpy(features).to(device)[None], torch.from_numpy(loads).to(device)[None])

    def step(self, actions):
        route, sequence = map(int, actions)
        if not 0 <= route < 4 or not 0 <= sequence < 5:
            raise ValueError('rule action')
        t = self.template
        task, reward = _step(self.inst.comp_time, self.inst.comm_time, self.inst.eligibility_mask, t.pred, t.pred_count, t.succ, t.succ_count, self.rank, self.desc, self.a, self.start, self.finish, self.avail, self.remaining_pred, route, sequence, self.reward_scale)
        self.order.append(int(task))
        return (float(reward), len(self.order) == self.n)
SchedulingEnvFast = FastSchedulingEnv
