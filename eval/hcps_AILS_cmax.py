"""Autoencoder-assisted iterated local search."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import os
import pickle
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from .hcps_heft_cmax import heft_solve

@dataclass
class AILSCfg:
    max_iters: int = 160
    hard_max_iters: int = 100000
    runtime_min_s: float = 0.3
    runtime_max_s: float = 1
    runtime_base_s: float = 0.8
    runtime_task_coeff_s: float = 0.05
    runtime_proc_coeff_s: float = 0.12
    runtime_pair_coeff_s: float = 0.012
    local_assignment_trials: int = 16
    local_priority_trials: int = 16
    assignment_shake_trials: int = 8
    priority_shake_trials: int = 12
    stagnation_limit: int = 12
    init_temp: float = 0.15
    cooling: float = 0.975
    min_temp: float = 0.0001
    use_lstm: bool = True
    lstm_assign_candidates: int = 8
    lstm_order_candidates: int = 8
    lstm_temperature: float = 1
    lstm_topk: int = 4
    lstm_assign_corrupt_prob: float = 0.1
    lstm_order_corrupt_swaps: int = 6
    lstm_order_corrupt_inserts: int = 3
    lstm_assignment_keep_prob: float = 0.1
    seed: int = 12345

@dataclass
class AILSSolution:
    assignment: np.ndarray
    priority: np.ndarray
    fitness: float
    order: List[int]

@dataclass
class AILSRunStats:
    heft_fitness: float
    best_fitness: float
    time_budget_s: float
    elapsed_s: float
    iterations: int
    improved_over_heft: bool

@dataclass
class LSTMModelBundle:
    kind: str
    seq_len: int
    token_count: int
    cond_dim: int
    bos_id: int
    eos_id: int
    device: str
    model: 'Seq2SeqLSTMDenoiser'

def get_device(device: str='auto') -> torch.device:
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)

def _looks_like_heft_supervision(path: str) -> bool:
    return 'heft_dataset' in os.path.basename(str(path)).lower()

def _repair_order(order: Sequence[int], n_tasks: int) -> np.ndarray:
    seen = set()
    fixed: List[int] = []
    for token in order:
        task = int(token)
        if 0 <= task < n_tasks and task not in seen:
            seen.add(task)
            fixed.append(task)
    for task in range(n_tasks):
        if task not in seen:
            fixed.append(task)
    return np.asarray(fixed[:n_tasks], dtype=np.int64)

def _get_inst_features_from_sample(sample: Any) -> np.ndarray:
    feats = np.asarray(getattr(sample, 'inst_features'), dtype=np.float32)
    if feats.ndim != 2:
        raise RuntimeError(f'inst_features should be 2D (L,D), got shape={feats.shape}')
    return feats

def _summarize_inst_features(inst_features: np.ndarray) -> np.ndarray:
    feats = np.asarray(inst_features, dtype=np.float32)
    if feats.ndim != 2:
        raise RuntimeError(f'inst_features should be 2D (L,D), got shape={feats.shape}')
    global_tok = feats[0]
    non_pad = np.abs(feats).sum(axis=1) > 0
    pooled = feats[non_pad].mean(axis=0) if np.any(non_pad) else np.zeros_like(global_tok)
    cond = np.concatenate([global_tok, pooled], axis=0).astype(np.float32)
    return cond

def _expected_cond_dim(assign_bundle: Optional[LSTMModelBundle], order_bundle: Optional[LSTMModelBundle]) -> Optional[int]:
    dims = [int(b.cond_dim) for b in (assign_bundle, order_bundle) if b is not None]
    if not dims:
        return None
    if len(set(dims)) != 1:
        raise ValueError(f'assignment/order checkpoints expect different cond_dim values: {dims}')
    return dims[0]

def _infer_topk_ct_from_cond_dim(cond_dim: int) -> Optional[int]:
    if cond_dim % 2 != 0:
        return None
    feat_dim = cond_dim // 2
    if feat_dim < 22:
        return None
    rem = feat_dim - 22
    if rem % 7 != 0:
        return None
    return rem // 7

def _priority_from_order(order: Sequence[int]) -> np.ndarray:
    return np.asarray([int(x) for x in order], dtype=np.int64)

def _decode_priority_to_feasible_order(priority: Sequence[int], preds: Dict[int, List[int]], n: int) -> List[int]:
    pred_sets = {task: set((int(p) for p in preds.get(task, []))) for task in range(n)}
    done: set[int] = set()
    order: List[int] = []
    while len(order) < n:
        progressed = False
        for token in priority:
            task = int(token)
            if task in done:
                continue
            if pred_sets[task].issubset(done):
                order.append(task)
                done.add(task)
                progressed = True
        if progressed:
            continue
        for task in range(n):
            if task not in done and pred_sets[task].issubset(done):
                order.append(task)
                done.add(task)
                progressed = True
                break
        if not progressed:
            remaining = [task for task in range(n) if task not in done]
            order.extend(remaining)
            break
    return order[:n]

def _evaluate_solution(sim, preds: Dict[int, List[int]], assignment: np.ndarray, priority: np.ndarray) -> Tuple[float, List[int]]:
    order = _decode_priority_to_feasible_order(priority, preds, int(assignment.shape[0]))
    res = sim.evaluate_schedule(assignment.astype(np.int64).tolist(), order)
    return (float(res.makespan), order)

def _make_solution(sim, preds: Dict[int, List[int]], assignment: np.ndarray, priority: np.ndarray) -> AILSSolution:
    fit, order = _evaluate_solution(sim, preds, assignment, priority)
    return AILSSolution(assignment=assignment.astype(np.int64, copy=True), priority=priority.astype(np.int64, copy=True), fitness=float(fit), order=list(order))

def _heft_initial_solution(inst, sim) -> AILSSolution:
    assignment, order = heft_solve(inst)
    return _make_solution(sim, inst.predecessors, np.asarray(assignment, dtype=np.int64), _priority_from_order(order))

def _accept_move(rng: np.random.Generator, old_fit: float, new_fit: float, temp: float) -> bool:
    if new_fit < old_fit:
        return True
    if temp <= 0.0 or old_fit <= 0.0 or (not np.isfinite(old_fit)):
        return False
    delta = (new_fit - old_fit) / max(old_fit, 1e-09)
    return rng.random() < math.exp(-delta / max(temp, 1e-12))

def _adaptive_runtime_budget_s(cfg: AILSCfg, inst) -> float:
    n = int(inst.num_tasks)
    m = int(inst.num_procs)
    raw = float(cfg.runtime_base_s) + float(cfg.runtime_task_coeff_s) * float(n) + float(cfg.runtime_proc_coeff_s) * float(m) + float(cfg.runtime_pair_coeff_s) * float(n * m)
    return float(np.clip(raw, float(cfg.runtime_min_s), float(cfg.runtime_max_s)))

def _priority_swap(priority: np.ndarray, i: int, j: int) -> np.ndarray:
    out = priority.copy()
    out[i], out[j] = (out[j], out[i])
    return out

def _priority_insert(priority: np.ndarray, i: int, j: int) -> np.ndarray:
    if i == j:
        return priority.copy()
    value = int(priority[i])
    rest = np.delete(priority, i)
    return np.insert(rest, j, value).astype(np.int64, copy=False)

def _assignment_local_search(sol: AILSSolution, inst, sim, rng: np.random.Generator, trials: int, temp: float) -> AILSSolution:
    n, m = (int(inst.num_tasks), int(inst.num_procs))
    current = sol
    task_order = rng.permutation(n)
    for task_id in task_order[:min(trials, n)]:
        task = int(task_id)
        best_candidate = current
        base_proc = int(current.assignment[task])
        proc_candidates = list(range(m))
        rng.shuffle(proc_candidates)
        for proc in proc_candidates:
            if proc == base_proc:
                continue
            cand_assign = current.assignment.copy()
            cand_assign[task] = int(proc)
            cand_fit, cand_order = _evaluate_solution(sim, inst.predecessors, cand_assign, current.priority)
            if cand_fit < best_candidate.fitness:
                best_candidate = AILSSolution(cand_assign, current.priority.copy(), cand_fit, cand_order)
        if best_candidate is not current:
            current = best_candidate
            continue
        proc = int(rng.integers(0, m))
        if proc != base_proc:
            cand_assign = current.assignment.copy()
            cand_assign[task] = proc
            cand_fit, cand_order = _evaluate_solution(sim, inst.predecessors, cand_assign, current.priority)
            if _accept_move(rng, current.fitness, cand_fit, temp):
                current = AILSSolution(cand_assign, current.priority.copy(), cand_fit, cand_order)
    return current

def _priority_local_search(sol: AILSSolution, inst, sim, rng: np.random.Generator, trials: int, temp: float) -> AILSSolution:
    n = int(inst.num_tasks)
    current = sol
    for _ in range(trials):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        if rng.random() < 0.5:
            cand_priority = _priority_swap(current.priority, i, j)
        else:
            cand_priority = _priority_insert(current.priority, i, j)
        cand_fit, cand_order = _evaluate_solution(sim, inst.predecessors, current.assignment, cand_priority)
        if _accept_move(rng, current.fitness, cand_fit, temp):
            current = AILSSolution(current.assignment.copy(), cand_priority, cand_fit, cand_order)
    return current

def _shake_solution(sol: AILSSolution, inst, sim, rng: np.random.Generator, assign_trials: int, priority_trials: int, temp: float) -> AILSSolution:
    n, m = (int(inst.num_tasks), int(inst.num_procs))
    assignment = sol.assignment.copy()
    priority = sol.priority.copy()
    for _ in range(assign_trials):
        task = int(rng.integers(0, n))
        assignment[task] = int(rng.integers(0, m))
    for _ in range(priority_trials):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        if rng.random() < 0.5:
            priority = _priority_swap(priority, i, j)
        else:
            priority = _priority_insert(priority, i, j)
    fit, order = _evaluate_solution(sim, inst.predecessors, assignment, priority)
    candidate = AILSSolution(assignment, priority, fit, order)
    if _accept_move(rng, sol.fitness, candidate.fitness, temp):
        return candidate
    return sol

class Seq2SeqLSTMDenoiser(nn.Module):

    def __init__(self, vocab_size: int, emb_dim: int, enc_hidden_dim: int, latent_dim: int, cond_dim: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.encoder = nn.LSTM(emb_dim, enc_hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(enc_hidden_dim, latent_dim)
        self.cond_proj = nn.Sequential(nn.Linear(int(cond_dim), latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim))
        self.fuse = nn.Linear(latent_dim * 2, latent_dim)
        self.decoder = nn.LSTM(emb_dim, latent_dim, batch_first=True)
        self.output = nn.Linear(latent_dim, vocab_size)

    def encode(self, src_tokens: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        src_emb = self.emb(src_tokens)
        _, (h_n, _) = self.encoder(src_emb)
        enc_last = h_n[-1]
        seq_latent = torch.tanh(self.to_latent(enc_last))
        cond_latent = torch.tanh(self.cond_proj(cond_vec))
        latent = torch.tanh(self.fuse(torch.cat([seq_latent, cond_latent], dim=-1)))
        return latent

    def _init_decoder_state(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = latent.unsqueeze(0)
        c0 = torch.zeros_like(h0)
        return (h0, c0)

    def forward(self, src_tokens: torch.Tensor, dec_in_tokens: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        latent = self.encode(src_tokens, cond_vec)
        h0, c0 = self._init_decoder_state(latent)
        dec_emb = self.emb(dec_in_tokens)
        dec_out, _ = self.decoder(dec_emb, (h0, c0))
        return self.output(dec_out)

    @torch.no_grad()
    def generate(self, src_tokens: torch.Tensor, cond_vec: torch.Tensor, bos_id: int, eos_id: int, max_len: int, temperature: float=1.0, topk: int=0, sample: bool=True) -> torch.Tensor:
        latent = self.encode(src_tokens, cond_vec)
        h, c = self._init_decoder_state(latent)
        batch = int(src_tokens.size(0))
        prev = torch.full((batch, 1), int(bos_id), dtype=torch.long, device=src_tokens.device)
        outputs: List[torch.Tensor] = []
        for _ in range(max_len + 1):
            emb = self.emb(prev[:, -1:])
            dec_out, (h, c) = self.decoder(emb, (h, c))
            logits = self.output(dec_out[:, 0, :])
            logits[:, int(bos_id)] = -1000000000.0
            if temperature != 1.0:
                logits = logits / max(float(temperature), 1e-06)
            if topk and 0 < topk < logits.size(-1):
                top_vals, top_idx = torch.topk(logits, k=int(topk), dim=-1)
                if sample:
                    probs = torch.softmax(top_vals, dim=-1)
                    choose = torch.multinomial(probs, num_samples=1)
                    nxt = top_idx.gather(1, choose).squeeze(1)
                else:
                    nxt = top_idx[:, 0]
            elif sample:
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                nxt = torch.argmax(logits, dim=-1)
            outputs.append(nxt)
            prev = torch.cat([prev, nxt.unsqueeze(1)], dim=1)
        out = torch.stack(outputs, dim=1)
        cleaned = torch.full((batch, max_len), int(eos_id), dtype=torch.long, device=src_tokens.device)
        for b in range(batch):
            buf: List[int] = []
            for tok in out[b].tolist():
                if int(tok) == int(eos_id):
                    break
                buf.append(int(tok))
                if len(buf) >= max_len:
                    break
            if buf:
                cleaned[b, :len(buf)] = torch.tensor(buf, dtype=torch.long, device=src_tokens.device)
        return cleaned

def _corrupt_assignment_numpy(clean: np.ndarray, token_count: int, replace_prob: float, swap_prob: float, rng: np.random.Generator) -> np.ndarray:
    noisy = clean.copy()
    mask = rng.random(size=noisy.shape[0]) < float(replace_prob)
    if mask.any():
        noisy[mask] = rng.integers(0, int(token_count), size=int(mask.sum()))
    if rng.random() < float(swap_prob) and noisy.shape[0] >= 2:
        i, j = rng.choice(noisy.shape[0], size=2, replace=False)
        noisy[i], noisy[j] = (noisy[j], noisy[i])
    return noisy

def _corrupt_order_numpy(clean: np.ndarray, num_swaps: int, num_inserts: int, rng: np.random.Generator) -> np.ndarray:
    noisy = clean.copy()
    n = noisy.shape[0]
    if n <= 1:
        return noisy
    for _ in range(int(num_swaps)):
        i, j = rng.choice(n, size=2, replace=False)
        noisy[i], noisy[j] = (noisy[j], noisy[i])
    for _ in range(int(num_inserts)):
        i, j = rng.choice(n, size=2, replace=False)
        value = int(noisy[i])
        rest = np.delete(noisy, i)
        noisy = np.insert(rest, j, value)
    return noisy.astype(np.int64, copy=False)

def _add_bos_eos(batch: np.ndarray, bos_id: int, eos_id: int) -> np.ndarray:
    bsz, seq_len = batch.shape
    out = np.empty((bsz, seq_len + 2), dtype=np.int64)
    out[:, 0] = int(bos_id)
    out[:, 1:-1] = batch
    out[:, -1] = int(eos_id)
    return out

def _build_decoder_input(batch: np.ndarray, bos_id: int) -> np.ndarray:
    bsz, seq_len = batch.shape
    out = np.empty((bsz, seq_len + 1), dtype=np.int64)
    out[:, 0] = int(bos_id)
    out[:, 1:] = batch
    return out

def _build_decoder_target(batch: np.ndarray, eos_id: int) -> np.ndarray:
    bsz, seq_len = batch.shape
    out = np.empty((bsz, seq_len + 1), dtype=np.int64)
    out[:, :-1] = batch
    out[:, -1] = int(eos_id)
    return out

def load_lstm_bundle(ckpt_path: str, device: str='auto') -> LSTMModelBundle:
    torch_device = get_device(device)
    ckpt = torch.load(ckpt_path, map_location=torch_device)
    train_cfg = ckpt.get('train_cfg', {})
    if isinstance(train_cfg, dict) and _looks_like_heft_supervision(str(train_cfg.get('dataset_path', ''))):
        print(f'[AILS-LSTM] warning: checkpoint {os.path.basename(ckpt_path)} was trained on a HEFT dataset and will mostly imitate HEFT-style schedules.')
    model = Seq2SeqLSTMDenoiser(vocab_size=int(ckpt['token_count']) + 2, emb_dim=int(ckpt['emb_dim']), enc_hidden_dim=int(ckpt['hidden_dim']), latent_dim=int(ckpt['latent_dim']), cond_dim=int(ckpt['cond_dim'])).to(torch_device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return LSTMModelBundle(kind=str(ckpt['kind']), seq_len=int(ckpt['seq_len']), token_count=int(ckpt['token_count']), cond_dim=int(ckpt['cond_dim']), bos_id=int(ckpt['bos_id']), eos_id=int(ckpt['eos_id']), device=str(torch_device), model=model)

def _bundle_generate(bundle: LSTMModelBundle, sequence: np.ndarray, cond_vec: np.ndarray, temperature: float, topk: int, sample: bool) -> np.ndarray:
    cond_arr = np.asarray(cond_vec, dtype=np.float32)
    if cond_arr.ndim != 1:
        raise ValueError(f'cond_vec must be 1D, got shape={cond_arr.shape}')
    if cond_arr.shape[0] != int(bundle.cond_dim):
        raise ValueError(f'Condition feature dimension mismatch: got cond_vec dim={cond_arr.shape[0]}, but checkpoint expects cond_dim={bundle.cond_dim}. This usually means you are using a checkpoint trained on a different env/dataset (for example GS vs FFT) or with a different instance feature extractor version.')
    device = torch.device(bundle.device)
    src = torch.from_numpy(_add_bos_eos(sequence[None, :], bundle.bos_id, bundle.eos_id)).to(device)
    cond = torch.from_numpy(cond_arr[None, :]).to(device)
    out = bundle.model.generate(src_tokens=src, cond_vec=cond, bos_id=bundle.bos_id, eos_id=bundle.eos_id, max_len=bundle.seq_len, temperature=float(temperature), topk=int(topk), sample=bool(sample))
    seq = out[0].detach().cpu().numpy().astype(np.int64)
    return seq

def _build_cond_vec_for_instance(inst, *, sample: Optional[Any], assign_bundle: Optional[LSTMModelBundle], order_bundle: Optional[LSTMModelBundle], extract_instance_features_simple) -> np.ndarray:
    expected_dim = _expected_cond_dim(assign_bundle, order_bundle)
    if sample is not None and hasattr(sample, 'inst_features'):
        sample_cond = _summarize_inst_features(_get_inst_features_from_sample(sample))
        if expected_dim is None or sample_cond.shape[0] == expected_dim:
            return sample_cond
    topk_ct = int(inst.num_procs)
    if expected_dim is not None:
        inferred_topk = _infer_topk_ct_from_cond_dim(int(expected_dim))
        if inferred_topk is not None:
            topk_ct = int(inferred_topk)
    feats = extract_instance_features_simple(inst, max_tasks=inst.num_tasks, max_procs=inst.num_procs, topk_ct=topk_ct)
    cond_vec = _summarize_inst_features(feats)
    if expected_dim is not None and cond_vec.shape[0] != expected_dim:
        raise ValueError(f'Condition feature dimension mismatch after runtime extraction: got cond_vec dim={cond_vec.shape[0]}, expected cond_dim={expected_dim}. Resolved topk_ct={topk_ct}. Your checkpoint and runtime feature extractor are still inconsistent.')
    return cond_vec

def _lstm_assignment_operator(sol: AILSSolution, inst, sim, rng: np.random.Generator, temp: float, bundle: Optional[LSTMModelBundle], cfg: AILSCfg, cond_vec: np.ndarray) -> AILSSolution:
    if bundle is None:
        return sol
    current = sol
    total_candidates = int(cfg.lstm_assign_candidates)
    for candidate_idx in range(total_candidates):
        noisy = _corrupt_assignment_numpy(current.assignment, token_count=int(inst.num_procs), replace_prob=float(cfg.lstm_assign_corrupt_prob), swap_prob=0.15, rng=rng)
        decoded = _bundle_generate(bundle, noisy, cond_vec, temperature=float(cfg.lstm_temperature), topk=int(cfg.lstm_topk), sample=bool(candidate_idx > 0))
        invalid = (decoded < 0) | (decoded >= int(inst.num_procs))
        candidate_assign = decoded.copy()
        if invalid.any():
            candidate_assign[invalid] = current.assignment[invalid]
        mix_mask = rng.random(size=decoded.shape[0]) < float(cfg.lstm_assignment_keep_prob)
        candidate_assign[mix_mask] = current.assignment[mix_mask]
        cand_fit, cand_order = _evaluate_solution(sim, inst.predecessors, candidate_assign, current.priority)
        if _accept_move(rng, current.fitness, cand_fit, temp):
            current = AILSSolution(candidate_assign, current.priority.copy(), cand_fit, cand_order)
    return current

def _lstm_priority_operator(sol: AILSSolution, inst, sim, rng: np.random.Generator, temp: float, bundle: Optional[LSTMModelBundle], cfg: AILSCfg, cond_vec: np.ndarray) -> AILSSolution:
    if bundle is None:
        return sol
    current = sol
    total_candidates = int(cfg.lstm_order_candidates)
    for candidate_idx in range(total_candidates):
        noisy = _corrupt_order_numpy(current.priority, num_swaps=int(cfg.lstm_order_corrupt_swaps), num_inserts=int(cfg.lstm_order_corrupt_inserts), rng=rng)
        decoded = _bundle_generate(bundle, noisy, cond_vec, temperature=float(cfg.lstm_temperature), topk=int(cfg.lstm_topk), sample=bool(candidate_idx > 0))
        candidate_priority = _repair_order(decoded, int(inst.num_tasks))
        cand_fit, cand_order = _evaluate_solution(sim, inst.predecessors, current.assignment, candidate_priority)
        if _accept_move(rng, current.fitness, cand_fit, temp):
            current = AILSSolution(current.assignment.copy(), candidate_priority, cand_fit, cand_order)
    return current

def ails_solve_with_sim(inst, sim, cfg: AILSCfg, *, assign_bundle: Optional[LSTMModelBundle]=None, order_bundle: Optional[LSTMModelBundle]=None, cond_vec: Optional[np.ndarray]=None) -> Tuple[List[int], List[int], float, AILSRunStats]:
    rng = np.random.default_rng(int(cfg.seed))
    current = _heft_initial_solution(inst, sim)
    best = AILSSolution(current.assignment.copy(), current.priority.copy(), current.fitness, list(current.order))
    heft_fitness = float(best.fitness)
    temp = float(cfg.init_temp)
    stagnation = 0
    start = time.perf_counter()
    time_budget_s = _adaptive_runtime_budget_s(cfg, inst)
    iterations = 0
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= float(time_budget_s):
            break
        if iterations >= int(cfg.hard_max_iters):
            break
        if best.fitness < heft_fitness and iterations >= int(cfg.max_iters):
            break
        iterations += 1
        if bool(cfg.use_lstm):
            if cond_vec is None:
                raise ValueError('cond_vec is required when cfg.use_lstm=True')
            current = _lstm_assignment_operator(current, inst, sim, rng, temp, assign_bundle, cfg, cond_vec)
            current = _lstm_priority_operator(current, inst, sim, rng, temp, order_bundle, cfg, cond_vec)
        current = _assignment_local_search(current, inst, sim, rng, trials=int(cfg.local_assignment_trials), temp=temp)
        current = _priority_local_search(current, inst, sim, rng, trials=int(cfg.local_priority_trials), temp=temp)
        if current.fitness < best.fitness:
            best = AILSSolution(current.assignment.copy(), current.priority.copy(), current.fitness, list(current.order))
            stagnation = 0
        else:
            stagnation += 1
        if stagnation >= int(cfg.stagnation_limit):
            current = _shake_solution(current, inst, sim, rng, assign_trials=int(cfg.assignment_shake_trials), priority_trials=int(cfg.priority_shake_trials), temp=temp)
            stagnation = 0
            if current.fitness < best.fitness:
                best = AILSSolution(current.assignment.copy(), current.priority.copy(), current.fitness, list(current.order))
        temp = max(float(cfg.min_temp), temp * float(cfg.cooling))
    elapsed_s = float(time.perf_counter() - start)
    stats = AILSRunStats(heft_fitness=float(heft_fitness), best_fitness=float(best.fitness), time_budget_s=float(time_budget_s), elapsed_s=elapsed_s, iterations=int(iterations), improved_over_heft=bool(best.fitness < heft_fitness))
    return (best.assignment.astype(int).tolist(), list(best.order), float(best.fitness), stats)
