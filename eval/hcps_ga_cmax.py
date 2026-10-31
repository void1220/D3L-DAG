"""Genetic algorithm for DAG scheduling."""
import time
import numpy as np
from core.decoding import decode, evaluate_population

def genetic(comp, comm, edges, seed, seconds=120.0, population=96, report=lambda **kw: None, max_generations=4000, patience=500, min_generations=500):
    begin = time.perf_counter()
    rng = np.random.default_rng(seed)
    n, m = comp.shape
    alloc = rng.integers(m, size=(population, n), dtype=np.int64)
    keys = rng.random((population, n))
    score = evaluate_population(comp, comm, edges, alloc, keys)
    evaluations = population
    gen = 0
    last = time.perf_counter()
    best = int(np.argmin(score))
    best_a = alloc[best].copy()
    best_k = keys[best].copy()
    best_value = float(score[best])
    best_at = time.perf_counter() - begin
    elite = max(2, population // 20)
    last_improved = 0
    stop_reason = 'generation_limit'
    while gen < max_generations:
        if time.perf_counter() - begin >= seconds:
            stop_reason = 'time_limit'
            break
        if patience > 0 and gen >= min_generations and (gen - last_improved >= patience):
            stop_reason = 'stagnation'
            break
        keep = np.argsort(score, kind='stable')[:elite]
        count = population - elite
        tournament = rng.integers(population, size=(2, count, 3))
        parents = np.take_along_axis(tournament, np.argmin(score[tournament], axis=2)[..., None], axis=2)[..., 0]
        p1, p2 = parents
        cross = rng.random((count, 1)) < 0.9
        child_a = np.where((rng.random((count, n)) < 0.5) & cross, alloc[p2], alloc[p1])
        child_k = np.where((rng.random((count, n)) < 0.5) & cross, keys[p2], keys[p1])
        mutate = rng.random((count, n)) < 1 / n
        child_a[mutate] = rng.integers(m, size=int(mutate.sum()))
        mutate = rng.random((count, n)) < 2 / n
        child_k[mutate] = np.clip(child_k[mutate] + rng.normal(0, 0.2, int(mutate.sum())), 0, 1)
        alloc = np.vstack((alloc[keep], child_a))
        keys = np.vstack((keys[keep], child_k))
        new = evaluate_population(comp, comm, edges, alloc[elite:], keys[elite:])
        score = np.concatenate((score[keep], new))
        evaluations += count
        gen += 1
        j = int(np.argmin(score))
        if score[j] < best_value:
            best_value = float(score[j])
            best_a = alloc[j].copy()
            best_k = keys[j].copy()
            best_at = time.perf_counter() - begin
            last_improved = gen
        if time.perf_counter() - last >= 10:
            report(phase='GA', generations=gen, max_generations=max_generations, stagnant_generations=gen - last_improved, evaluations=evaluations, best=best_value, elapsed=time.perf_counter() - begin)
            last = time.perf_counter()
    mk, st, ft, _ = decode(comp, comm, edges, best_a, best_k)
    return dict(status='completed', makespan=mk, assignment=best_a.tolist(), start=st.tolist(), finish=ft.tolist(), wall_seconds=time.perf_counter() - begin, best_found_seconds=best_at, generations=gen, evaluations=evaluations, warm_start='none', initialization='uniform_random_population', seed=int(seed), stop_reason=stop_reason, max_generations=max_generations, patience=patience, min_generations=min_generations, last_improved_generation=last_improved, requested_seconds=seconds)
