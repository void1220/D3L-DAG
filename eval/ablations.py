"""Run controlled D3L ablations."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from core.config import D3LConfig
SUITES = {'noise': {'absorbing': {'kernel': 'absorbing'}, 'uniform': {'kernel': 'uniform'}}, 'steps': {str(t): {'steps': t} for t in (20, 30, 50)}, 'schedule': {s: {'noise_schedule': s} for s in ('linear', 'cosine', 'quadratic', 'sigmoid', 'piecewise')}, 'ema': {'on': {'ema': True}, 'off': {'ema': False}}, 'decoder': {s: {'decoder': s} for s in ('ar', 'continuous', 'd3l')}, 'features': {s: {'feature_mode': s} for s in ('gcn', 'base', 'position', 'full')}}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', choices=[*SUITES, 'all'], required=True)
    p.add_argument('--scale', default='GS03')
    p.add_argument('--data', default='datasets/dts5000')
    p.add_argument('--out', default='runs/ablations')
    p.add_argument('--device', default='cuda')
    p.add_argument('--config')
    p.add_argument('--workers', type=int, default=0)
    a = p.parse_args()
    base = D3LConfig().to_dict()
    if a.config:
        base.update(json.loads(Path(a.config).read_text()))
    for suite in list(SUITES) if a.suite == 'all' else [a.suite]:
        for name, change in SUITES[suite].items():
            out = Path(a.out) / a.scale / suite / name
            out.mkdir(parents=True, exist_ok=True)
            cfg = base | change
            path = out / 'experiment.json'
            path.write_text(json.dumps(cfg, indent=2))
            train = [sys.executable, '-u', '-m', 'core.train', '--scale', a.scale, '--data', a.data, '--out', str(out), '--config', str(path), '--device', a.device, '--workers', str(a.workers)]
            if not (out / 'test/summary.json').exists():
                if (out / 'last.pt').exists():
                    train.append('--resume')
                subprocess.run(train, check=True)
                subprocess.run([sys.executable, '-u', '-m', 'eval.evaluate', '--checkpoint', str(out / 'best.pt'), '--data', a.data, '--out', str(out / 'test'), '--device', a.device], check=True)
if __name__ == '__main__':
    main()
