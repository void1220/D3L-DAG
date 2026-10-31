"""Train the retained AILS sequence denoisers."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from core.train import configure_runtime
from core.decoding import decode_priority_schedule
from data.dataset import SCALES, load, reconstruct
from data.ptr_features import extract_instance_features_simple
from .hcps_AILS_cmax import Seq2SeqLSTMDenoiser, _summarize_inst_features, _add_bos_eos, _build_decoder_input, _build_decoder_target, _corrupt_assignment_numpy, _corrupt_order_numpy

def arrays(raw, scale):
    cond = []
    assign = []
    order = []
    for i in range(len(raw['seeds'])):
        inst = reconstruct(scale, raw, i)
        n, m = (inst.num_tasks, inst.num_procs)
        cond.append(_summarize_inst_features(extract_instance_features_simple(inst, max_tasks=n, max_procs=m, topk_ct=m)))
        a = raw['tokens'][i, :n]
        assign.append(a)
        order.append(decode_priority_schedule(inst, a, raw['tokens'][i, n:]).order)
    return (np.stack(cond), {'assignment': np.stack(assign), 'order': np.stack(order)})

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='datasets/dts5000')
    p.add_argument('--scale', choices=list(SCALES), required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--validation-every', type=int, default=10)
    p.add_argument('--validation-limit', type=int, default=200)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    configure_runtime(a.seed, a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'assignment.pt').exists() or (out / 'order.pt').exists():
        raise FileExistsError('Use a new output directory')
    raw = load(a.data, a.scale, 'train')
    val = load(a.data, a.scale, 'validation', a.validation_limit)
    cond, ys = arrays(raw, a.scale)
    vc, vys = arrays(val, a.scale)
    n, m = raw['comp_time'].shape[1:]
    for kind in ('assignment', 'order'):
        count = m if kind == 'assignment' else n
        bos = count
        eos = count + 1
        best = float('inf')
        model = Seq2SeqLSTMDenoiser(count + 2, 128, 256, 128, cond.shape[-1]).to(a.device)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(1, a.epochs + 1):
            rng = np.random.default_rng(a.seed + epoch)
            model.train()
            values = []
            for ids in np.array_split(rng.permutation(len(cond)), max(1, int(np.ceil(len(cond) / a.batch_size)))):
                clean = ys[kind][ids]
                noisy = np.stack([_corrupt_assignment_numpy(y, count, 0.1, 0.1, rng) if kind == 'assignment' else _corrupt_order_numpy(y, 6, 3, rng) for y in clean])
                src = torch.as_tensor(_add_bos_eos(noisy, bos, eos), device=a.device)
                inp = torch.as_tensor(_build_decoder_input(clean, bos), device=a.device)
                target = torch.as_tensor(_build_decoder_target(clean, eos), device=a.device)
                logits = model(src, inp, torch.as_tensor(cond[ids], device=a.device))
                loss = F.cross_entropy(logits.transpose(1, 2), target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
                values.append(float(loss.detach()))
            print(f'AILS {kind} epoch {epoch}/{a.epochs} loss={np.mean(values):.6f}', flush=True)
            if epoch % a.validation_every == 0 or epoch == a.epochs:
                model.eval()
                scores = []
                with torch.no_grad():
                    for ids in np.array_split(np.arange(len(vc)), max(1, int(np.ceil(len(vc) / a.batch_size)))):
                        clean = vys[kind][ids]
                        noisy = np.stack([_corrupt_assignment_numpy(y, count, 0.1, 0.1, np.random.default_rng(a.seed + int(i))) if kind == 'assignment' else _corrupt_order_numpy(y, 6, 3, np.random.default_rng(a.seed + int(i))) for i, y in zip(ids, clean)])
                        logits = model(torch.as_tensor(_add_bos_eos(noisy, bos, eos), device=a.device), torch.as_tensor(_build_decoder_input(clean, bos), device=a.device), torch.as_tensor(vc[ids], device=a.device))
                        score = F.cross_entropy(logits.transpose(1, 2), torch.as_tensor(_build_decoder_target(clean, eos), device=a.device))
                        scores.append((float(score), len(ids)))
                value = sum((s * k for s, k in scores)) / sum((k for _, k in scores))
                if value < best:
                    best = value
                    torch.save(dict(state_dict=model.state_dict(), kind=kind, seq_len=n, token_count=count, cond_dim=cond.shape[-1], bos_id=bos, eos_id=eos, emb_dim=128, hidden_dim=256, latent_dim=128, epoch=epoch, validation_loss=value), out / (kind + '.pt'))
if __name__ == '__main__':
    main()
