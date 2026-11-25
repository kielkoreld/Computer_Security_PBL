#!/usr/bin/env python3
"""Summarize HiVT AV2 training log / checkpoints."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / 'logs' / 'train_av2_rtx4080.log'
CKPT_ROOT = ROOT / 'lightning_logs'


def main() -> None:
    text = LOG.read_text(errors='ignore') if LOG.exists() else ''
    print('log_exists', LOG.exists(), 'size_mb', round(LOG.stat().st_size / 1e6, 2) if LOG.exists() else None)
    print('has_traceback', 'Traceback' in text)
    print('trainer_finished', any(s in text for s in ('`Trainer.fit` stopped', 'Training complete', 'MAX EPOCHS')))

    # Epoch completion markers with val metrics often appear near validating end.
    # Prefer explicit metric dumps from progress / epoch end lines.
    epoch_vals = {}
    # Patterns like: Epoch 12: ... val_minADE=1.23, val_minFDE=2.34, val_minMR=0.12
    for m in re.finditer(
            r'Epoch\s+(\d+):.*?val_minADE=([-\d.]+).*?val_minFDE=([-\d.]+).*?val_minMR=([-\d.]+)',
            text):
        ep = int(m.group(1))
        # Lightning shows previous epoch's val during next train; store last seen per ep label.
        epoch_vals[ep] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))

    # Also catch standalone Validating completion lines if present.
    max_epoch_seen = -1
    for m in re.finditer(r'Epoch\s+(\d+):', text):
        max_epoch_seen = max(max_epoch_seen, int(m.group(1)))
    print('max_epoch_seen_in_progress', max_epoch_seen)
    print('epochs_with_val_in_progress_bar', sorted(epoch_vals))

    if epoch_vals:
        # Map: during Epoch N progress, displayed val is usually from end of Epoch N-1
        # except after validation of epoch N completes Lightning may update.
        print('--- last val metric snapshot per progress-bar epoch label ---')
        for ep in sorted(epoch_vals):
            ade, fde, mr = epoch_vals[ep]
            print(f'  progress_epoch={ep}: val_minADE={ade:.3f} val_minFDE={fde:.3f} val_minMR={mr:.3f}')
        best_ep = min(epoch_vals, key=lambda e: epoch_vals[e][1])
        ade, fde, mr = epoch_vals[best_ep]
        print(f'best_by_val_minFDE_in_log: progress_epoch={best_ep} FDE={fde:.3f} ADE={ade:.3f} MR={mr:.3f}')

    # Checkpoints
    ckpts = sorted(CKPT_ROOT.glob('**/checkpoints/*.ckpt'))
    print('--- checkpoints ---')
    print('num_ckpts', len(ckpts))
    for p in ckpts:
        print(f'  {p.relative_to(ROOT)}  {p.stat().st_size/1e6:.2f}MB')

    # Parse epoch/score from filenames if present
    scored = []
    for p in ckpts:
        m = re.search(r'epoch[=_-]?(\d+).*?(?:val_minFDE|minFDE)[=_-]?([0-9.]+)', p.name, re.I)
        if m:
            scored.append((int(m.group(1)), float(m.group(2)), p))
        else:
            m2 = re.search(r'epoch[=_-]?(\d+)', p.name, re.I)
            if m2:
                scored.append((int(m2.group(1)), None, p))
    if scored:
        scored_with = [x for x in scored if x[1] is not None]
        if scored_with:
            best = min(scored_with, key=lambda x: x[1])
            print(f'best_ckpt_by_filename_FDE: epoch={best[0]} FDE={best[1]} path={best[2].relative_to(ROOT)}')
        print('ckpt_epochs', sorted({e for e, _, _ in scored}))

    # Tail breadcrumbs without huge tqdm spam
    parts = re.split(r'[\r\n]+', text)
    interesting = [p for p in parts if p.strip() and (
        'Traceback' in p or 'Error' in p or 'val_minFDE' in p or 'Saving' in p or
        'Trainer.fit' in p or 'Epoch 63' in p or 'Epoch 64' in p)]
    print('--- last interesting lines ---')
    for p in interesting[-15:]:
        print(p[-350:])


if __name__ == '__main__':
    main()
