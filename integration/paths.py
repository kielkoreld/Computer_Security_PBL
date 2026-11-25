"""Repo layout helpers after Sparse4D / HiVT / integration split."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HIVT_ROOT = REPO_ROOT / 'HiVT'
SPARSE4D_ROOT = REPO_ROOT / 'Sparse4D'
WEIGHTS_NOTES = SPARSE4D_ROOT / 'weights_notes'
INTEGRATION_ROOT = REPO_ROOT / 'integration'
OUTPUTS_ROOT = REPO_ROOT / 'outputs'
LIGHTNING_LOGS = REPO_ROOT / 'lightning_logs'
DEFAULT_HIVT_CKPT = (
    LIGHTNING_LOGS / 'version_1' / 'checkpoints' / 'epoch=55-step=349887.ckpt'
)
DEFAULT_SPARSE4D_CKPT = (
    SPARSE4D_ROOT / 'checkpoints' / 'sparse4dv3_r50_b2d_10hz_epoch14.pth'
)
SPARSE4D_HANDOFF_CORE = SPARSE4D_ROOT / 'handoff' / 'core'


def ensure_import_paths() -> None:
    """Put HiVT and repo root on sys.path for models/datasets + integration."""
    # av2 is usually installed in the HiVT conda env (site-packages).
    for p in (str(HIVT_ROOT), str(REPO_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)


def ensure_sparse4d_paths() -> None:
    ensure_import_paths()
    for p in (str(SPARSE4D_ROOT), str(SPARSE4D_HANDOFF_CORE)):
        if p not in sys.path:
            sys.path.insert(0, p)
