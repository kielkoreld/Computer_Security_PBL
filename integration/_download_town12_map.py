#!/usr/bin/env python3
"""Download Bench2Drive Town12 HD map from Hugging Face."""
from pathlib import Path

from huggingface_hub import hf_hub_download

out_dir = Path(__file__).resolve().parent / 'maps'
out_dir.mkdir(parents=True, exist_ok=True)
p = hf_hub_download(
    repo_id='rethinklab/Bench2Drive-Map-V0.0.4',
    filename='Town12_HD_map.npz',
    repo_type='dataset',
    local_dir=str(out_dir),
    local_dir_use_symlinks=False,
)
print('downloaded', p, 'size_mb', Path(p).stat().st_size / 1e6)
