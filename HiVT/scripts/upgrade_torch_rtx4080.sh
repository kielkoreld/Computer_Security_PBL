#!/usr/bin/env bash
set -eo pipefail

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
# Avoid nounset failures from conda activate scripts.
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64}"
conda activate HiVT

echo "=== before ==="
python - <<'PY'
import sys
print(sys.version)
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda)
except Exception as e:
    print("torch import failed", e)
PY
pip list | grep -iE 'torch|lightning|geometric|scatter|sparse|cluster|spline|pyg' || true

echo "=== uninstall old torch stack ==="
pip uninstall -y torch torchvision torchaudio \
  torch-geometric torch_geometric \
  torch-scatter torch_scatter \
  torch-sparse torch_sparse \
  torch-cluster torch_cluster \
  torch-spline-conv torch_spline_conv \
  pyg-lib pyg_lib 2>/dev/null || true

echo "=== install torch 2.4.1 + cu121 ==="
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121

echo "=== install torch-geometric + extensions ==="
pip install torch-geometric==2.5.3
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# Keep Lightning 1.5.x API (from_argparse_args). Reinstall pin if pulled away.
pip install "pytorch-lightning==1.5.2" --no-deps 2>/dev/null || pip install "pytorch-lightning==1.5.2"

echo "=== after ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
    x = torch.randn(2, 3, device="cuda")
    y = torch.softmax(x, dim=-1)
    print("cuda softmax ok", y.shape)
import torch_geometric
print("torch_geometric", torch_geometric.__version__)
import pytorch_lightning as pl
print("lightning", pl.__version__)
import av2
print("av2 ok", getattr(av2, "__version__", "?"))
PY

echo "DONE"
