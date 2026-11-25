#!/usr/bin/env bash
# Build mmcv-full (CUDA ops) into HiVT conda only. One script, Linux FS build dir.
# Typical: 15–40 min. Do NOT build under /mnt/c (Windows mount).
set -eo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
fi
conda activate HiVT

echo "python=$(which python) CONDA_PREFIX=$CONDA_PREFIX"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

CC_BIN="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
CXX_BIN="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
if [[ ! -x "$CC_BIN" ]]; then
  echo "Installing gcc/g++ 11 into HiVT..."
  conda install -y -c conda-forge 'gcc_linux-64=11.4.*' 'gxx_linux-64=11.4.*'
fi
if [[ ! -x "$CONDA_PREFIX/bin/nvcc" ]]; then
  echo "Installing cuda-nvcc into HiVT..."
  conda install -y -c nvidia cuda-nvcc=12.1.105 cuda-cudart-dev=12.1.105
fi

SRC="$REPO_ROOT/Sparse4D/third_party/mmcv"
BUILD=$HOME/mmcv_build_hivt
echo "Building on Linux FS: $BUILD"
rm -rf "$BUILD"
mkdir -p "$BUILD"
# Prefer rsync if available (faster); else cp
if command -v rsync >/dev/null; then
  rsync -a --delete --exclude build --exclude dist --exclude '*.egg-info' "$SRC"/ "$BUILD"/
else
  cp -a "$SRC"/. "$BUILD"/
  rm -rf "$BUILD"/build "$BUILD"/dist "$BUILD"/*.egg-info 2>/dev/null || true
fi
cd "$BUILD"

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
# Python.h (conda) needs crypt.h; link system header if missing in env
if [[ ! -f "$CONDA_PREFIX/include/crypt.h" && -f /usr/include/crypt.h ]]; then
  ln -sf /usr/include/crypt.h "$CONDA_PREFIX/include/crypt.h"
fi
SP="$CONDA_PREFIX/lib/python3.8/site-packages"
INCS=("$CONDA_PREFIX/include" /usr/include)
for d in "$SP"/nvidia/*/include; do [[ -d "$d" ]] && INCS+=("$d"); done
LIBS=(); for d in "$SP"/nvidia/*/lib; do [[ -d "$d" ]] && LIBS+=("$d"); done
export CPATH=$(IFS=:; echo "${INCS[*]}"):${CPATH:-}
export CPLUS_INCLUDE_PATH="$CPATH"
export LIBRARY_PATH=$(IFS=:; echo "${LIBS[*]}"):${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$(IFS=:; echo "${LIBS[*]}"):${LD_LIBRARY_PATH:-}
export CUDA_HOME="$CONDA_PREFIX"
# Prefer system ld; conda compiler_compat/ld breaks WSL libm linking
export PATH="$CONDA_PREFIX/bin:/usr/bin:/bin"
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v compiler_compat | paste -sd:)
export PATH
export LD=/usr/bin/ld
CUDART_LIB=$(find "$SP/nvidia" -name 'libcudart.so*' 2>/dev/null | head -1)
CUDART_DIR=$(dirname "${CUDART_LIB:-$SP/nvidia/cuda_runtime/lib}")
LIBS+=("$CUDART_DIR" /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu)
export LIBRARY_PATH=$(IFS=:; echo "${LIBS[*]}"):${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$(IFS=:; echo "${LIBS[*]}"):${LD_LIBRARY_PATH:-}
export LDFLAGS="-L$CUDART_DIR -L/usr/lib/x86_64-linux-gnu -L/lib/x86_64-linux-gnu"
export FORCE_CUDA=1 MMCV_WITH_OPS=1 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=6

echo "CC=$CC"; "$CXX" --version | head -1
nvcc --version | head -4
ls "$SP/nvidia/cusparse/include/cusparse.h"

pip uninstall -y mmcv mmcv-full 2>/dev/null || true
pip install -q ninja
echo "=== compile start $(date -Is) ==="
pip install --no-build-isolation --no-deps . 2>&1 | tee "$HOME/mmcv_build_hivt.log"
echo "=== compile end $(date -Is) ==="
python -c "import mmcv; from mmcv.ops import nms; print('OK', mmcv.__version__, 'ops')"
