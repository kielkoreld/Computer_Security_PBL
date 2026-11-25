#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${CUDA_HOME:?CUDA 12.8 toolkit 경로를 CUDA_HOME으로 지정하세요}"
: "${TORCH_CUDA_ARCH_LIST:=12.0}"
export CUDA_HOME TORCH_CUDA_ARCH_LIST
export MAX_JOBS=2 MMCV_WITH_OPS=1 FORCE_CUDA=1 CFLAGS=-g0 CXXFLAGS=-g0
python3.12 -m venv .venv-models
.venv-models/bin/python -m pip install setuptools==75.8.0 wheel ninja
.venv-models/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.venv-models/bin/python -m pip install numpy==1.26.4 scipy==1.14.1 Pillow==11.1.0 addict yapf==0.40.1 packaging opencv-python==4.11.0.86 mmdet==2.28.2 mmsegmentation==0.30.0 pyquaternion==0.9.9 nuscenes-devkit==1.2.0 numba==0.61.2 plyfile==1.1.2 einops==0.8.1 terminaltables scikit-image==0.25.2 lyft-dataset-sdk==0.0.8 trimesh==4.6.4 pandas==2.2.3 PyYAML==6.0.2 ipython==8.32.0
export PATH="$PWD/.venv-models/bin:$PATH"
.venv-models/bin/python -m pip install --no-build-isolation --no-deps ./third_party/mmcv
package_python="$PWD/.venv-models/bin/python"
cd projects/mmdet3d_plugin/ops
"$package_python" setup.py build_ext --inplace
"$package_python" -m pip check
