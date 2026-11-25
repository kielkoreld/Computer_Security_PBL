# Sparse4D × HiVT Workspace

Bench2Drive 기반 **3D 탐지(Sparse4D)** → **다중 객체 궤적 예측(HiVT)** → **시간 동기·위험 패킷(integration)** 워크스페이스입니다.

## 폴더 구조

| 폴더 | 역할 |
|------|------|
| [`Sparse4D/`](Sparse4D/) | Sparse4Dv3 10Hz 탐지 모델·가중치·인계 문서 |
| [`HiVT/`](HiVT/) | AV2 HiVT 궤적 예측 (actor_type 포함) |
| [`integration/`](integration/) | 연결 브리지, `T_fuse` 시간 동기화, 핸드오프 export |

상세: [`STRUCTURE.md`](STRUCTURE.md) · 시간 동기: [`integration/docs/TIME_SYNC.md`](integration/docs/TIME_SYNC.md)

```text
repo/
  Sparse4D/          # detection
  HiVT/              # motion forecasting
  integration/       # glue + time sync + handoff packets
  lightning_logs/    # HiVT checkpoints (local; gitignored)
  outputs/           # E2E / handoff exports (runtime)
```

의존성 파일:

| 파일 | 용도 |
|------|------|
| [`environment.yml`](environment.yml) | conda 환경 `HiVT` (python 3.8, lightning, mmcv 빌드용 gcc/nvcc) |
| [`requirements-torch.txt`](requirements-torch.txt) | PyTorch CUDA 12.1 wheel |
| [`requirements.txt`](requirements.txt) | av2 / mmdet 등 일반 pip 패키지 |
| [`requirements-pyg.txt`](requirements-pyg.txt) | PyG + scatter/sparse 등 |

`mmcv-full`은 위 목록에 **넣지 않습니다.** 아래 mmcv 절을 따르세요.

## 최초 환경 세팅 (전체 프로젝트)

Sparse4D / HiVT / integration 모두 **`conda` 환경 `HiVT` 하나**에서 동작합니다.

**전제:** Linux 또는 **WSL2** + NVIDIA GPU + 드라이버.  
Windows PowerShell만으로의 Sparse4D(mmcv CUDA 빌드)는 권장하지 않습니다.

> [`HiVT/README.md`](HiVT/README.md)의 논문용 설치(torch 1.8 등)는 **이 연결 워크스페이스용이 아닙니다.**

### 1) 클론

```bash
git clone <private-repo-url>
cd <repo-root>   # Sparse4D/ HiVT/ integration/ 이 보이는 루트
```

### 2) conda 환경

```bash
conda env create -f environment.yml
conda activate HiVT
```

이미 `HiVT` 이름이 있으면 `conda env remove -n HiVT` 후 다시 만들거나, yml의 `name:` 을 바꿔 생성하세요.

### 3) PyTorch → pip 패키지 → PyG

검증된 조합: **Python 3.8 · torch 2.4.1+cu121 · pytorch-lightning 1.5.2**

```bash
conda activate HiVT
cd <repo-root>

pip install -r requirements-torch.txt
pip install -r requirements.txt
pip install -r requirements-pyg.txt
```

CUDA 버전이 다르면 `requirements-torch.txt` / `requirements-pyg.txt`의 index·버전을 GPU/드라이버에 맞게 수정하세요.

### 4) mmcv-full 1.7.1 설치 (가장 중요 · 소스 빌드)

#### 왜 pip로 그냥 안 되나

- Sparse4D(mmdet 2.28)는 **mmcv-full 1.7.x** 가 필요합니다.
- 이 환경의 PyTorch는 **2.4.1+cu121** 입니다.
- PyPI/공식 prebuilt **mmcv-full 1.7 wheel은 torch 2.4용으로 제공되지 않습니다.**
- 그래서 `pip install mmcv-full==1.7.1` 은 실패하거나, 엉뚱한 wheel이 깔려 **import/ops가 깨집니다.**
- 해결: 레포에 포함된 소스 [`Sparse4D/third_party/mmcv`](Sparse4D/third_party/mmcv)를 **CUDA ops 포함해 직접 컴파일**합니다.

#### 스크립트가 하는 일 (`integration/setup_mmcv_hivt.sh`)

1. `conda activate HiVT` 확인  
2. 없으면 gcc/g++ 11, `cuda-nvcc` 12.1을 HiVT env에 설치 (`environment.yml`에 이미 있으면 생략)  
3. mmcv 소스를 **`$HOME/mmcv_build_hivt`** 로 복사 (Windows 마운트 `/mnt/c` 위에서 빌드하면 링크/성능 문제로 자주 실패)  
4. `FORCE_CUDA=1 MMCV_WITH_OPS=1` 로 CUDA 확장 컴파일  
5. `pip install --no-build-isolation --no-deps .` 로 **mmcv-full 1.7.1** 설치  
6. `from mmcv.ops import nms` 로 검증  

로그: `~/mmcv_build_hivt.log`

#### GPU 아키텍처 (`TORCH_CUDA_ARCH_LIST`)

스크립트 기본값은 **`8.9`** (Ada, 예: RTX 4080 Laptop)입니다.  
다른 GPU면 `integration/setup_mmcv_hivt.sh`에서 다음 줄을 수정하세요.

| GPU 예 | `TORCH_CUDA_ARCH_LIST` |
|--------|------------------------|
| RTX 30xx | `8.6` |
| RTX 40xx (Ada) | `8.9` |
| RTX 50xx (확인 필요) | 해당 compute capability |
| 여러 GPU 지원 | `8.6;8.9` 처럼 `;`로 나열 |

본인 GPU:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

#### 빌드 실행

```bash
conda activate HiVT
cd <repo-root>

# (권장) torch CUDA가 보이는지 먼저 확인
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

bash integration/setup_mmcv_hivt.sh
```

성공 시 마지막에 비슷하게 출력됩니다.

```text
OK 1.7.1 ops
```

수동 확인:

```bash
python -c "import mmcv; from mmcv.ops import nms; print(mmcv.__version__)"
```

#### 자주 나는 문제

| 증상 | 대처 |
|------|------|
| `/mnt/c/...` 에서 빌드 중 링크/권한 오류 | 스크립트대로 `$HOME/mmcv_build_hivt` 사용 (스크립트가 복사함). 레포만 `/mnt/c`에 있어도 됨 |
| `nvcc` / `crypt.h` 없음 | `environment.yml`로 env를 다시 만들거나 스크립트가 conda로 설치하게 둠 |
| `mmcv.ops` import 실패 | `pip uninstall -y mmcv mmcv-full` 후 스크립트 재실행 |
| 빌드는 됐는데 Sparse4D만 느림/경고 | deformable CUDA ext는 선택. 라이브 스모크는 PyTorch sampling으로도 동작 (`use_deformable_func=False`) |
| torch 1.x용 mmcv wheel을 깔아 버림 | 반드시 제거 후 **소스 빌드**로 교체 |

소요 시간: 대략 **15–40분** (CPU/디스크에 따라 다름).

### 5) 가중치 배치 (레포 미포함 · 별도 전달)

가중치: [Google Drive](https://drive.google.com/drive/folders/11BGcHYZHCIN4TDXCRHS6aWas7qbVYl5j?usp=sharing)

| 모델 | 파일 | 둘 위치 |
|------|------|---------|
| Sparse4D 10Hz | `sparse4dv3_r50_b2d_10hz_epoch14.pth` | `Sparse4D/checkpoints/` |
| HiVT | `epoch=55-step=349887.ckpt` | `lightning_logs/version_1/checkpoints/` |

```bash
mkdir -p Sparse4D/checkpoints lightning_logs/version_1/checkpoints
# Drive에서 받은 파일을 위 경로에 복사
```

맵(`integration/maps/`)·fixtures(`integration/fixtures/`)가 레포에 있으면 스모크에 그대로 사용합니다.  
`external_data/`·`outputs/`는 레포에 포함하지 않습니다.

### 6) 동작 확인

```bash
conda activate HiVT
cd <repo-root>
export PYTHONPATH=HiVT:.

python -u integration/run_e2e_smoke.py --device cuda:0
python -u integration/run_live_sparse4d.py --device cuda
```

둘 다 성공하면 기본 연결 환경이 준비된 것입니다.

### 7) (선택) Bench2Drive 전체 E2E

데이터는 [`Sparse4D/README.md`](Sparse4D/README.md)의 `download_and_extract` / `prepare_episode`로  
**같은 HiVT conda**에서 준비한 뒤:

```bash
PYTHONPATH=HiVT:. python -u integration/run_b2d_e2e.py --max_windows 10 --history_source both
```

## 일상 실행

```bash
conda activate HiVT
cd <repo-root>
export PYTHONPATH=HiVT:.

python -u integration/run_e2e_smoke.py --device cuda:0
python -u integration/run_b2d_e2e.py --max_windows 10 --history_source both
python -u integration/export_handoff_packet.py --max_windows 5
python -u integration/run_live_sparse4d.py --device cuda
```

## 하위 프로젝트 README

- Sparse4D: [`Sparse4D/README.md`](Sparse4D/README.md)
- HiVT: [`HiVT/README.md`](HiVT/README.md)
- integration: [`integration/README.md`](integration/README.md)
