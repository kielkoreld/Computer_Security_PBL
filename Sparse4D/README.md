# Sparse4D: Bench2Drive 연결용 추론 코드

> 워크스페이스의 탐지 프로젝트 폴더입니다. HiVT 연결은 [`../integration/`](../integration/), 전체 구조는 [`../STRUCTURE.md`](../STRUCTURE.md).

실제로 사용한 호환성 수정판 Sparse4D 소스입니다. `projects/`, `tools/`, `local_train.sh`, `local_test.sh`는 Sparse4D 원래 구조입니다. 원본 설명은 [공식 README](https://github.com/HorizonRobotics/Sparse4D/tree/249ffbb695f4e9db628d953e2bf6d36de04bbb69)에서 확인할 수 있고, 배포에 필요한 `LICENSE`는 이 폴더에 남겼습니다. 인계용 작업은 아래 `handoff/` 명령을 사용합니다.

**현재 HiVT 연결 기본 가중치는 10Hz epoch14** (`checkpoints/sparse4dv3_r50_b2d_10hz_epoch14.pth`, 인계 문서·원본: `weights_notes/`).  
입력·캐시는 `prepare_episode.py`의 `--frame-step 1`(10Hz)로 준비하며, 2Hz→10Hz 이력 보간은 사용하지 않습니다.  
인계 PDF/요약: `docs/`

## 구조와 역할

- `projects/`: 실제 사용한 Sparse4Dv3 모델, dataset, config, custom CUDA op. 모델 연산은 기존 실행 소스와 같습니다.
- `third_party/mmcv/`: 실제 사용한 MMCV 1.7.1 소스. 자체 LICENSE와 고지문을 보존합니다.
- `handoff/core/`: CARLA 좌표 변환, GT-free 모델 입력/출력, 공통 tracker, HiVT 이력 adapter, 원본 archive 검증·변환.
- `handoff/training/`: 공식 입력 RGB 캐시, 정적 GT 보정, 원래 optimizer/loss/배치/학습 loop, 추론 구현. `model_builder.py`는 기존 모델 생성·공통 클래스 loss mask·DN 처리를 그대로 유지합니다.
- `handoff/evaluation/`: nuScenes-devkit의 AP 함수를 호출하는 CARLA 공통 클래스 평가. 공식 nuScenes NDS 평가가 아닙니다.
- `handoff/tools/`: 담당자가 사용하는 실행 진입점.
- `handoff/protocol/`: 고정 train/validation/evaluation44 분할 및 학습 선택 정보.
- `handoff/settings/dataset_download.json`: 고정 Bench2Drive 원격 revision/파일 크기 목록.
- `handoff/examples/`: 저장된 공식 입력 3시점과 예측 예제.
- `handoff/cache/normalization_lut.npy`: 공식 입력을 정확히 복원하는 LUT.
- `handoff/artifacts/`: 가중치 배치 도구가 만드는 anchor 초기값.
- `checkpoints/`: Sparse4D 가중치. 기본은 `sparse4dv3_r50_b2d_10hz_epoch14.pth` (10Hz). 구 2Hz 파일은 `sparse4dv3_r50_2hz_epoch12.pth`로 보관할 수 있습니다.
- `../tests/`: 두 repository를 함께 검사하는 CPU 회귀 테스트.

이하 명령의 현재 디렉터리는 이 README가 있는 `Sparse4D`입니다. 경로에 공백이 있으면 따옴표로 감싸십시오.

## 환경

순수 CPU 회귀 테스트와 archive/좌표 전처리에는 NumPy, SciPy, Pillow가 필요합니다. 다운로드 시에는 `huggingface_hub`가 추가로 필요합니다. 예:

```bash
python3.12 -m venv .venv-data
.venv-data/bin/python -m pip install numpy==1.26.4 scipy==1.14.1 Pillow==11.1.0 huggingface_hub
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv-data/bin/python -m unittest discover -s ../tests -v
```

Sparse4D 실제 모델 로딩·공식 이미지 캐시·학습·추론에는 별도의 모델 환경이 필요합니다. 당시 기준은 Python 3.12, PyTorch 2.7.1/cu128, CUDA toolkit 12.8, MMCV 1.7.1 소스 빌드, MMDetection 2.28.2입니다. 모델을 쓰지 않는 CPU 검사와 별개의 환경입니다.

```bash
export CUDA_HOME=/실제/CUDA-12.8/경로
export TORCH_CUDA_ARCH_LIST=12.0  # 당시 GPU 기준. 담당자 GPU에 맞게 지정
bash handoff/tools/setup_sparse4d_cuda.sh
```

설치 스크립트는 새 `.venv-models`를 만들고 패키지를 내려받아 CUDA extension을 빌드합니다. 담당자 GPU/드라이버/컴파일러에 맞는지 확인한 뒤 실행하십시오. 새 환경의 설치와 GPU 실행은 별도 확인이 필요합니다. 모델 환경에서 다운로드도 수행하려면 해당 환경에 `huggingface_hub`를 별도로 설치하십시오.

## Sparse4D 가중치 연결과 입력 예제

이미 받은 파일만 연결합니다. 초기 pretrained 파일은 새 미세조정에, epoch12 파일은 제공된 결과의 모델 로딩/추론에 사용합니다. 실제 보유 파일에 해당하는 옵션만 지정할 수 있습니다.

```bash
.venv-models/bin/python handoff/tools/configure_weights.py \
  --sparse-initial /별도/가중치/sparse4dv3_r50.pth \
  --sparse-epoch12 /별도/가중치/best_epoch_012.pth
.venv-models/bin/python handoff/tools/sparse4d_example.py
```

기본은 CPU strict loading과 저장 입력 3시점의 계약 검사입니다. `--forward`를 명시하면 GPU 추론을 실행합니다. 파일은 기본 심볼릭 링크이며 `--copy`로 복사할 수 있고, 기존 파일을 덮어쓰지 않습니다. 가중치에서 anchor 배열을 추출해 `handoff/artifacts/nuscenes_kmeans900_from_checkpoint.npy`를 생성합니다. 신뢰하는 checkpoint만 로드하십시오.

## 다운로드 및 한 클립 전처리

```bash
.venv-data/bin/python handoff/tools/download_and_extract.py --split sample
# 위 명령은 네트워크 없는 계획 출력입니다. 실제 다운로드/해시 검증/추출:
.venv-data/bin/python handoff/tools/download_and_extract.py --split sample --download --extract
```

`--split`은 sample/evaluation44/train/validation/all을 지원합니다. 기본 archive 위치는 `external_data/archives`, 추출 위치는 `external_data/extracted_work/data/extracted`입니다. `handoff/tools/bench2drive_subset.py`는 시나리오별 N개를 고르는 별도 원본 도구이며, 고정 인계 분할과 다른 선택임을 유의하십시오.

```bash
EP='DynamicObjectCrossing_Town12_Route17906_Weather0_02-06-18-12-57'
.venv-models/bin/python handoff/tools/prepare_episode.py \
  --extracted-root external_data/extracted_work/data/extracted \
  --episode "$EP" --output "external_data/prepared/$EP"
```

출력은 새 디렉터리에만 생성합니다. 변환의 8개 geometry 기준과 이동 기반 dt 검사 후 정적 GT 보정과 공식 영상 캐시를 만듭니다. `--metadata-only`는 영상 캐시를 생략하며, 그 결과만으로 학습 작업공간을 만들 수는 없습니다. 정지 장면 등으로 dt 표본이 없으면 `motion_timing.verified=false`를 기록합니다. 이 관문은 원본 archive CRC 검사를 대체하지 않습니다. archive 검증은 다운로드/추출 단계에서 수행합니다.

## 새 학습·추론 작업공간

`--prepared-root` 아래에 `클립명/{converted,prepared,...}` 형식의 결과를 둡니다. infer는 evaluation44 전체, train은 train+validation+evaluation44 전체를 요구합니다. 위 대표 한 클립 예제만으로 전체 작업공간 생성이 성공하지 않습니다.

```bash
.venv-models/bin/python handoff/tools/create_workspace.py --mode train \
  --prepared-root /전체/전처리결과 --output /새/학습작업공간
.venv-models/bin/python /새/학습작업공간/training/train_portable.py
```

작업공간은 `training/`, `protocol/`, `cache/`, `data/`, `converted/`, `logs/`와 repository/core/anchor 링크로 구성됩니다. 원본 repo와 전처리 디렉터리를 이동·삭제하면 링크가 끊어집니다. 학습은 원래 2Hz loss/optimizer/gradient accumulation/early stopping을 유지하며, 과거 호스트 운영 관문과 별도 8클립 비교 probe만 제외합니다. 실제 생성된 `train_portable.py` 자체도 resume fingerprint에 포함됩니다. 해시는 모든 데이터나 간접 의존성의 완전한 무결성을 보장하는 것은 아닙니다.

epoch12 평가용 작업공간을 만들고 공식 캐시로 44클립 추론:

```bash
.venv-models/bin/python handoff/tools/create_workspace.py --mode infer \
  --prepared-root /전체/전처리결과 --output /새/추론작업공간
.venv-models/bin/python /새/추론작업공간/training/infer44.py --tag best
.venv-models/bin/python handoff/tools/evaluate44.py \
  --predictions /새/추론작업공간/after_predictions \
  --gt /새/추론작업공간/data --output /새/평가결과
```

`--predictions` 아래에 `sparse4d/클립명/predictions.jsonl`이 있어야 합니다. 새 학습이 완료된 작업공간에서도 `training/infer44.py --tag best`를 사용할 수 있습니다. `--tag pretrained`는 초기 Sparse4D checkpoint를 사용합니다. 

## 검증 범위

공통 회귀 명령은 `python -m unittest discover -s ../tests -v`입니다. CPU 테스트는 geometry 관문, 작업공간 생성·파일 지문, HiVT 전처리와 저장 출력 좌표복원 및 오류 보고를 검사합니다. 실제 모델 import/가중치 strict load/GPU forward/학습이나 과거 528회 HiVT 실행을 재검증한 결과가 아닙니다. 실제 모델 실행은 각 환경과 데이터/가중치를 준비한 뒤 담당자가 수행해야 합니다.
