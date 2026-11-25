# Sparse4D → HiVT 연결 (integration)

루트 워크스페이스의 **연결·시간 동기·핸드오프** 패키지입니다.

- Sparse4D 코드/가중치: [`../Sparse4D/`](../Sparse4D/)
- HiVT 코드: [`../HiVT/`](../HiVT/)
- 경로 헬퍼: [`paths.py`](paths.py)
- 시간 동기화: [`docs/TIME_SYNC.md`](docs/TIME_SYNC.md)
- 전체 구조: [`../STRUCTURE.md`](../STRUCTURE.md)

## 데이터 흐름

1. **클래스**: Sparse4D `objects[].class` → tracker → HiVT `actor_type`
2. **맵**: Bench2Drive Town HD맵 → `maps/TownXX_lanes_rh.npz` (시작 시 1회 로드)
3. **이력**: Sparse4D **10Hz** 탐지(또는 GT) → HiVT 0.1s 격자 → ADE/FDE
4. **시계**: `risk_fusion`이 ego / boxes / HiVT futures를 **`T_fuse`** 로 정렬

## 최초 설치

전체 프로젝트 세팅은 루트 [`../README.md`](../README.md)의 **「최초 환경 세팅」** 을 따르세요.

## 실행

```bash
conda activate HiVT
cd <repo-root>
export PYTHONPATH=HiVT:.

python -u integration/run_b2d_e2e.py --max_windows 10 --history_source both
python -u integration/run_live_sparse4d.py --device cuda
python -u integration/run_e2e_smoke.py --device cuda:0
python -u integration/export_handoff_packet.py --max_windows 5
```

Town12 맵 최초 1회:

```bash
PYTHONPATH=HiVT:. python integration/_download_town12_map.py
PYTHONPATH=HiVT:. python integration/_extract_centerlines.py
```

## 가중치 (별도 전달 · 레포 미포함)

받은 파일을 아래 경로에 두세요.

- Sparse4D: `Sparse4D/checkpoints/sparse4dv3_r50_b2d_10hz_epoch14.pth`
- HiVT: `lightning_logs/version_1/checkpoints/epoch=55-step=349887.ckpt`
