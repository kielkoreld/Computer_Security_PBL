# 워크스페이스 구조

목표: **Bench2Drive → Sparse4D(10Hz) → AV2 HiVT → T_fuse 위험 패킷**.

## 3분할

| 경로 | 역할 |
|------|------|
| [`Sparse4D/`](Sparse4D/) | Sparse4Dv3 탐지 (`projects/`, `handoff/`, `checkpoints/`, `external_data/`) |
| [`Sparse4D/docs/`](Sparse4D/docs/) | 인계 요약·상세 문서 |
| [`Sparse4D/weights_notes/`](Sparse4D/weights_notes/) | 가중치 인계 문서·원본 pth 포인터 |
| [`HiVT/`](HiVT/) | AV2 HiVT (`models/`, `datasets/`, `train.py`, `eval.py`, API) |
| [`integration/`](integration/) | 브리지·맵·시간동기·E2E·핸드오프 export |

## integration (연결)

| 파일 | 역할 |
|------|------|
| `paths.py` | `REPO_ROOT` / `HiVT/` / `Sparse4D/` 경로·`sys.path` |
| `run_b2d_e2e.py` | B2D 클립 E2E: AP + ADE/FDE·지연 |
| `run_live_sparse4d.py` | Sparse4D 단독 추론 |
| `run_e2e_smoke.py` | fixtures → HiVT 스모크 |
| `export_handoff_packet.py` | JSON+NPZ 핸드오프 패킷 |
| `bridge.py` | 10Hz 탐지/GT → AV2 윈도우 |
| `av2_adapter.py` | HiVT `TemporalData` 필드 |
| `b2d_map.py` | Town HD맵 조회 |
| `time_sync.py`, `risk_fusion.py` | `T_fuse` 동기화 |
| `maps/`, `fixtures/` | 맵 패키지·스모크 픽스처 |

## 런타임 (루트, gitignore)

`outputs/`, `logs/`, `lightning_logs/`

## 실행

```bash
conda activate HiVT
cd <repo-root>
export PYTHONPATH=HiVT:.

python -u integration/run_e2e_smoke.py --device cuda:0
python -u integration/run_b2d_e2e.py --max_windows 10 --history_source both
python -u integration/export_handoff_packet.py --max_windows 5
```

시간 동기화: [`integration/docs/TIME_SYNC.md`](integration/docs/TIME_SYNC.md)
