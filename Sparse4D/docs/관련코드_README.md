# 관련 코드 (Sparse4D vendor)

워크스페이스 루트의 AV2 HiVT와 연결하는 **Sparse4D** 소스입니다.  
인계 AV1 HiVT 트리는 제거되었고, e2e 예제는 `../../integration/fixtures/`로 옮겼습니다.

- **Sparse4D/**: live 추론. 가중치 `Sparse4D/checkpoints/sparse4dv3_r50_b2d_10hz_epoch14.pth`
- **tests/**: Sparse4D·전처리 CPU 회귀 (`test_hivt_contract.py`는 HiVT vendor 제거로 제외)

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
```
