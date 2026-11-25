# Sparse4D 인계자료 (이 워크스페이스에서의 쓰임)

이 폴더(`Sparse4D/`)가 탐지 프로젝트 루트입니다. HiVT와는 [`../integration/`](../integration/)으로 연결합니다.

```text
Sparse4D/
  checkpoints/     sparse4dv3_r50_b2d_10hz_epoch14.pth
  projects/ handoff/ external_data/ ...
  docs/            요약본·상세인계서·출처
  weights_notes/   가중치 인계 문서·원본 pth
```

- 궤적 예측은 [`../HiVT/`](../HiVT/) 를 사용합니다.
- e2e용 탐지/pose 예제: [`../integration/fixtures/`](../integration/fixtures/)
- 연결 실행: [`../integration/README.md`](../integration/README.md) · [`../STRUCTURE.md`](../STRUCTURE.md)
