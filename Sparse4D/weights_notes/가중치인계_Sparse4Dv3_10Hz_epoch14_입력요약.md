# Sparse4Dv3 10Hz epoch 14 가중치 인계 — 입력 요약

전달하는 모델은 **Bench2Drive의 6카메라 영상으로 10Hz 추가 학습한 Sparse4Dv3-R50 객체 탐지기**입니다. 기존 2Hz epoch 12에서 시작해 학습 835클립으로 총 19 epoch를 진행했고, 검증 221클립의 손실이 가장 낮은 **추가 epoch 14**를 선정했습니다. HiVT 모델 가중치와는 다릅니다.

## 1. 가중치와 모델 로딩

파일명: `Sparse4Dv3_R50_Bench2Drive_10Hz_finetuned_best_additional_epoch_014.pth`

원래 프로젝트의 `model_exports/Sparse4Dv3_R50_Bench2Drive_10Hz_tg-563161899/`에 전달용 사본이 있습니다. 문서 ZIP에는 가중치를 넣지 않았습니다. 이 파일은 `state_dict`와 선정 정보를 담고 있으며 optimizer는 포함하지 않습니다.

공식 Sparse4Dv3 temporal R50 구조, 900개 anchor 파일과 호환 환경을 준비하고 `scripts/engine.py`의 `build(checkpoint=..., default_dt=0.1)`처럼 전체 state를 strict 로딩합니다. 여기서 scripts는 원래 프로젝트의 `research_run/continuations/20261110_finetune_10hz/scripts`입니다. 코드 사본은 문서 ZIP의 `04_구현참고`에 같은 경로로 들어 있습니다.

## 2. 영상과 보정·시간 입력

같은 시점의 1600×900 RGB 6장을 다음 순서로 넣었습니다.

`CAM_FRONT → CAM_FRONT_RIGHT → CAM_FRONT_LEFT → CAM_BACK → CAM_BACK_LEFT → CAM_BACK_RIGHT`

영상을 704×396으로 줄이고 위쪽 140픽셀을 잘라 704×256으로 만듭니다. RGB의 0~255 값에 `(value−mean)/std`를 적용합니다. Mean은 `[123.675,116.28,103.53]`, std는 `[58.395,57.12,57.375]`입니다. 별도로 0~1로 나누지 않습니다. 투영행렬에도 같은 축소·자르기를 반영해야 합니다.

| 필드 | 추론 batch 1의 형상·자료형 | 뜻 |
| --- | --- | --- |
| img | [1,6,3,256,704], float32 | 정규화한 RGB 영상 |
| projection_mat | [1,6,4,4], float32 | 모델 좌표에서 최종 영상으로의 투영 |
| image_wh | [1,6,2], float32 | 카메라마다 [704,256] |
| timestamp | [1], float64 | 초 단위 시각, 인접 입력 간격 0.1초 |
| img_metas | 길이 1의 list | T_global·T_global_inv: 각 [4,4], float64 |

`T_global`은 모델 기준 좌표(오른쪽·앞·위)를 공통 세계 좌표로 바꿉니다. 세계 좌표는 CARLA 원본 Y축을 반전한 오른손 좌표입니다. 단위는 거리 m, 각도 rad, 속도 m/s입니다. **LiDAR 점군과 Depth 영상은 모델에 넣지 않았습니다.** 센서 좌표 변환에는 LiDAR 기준 메타데이터를 사용했습니다.

각 경로를 0.1초 간격으로 순서대로 넣고, 같은 경로에서는 과거 상태를 유지하며 경로가 바뀔 때 instance bank를 초기화합니다. 기본 시간 간격은 0.1초, confidence decay는 0.6입니다. 학습은 최대 batch 4, 평가는 batch 1이었습니다. 학습 때만 GT 박스[M,9]·label[M]·정답 ID를 손실 계산에 추가했고 **평가 추론 입력에는 GT를 넣지 않았습니다.**

<div class="page-break"></div>

## 3. 출력과 HiVT 연결 전 확인할 사항

모델은 시점마다 상위 300개 탐지 후보의 박스·점수·클래스를 출력합니다. 박스는 `[x,y,z,l,w,h,yaw,vx,vy,vz]`의 10성분입니다. 이번 GT에서는 vz를 직접 감독하지 않았습니다. 저장하는 공통 JSON은 중심·yaw·속도를 세계 좌표로 변환하고 크기·점수·클래스·timestamp를 함께 담습니다. 변환 코드는 `research_run/scripts/model_inputs.py`의 `export_boxes()`입니다.

**현재 평가 JSON의 `track_id`는 null입니다.** 추론 결과를 저장할 때 모델의 native ID를 전달하지 않았으므로, 이 파일에서 HiVT 입력을 만들려면 시점 간 동일 객체를 연결하는 추적 처리가 필요합니다. 시간 특징을 사용하는 탐지기라는 사실만으로 외부 추적 ID가 완성되는 것은 아닙니다. 상위 300개를 모두 확정 객체로 사용하지 말고 대상 클래스·신뢰도·이력 조건을 정해야 합니다.

HiVT에는 RGB나 Sparse4D 가중치를 직접 넣지 않습니다. 연결된 객체의 시간별 XY 위치, 관측 유효성 마스크, 객체 간 관계·지도 정보를 해당 HiVT 구현의 형식으로 구성합니다. 기존 Argoverse용 HiVT는 과거 20시점과 미래 30시점을 다룹니다. 0.1초 간격의 과거 20시점에서 첫 시점과 마지막 시점의 차이는 1.9초입니다. 연속 관측이 10Hz라면 주기를 맞추기 위한 보간은 필요 없지만 탐지 누락·ID 교체·좌표·지도 입력은 별도 처리가 필요합니다.

## 4. 확인된 성능과 미수행 범위

선정 검증 손실은 **6.690305714**, 기존 평가 44클립 11,645시점의 10Hz mAP는 **82.4727%**입니다. 정답이 있는 car·truck·pedestrian·bicycle 4클래스의 거리 기반 AP 평균입니다. Bus·motorcycle은 평가 GT가 없어 측정하지 못했습니다. 전체 1,100클립의 mAP나 HiVT의 궤적 예측 성능을 뜻하지 않습니다.

**이번 epoch 14 모델의 HiVT 연결 비교·미래 궤적 정확도 평가는 수행하지 않았습니다.** 과거 2Hz 모델로 수행한 연결 확인 결과를 이번 모델의 검증으로 대신하지 않습니다.

원본→정답·모델 입력은 전체 과정 보고서 4~6절, 실제 저장 출력 예제는 9절, 실행 환경과 진입점은 14~15절을 참고해 주세요. PDF는 그림을 포함하며 단독으로 전달할 수 있습니다.
