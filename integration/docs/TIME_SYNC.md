# 위험 판단용 시간 동기화

목표: Bench2Drive · Sparse4D · HiVT가 **서로 다른 지연**을 갖고 있어도, 위험 판단이 **같은 소비 시각 `T_fuse`**에서 세 정보를 해석한다.

구현: `integration/risk_fusion.py` (`build_risk_packet`, `aligned_at_t_fuse`)

## 0. 출력이 “어느 시각의 기하”인가 (코드 확인)

| 모듈 | 기하/예측이 붙는 시각 | 결과가 손에 들어오는 시각 | 근거 |
|------|----------------------|---------------------------|------|
| **Sparse4D** | **T_obs** (카메라 프레임, 기본 10Hz) | T_obs + L_s4d | 탐지 `center`는 해당 프레임 timestamp의 위치. 2Hz→10Hz 이력 보간 없음 |
| **HiVT** | **T_obs 기준** future: `T_obs+(k+1)·0.1` | T_obs + L_s4d + L_hivt | `world = local_delta + positions[hist-1]`; hist 끝 = `end_timestamp`(=T_obs). 추론 종료 시각으로 리베이스하지 않음 |
| **Bench2Drive** | pose의 `timestamp` | 시뮬 상태 시각 | `poses_10hz` |

핵심: **HiVT 출력 시계 ≠ 추론 완료 시각.**  
예측은 “사진을 본 순간(T_obs) 이후 0.1초 격자”이고, 추론이 끝나는 건 그보다 늦다.

## 1. 정렬 기준 (소비 시계)

공동 위험 판단의 “지금”은 **HiVT 추론이 끝난 뒤**:

```
T_fuse := T_hivt_ready = T_obs + L_s4d + L_hivt   (기본값)
```

`fusion_clock: "hivt_ready"` 로 패킷에 기록된다.

## 2. 세 소스를 T_fuse에 맞추는 방법

| 소스 | 보정 방법 | 보정량 의미 |
|------|-----------|-------------|
| **Bench2Drive** | `poses_10hz`에서 ego **보간** | T_fuse 시점 자차 |
| **Sparse4D** | `velocity_xy`로 **등속 전파** | Δt = T_fuse − T_obs (= L_s4d+L_hivt) |
| **HiVT** | `t_k ≥ T_fuse`인 첫 step부터 사용 (**skip**) | T_obs 격자 중 이미 지난 점 제거 |

모듈마다 **방법**은 다르고, **목표 시각은 하나(T_fuse)** 이다.  
모듈별 ready 시각을 각각 fuse 기준으로 쓰면 같은 판단 순간에 시계가 갈라진다.

## 3. 위험 판단이 볼 필드

`build_risk_packet(...)['aligned_at_t_fuse']`:

```text
t_obs, t_fuse, t_hivt_ready, fusion_clock, same_clock
bench2drive.ego_at_fuse.ego_xy
sparse4d.objects_at_fuse[].center_xy_at_fuse
hivt.position_at_fuse / future_world_xy / future_abs_times
```

`summarize_aligned_packet(risk)` → `same_clock=true`이면 세 도메인이 같은 시계.

## 4. 숫자 예

`T_obs=10.0`, `L_s4d=0.129`, `L_hivt=0.015` → `T_fuse=T_hivt_ready≈10.144`

- HiVT: step@10.1은 과거 → skip≥1, 소비는 ≈10.2부터
- Sparse4D: 각 박스 `center + v * 0.144`
- B2D: poses에서 10.144초 ego XY 보간

## 5. 한계

- Sparse4D 전파는 짧은 Δt 등속 가정 (속도 없으면 T_obs 유지 + 플래그)
- B2D는 제공된 `poses_10hz` 기준
- ADE는 raw HiVT(T_obs 격자) 기준; 위험 판단 입력은 `aligned_at_t_fuse`
