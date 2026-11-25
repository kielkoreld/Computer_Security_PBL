"""변환 결과의 좌표·시간 검증. 모델을 로딩하지 않는 인계용 검사다."""
import math

# prepare_b2d_expansion.py의 과거 검사 기준과 동일하다.
GEOMETRY_LIMITS = {
    'rotation_orthogonality': 1e-5,
    'rotation_determinant': 1e-5,
    'matrix_inverse_roundtrip': 1e-8,
    'camera_location_error_m': .005,
    'center_vs_corners_m': .002,
    'extent_vs_corners_m': .002,
    'projection_path_difference_px': 1e-4,
    'augmentation_difference_px': 1e-6,
}


def validate_conversion_report(report, episode):
    """기준 미달/누락은 ValueError. 움직임 표본 없는 시간 추정은 미검증으로 표시."""
    def require(ok, message):
        if not ok:
            raise ValueError(f'{episode}: {message}')

    def finite_number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    require(report.get('episode_id') == episode, '변환 보고서의 클립 이름이 다릅니다')
    for key in ['frames_10hz', 'selected_frames']:
        value = report.get(key)
        require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
                f'{key}는 양의 정수여야 합니다')
    require(report['selected_frames'] <= report['frames_10hz'], '선택 프레임 수가 원본보다 많습니다')
    require(report.get('selected_hz') in (2, 10, 2.0, 10.0),
            '변환 주기는 2Hz 또는 10Hz여야 합니다')
    checks = report.get('checks')
    require(isinstance(checks, dict), '좌표 검사 결과가 없습니다')
    gates = {}
    for key, limit in GEOMETRY_LIMITS.items():
        row = checks.get(key, {})
        require(isinstance(row, dict), f'{key}: 검사 형식 오류')
        count = row.get('count')
        require(isinstance(count, int) and not isinstance(count, bool) and count > 0,
                f'{key}: 유효한 검사 표본이 없습니다')
        maximum = row.get('max')
        require(finite_number(maximum) and maximum >= 0, f'{key}: 최대 오차가 유한한 비음수가 아닙니다')
        require(maximum < limit, f'{key}: 최대 오차 {maximum}가 기준 {limit} 미만이 아닙니다')
        gates[key] = dict(maximum=maximum, limit=limit, passed=True, samples=count)
    timing = checks.get('inferred_dt_from_ego_motion_s')
    if timing is None:
        timing_result = dict(verified=False, reason='속도 2m/s 초과 자차 이동 표본이 없어 독립 시간 추정을 수행하지 못함')
    else:
        require(isinstance(timing, dict), '독립 시간 검사 형식 오류')
        count = timing.get('count')
        require(isinstance(count, int) and not isinstance(count, bool) and count > 0,
                '독립 시간 검사 표본이 없습니다')
        dt = timing.get('median')
        require(finite_number(dt) and .098 < dt < .102,
                f'독립 시간 추정 중앙값 {dt}가 (0.098, 0.102)초 범위 밖입니다')
        timing_result = dict(verified=True, median_s=dt, samples=count, lower_exclusive_s=.098, upper_exclusive_s=.102)
    return dict(passed=True, episode=episode, selected_frames=report['selected_frames'],
                geometry_gates=gates, motion_timing=timing_result,
                scope='변환 보고서의 수치 기준 검사. 원본 archive CRC나 카메라 가림 검증을 대신하지 않음')
