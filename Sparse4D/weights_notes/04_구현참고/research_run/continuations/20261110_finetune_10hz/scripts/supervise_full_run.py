"""기존 epoch 검수 체인을 foreground로 실행하고 현재 정식 작업에도 진행 보고를 남긴다.

분리 실행이나 서비스 설치를 하지 않는다. 감독 프로세스가 종료 신호를 받으면
자식 감독으로 전달하고 상태 저장을 기다린다. 후속 대화의 1시간 제한은 이
프로그램으로 제거되지 않으므로 정식 작업 또는 직접 열린 터미널에서 실행한다.
"""
import argparse
import signal
import subprocess
import time
from common import *

child = None
stopped = False

def interrupt(sig, frame):
    global stopped
    stopped = True
    if child is not None and child.poll() is None:
        child.send_signal(sig)

def mirror(source, target, offset):
    if source == target or not source.exists():
        return offset
    with source.open('rb') as f:
        f.seek(offset)
        data = f.read()
    end = data.rfind(b'\n') + 1
    if end:
        block = data[:end]
        for line in block.splitlines():
            json.loads(line)
        with target.open('ab') as f:
            f.write(block)
        return offset + end
    return offset

def snapshot(current, running):
    """별도 새 TASK_ID에서도 오래된 STATE를 현재 실행으로 오인하지 않게 한다."""
    source = P/'training/status.json'
    state = read(source) if source.exists() else {}
    fields = ['complete','phase','epoch','next_group','current_groups','current_frames',
              'current_loss','train_frames_total','updates_total','best_epoch',
              'best_val_loss','bad_epochs','updated_utc','stop_reason']
    summary = {k:state[k] for k in fields if k in state}
    live = dict(checked_utc=now(),foreground_pipeline_running=running,
                last_training_record=summary,source=str(source),
                note='감독 체인의 실행 여부와 마지막 학습 기록을 구분한다. 이후 평가/보고서 단계에서는 학습 complete가 true여도 체인은 실행 중일 수 있다.')
    save(current/'live_status.json',live)
    text = '\n'.join([
        f'# {current.name} 현재 상태', '',
        f"확인 시각: {live['checked_utc']}", '',
        f"foreground 감독 체인: {'실행 중' if running else '종료됨'}", '',
        '학습835클립213,580프레임 / 검증221클립53,807프레임 / 평가44클립11,645프레임. 학습·검증·평가 역할 유지.', '',
        '아래는 마지막 기록된 학습 진행이다. 매 epoch 전체 검증221클립의 loss와 patience5를 유지한다. 첫 epoch의 전체 과정 검수 후 반복 심층검수와 별도8클립 AP는 사용자 우선 지시에 따라 생략한다.', '',
        '```json', json.dumps(summary,ensure_ascii=False,indent=2), '```', '',
        f'실행 계획: {current}/PLAN_KO.md',
        f'명령/종료 결과: {current}/full_10hz_foreground_execution.json',
        f'감독 출력: {current}/full_10hz_foreground.log',
        f'실제 학습 로그: {P}/logs/train_main.log',
        f'과학적 결과와 관문: {P}', '',
        '전체 학습 후 평가44/검증221, HiVT 연결 비교, 상세보고서 전페이지 검수까지 남아 있다. 체인 반환만으로 전체 작업 완료를 선언하지 않는다.', ''
    ])
    dest=current/'STATE.md';temp=dest.with_suffix('.tmp');temp.write_text(text);temp.replace(dest)

def main():
    global child
    a = argparse.ArgumentParser()
    a.add_argument('--run-directory', type=Path, required=True)
    args = a.parse_args()
    current = args.run_directory.resolve()
    current.relative_to(ROOT/'.agent_runs')
    current.mkdir(parents=True,exist_ok=True)
    # 별도 모델/학습기를 만들지 않고 검증된 관문 체인을 사용한다.
    jobs = P/'protocol/resume_remaining_jobs.json'
    cmd = [str(PY),str(P/'scripts/run_jobs.py'),str(jobs)]
    original_progress = RUN/'progress_reports.jsonl'
    target_progress = current/'progress_reports.jsonl'
    offset = original_progress.stat().st_size if original_progress.exists() else 0
    signal.signal(signal.SIGINT,interrupt)
    signal.signal(signal.SIGTERM,interrupt)
    execution = current/'full_10hz_foreground_execution.json'
    state = dict(complete=False,phase='running',started_utc=now(),command=cmd,
                 training_path=str(P/'training'),source_run_directory=str(RUN),
                 next_required_after_pipeline='PDF 전체 페이지의 내용·그림 검수 후 editorial_review.json 및 package_results.py. 코드 체인 완료가 전체 작업 완료는 아님.')
    save(execution,state)
    with (current/'full_10hz_foreground.log').open('a') as f:
        child = subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,
            env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'4'})
        try:
            previous_status = None
            while child.poll() is None:
                offset = mirror(original_progress,target_progress,offset)
                source=P/'training/status.json'
                modified=source.stat().st_mtime_ns if source.exists() else 0
                if modified != previous_status:
                    snapshot(current,True);previous_status=modified
                time.sleep(1)
            code = child.wait()
        finally:
            # 감독 내부 오류도 실행 중인 자식을 방치하지 않는다.
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)
                child.wait()
            offset = mirror(original_progress,target_progress,offset)
            state.update(complete=False,phase='stopped' if stopped else 'pipeline_returned',
                         exit_code=child.returncode,finished_utc=now(),stopped=stopped,
                         pipeline_succeeded=child.returncode==0 and not stopped)
            save(execution,state)
            snapshot(current,False)
    child = None
    if stopped:
        raise SystemExit(130)
    raise SystemExit(code)

if __name__ == '__main__':
    main()
