"""기존 선택 코드의 전체 manifest를 받아 누락 파일만 다운로드·SHA256 검증한다."""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import threading
import time

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
from huggingface_hub import HfApi, RepoFile, hf_hub_download


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    assert manifest['archive_count'] == 1100 and manifest['scenarios'] == 44
    assert 1 <= args.workers <= 4
    args.output.mkdir(parents=True, exist_ok=True)
    args.records.mkdir(parents=True, exist_ok=True)
    selected = {row['path']: row for row in manifest['files']}
    assert len(selected) == 1100
    api = HfApi(token=False)
    catalog = {}
    for row in api.list_repo_tree(manifest['repo_id'], repo_type='dataset',
                                 revision=manifest['revision_sha'], recursive=True):
        if isinstance(row, RepoFile) and row.path in selected:
            assert Path(row.path).name == row.path and row.path.endswith('.tar.gz')
            assert row.size == selected[row.path]['compressed_bytes']
            assert row.lfs is not None
            digest = row.lfs.sha256
            assert len(digest) == 64
            catalog[row.path] = {**selected[row.path], 'sha256': digest}
    assert set(catalog) == set(selected)
    catalog_record = {'checked_utc': now(), 'repo_id': manifest['repo_id'],
                      'revision': manifest['revision_sha'], 'files': catalog}
    catalog_path = args.records / 'remote_catalog.json'
    if catalog_path.exists():
        old = json.loads(catalog_path.read_text())
        assert old['files'] == catalog and old['revision'] == catalog_record['revision']
    else:
        save(catalog_path, catalog_record)
    completed = []
    pending = []
    for name, row in sorted(catalog.items()):
        dest = args.output / name
        if dest.exists():
            # Existing source archives are never overwritten, including mismatches.
            assert dest.is_file() and dest.stat().st_size == row['compressed_bytes'], name
            actual = sha256(dest)
            assert actual == row['sha256'], '기존 파일 SHA256 불일치: ' + name
            completed.append({'path': name, 'bytes': dest.stat().st_size,
                              'sha256': actual, 'source': 'existing_verified'})
        else:
            pending.append(row)
    needed = sum(row['compressed_bytes'] for row in pending)
    free = shutil.disk_usage(args.output).free
    assert needed + 20 * 10**9 < free, '다운로드 및 보존 여유 공간 부족'
    state = {'started_utc': now(), 'repo_id': manifest['repo_id'],
             'revision': manifest['revision_sha'], 'total': len(catalog),
             'existing_verified': len(completed), 'missing': len(pending),
             'missing_bytes': needed, 'disk_free_before': free,
             'complete': False, 'completed': completed, 'failures': []}
    state_path = args.records / ('download_preflight.json' if args.prepare_only else 'download_status.json')
    save(state_path, state)
    print(json.dumps({k: v for k, v in state.items() if k not in ['completed', 'failures']}, ensure_ascii=False), flush=True)
    if args.prepare_only:
        return
    stopping = threading.Event()
    def stop(signum, frame):
        stopping.set()
        raise KeyboardInterrupt('다운로드 종료 신호: ' + str(signum))
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def fetch(row):
        failures = []
        for attempt in range(1, 3):
            if stopping.is_set():
                raise InterruptedError('다운로드 중단 요청')
            try:
                result = Path(hf_hub_download(
                    repo_id=manifest['repo_id'], repo_type='dataset', filename=row['path'],
                    revision=manifest['revision_sha'], local_dir=args.output, token=False))
                assert result.resolve() == (args.output / row['path']).resolve()
                assert result.stat().st_size == row['compressed_bytes']
                actual = sha256(result)
                assert actual == row['sha256'], '새 다운로드 SHA256 불일치'
                return {'path': row['path'], 'bytes': result.stat().st_size,
                        'sha256': actual, 'source': 'downloaded_verified',
                        'attempts': attempt, 'previous_error_types': failures, 'finished_utc': now()}
            except Exception as error:
                failures.append(type(error).__name__)
                if attempt == 2:
                    # Do not print signed download URLs, headers or credentials.
                    raise RuntimeError(row['path'] + ': ' + ','.join(failures)) from None
                stopping.wait(3)
        raise AssertionError('unreachable')
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
    futures = {}
    try:
        futures = {pool.submit(fetch, row): row['path'] for row in pending}
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                completed.append(result)
                print(json.dumps({'verified': len(completed), 'total': len(catalog),
                                  'file': result['path'], 'bytes': result['bytes']}, ensure_ascii=False), flush=True)
            except Exception as error:
                state['failures'].append({'path': futures[future], 'error': str(error), 'utc': now()})
                stopping.set()
                raise
            state['updated_utc'] = now()
            save(state_path, state)
        state['complete'] = len(completed) == len(catalog) and not state['failures']
    finally:
        stopping.set()
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        state['finished_utc'] = now()
        save(state_path, state)
    assert state['complete']
    print('전체 1100개 다운로드 및 SHA256 검사 완료', flush=True)


if __name__ == '__main__':
    main()
