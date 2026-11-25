"""전체 manifest 고정 및 새 archive의 선택 추출·검증·동일 좌표 변환."""
import argparse
from collections import Counter, defaultdict
import concurrent.futures
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import tarfile
import time

from PIL import Image

R = Path(__file__).resolve().parents[1]
CAMS = ['rgb_front', 'rgb_front_right', 'rgb_front_left', 'rgb_back', 'rgb_back_left', 'rgb_back_right']
LIMITS = {'rotation_orthogonality': 1e-5, 'rotation_determinant': 1e-5,
          'matrix_inverse_roundtrip': 1e-8, 'camera_location_error_m': .005,
          'center_vs_corners_m': .002, 'extent_vs_corners_m': .002,
          'projection_path_difference_px': 1e-4, 'augmentation_difference_px': 1e-6}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def audit_episode(archive, root, expected_sha, split):
    ep = archive.name[:-7]
    assert sha(archive) == expected_sha, '원격 archive SHA256 불일치'
    out = root / 'data/extracted' / ep
    counts = Counter(); dims = Counter(); frames = defaultdict(set)
    seen = set(); hashes = []; members = total = written = 0
    with gzip.open(archive, 'rb') as gz:
        with tarfile.open(fileobj=gz, mode='r|') as tar:
            for member in tar:
                path = PurePosixPath(member.name)
                assert not path.is_absolute() and '..' not in path.parts and path.parts[0] == ep
                assert member.isdir() or member.isfile(), '링크·특수 파일 거부'
                if not member.isfile():
                    continue
                assert member.name not in seen
                seen.add(member.name); members += 1; total += member.size
                sub = path.parts[1:]; counts['/'.join(sub[:-1])] += 1
                anno = len(sub) == 2 and sub[0] == 'anno'
                rgb = len(sub) == 3 and sub[0] == 'camera' and sub[1] in CAMS
                if not (anno or rgb):
                    continue
                body = tar.extractfile(member).read()
                frame = int(sub[-1].split('.')[0])
                if anno:
                    record = json.loads(gzip.decompress(body) if sub[-1].endswith('.gz') else body)
                    assert 'sensors' in record and 'bounding_boxes' in record
                    frames['anno'].add(frame)
                else:
                    with Image.open(io.BytesIO(body)) as image:
                        image.load()
                        assert image.size == (1600, 900) and image.mode == 'RGB'
                        dims[(sub[1], *image.size, image.mode)] += 1
                    frames[sub[1]].add(frame)
                dest = out.joinpath(*sub); dest.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256(body).hexdigest()
                if dest.exists():
                    assert sha(dest) == digest, '기존 추출 파일 충돌'
                else:
                    tmp = dest.with_name(dest.name + '.tmp'); tmp.write_bytes(body); tmp.replace(dest)
                hashes.append((str(dest.relative_to(root)), digest)); written += len(body)
        # tar end marker 이후 gzip trailer까지 소비해 CRC를 검사한다.
        while gz.read(8 << 20):
            pass
    ids = frames['anno']; assert ids
    assert min(ids) == 0 and sorted(ids) == list(range(max(ids) + 1))
    assert all(frames[cam] == ids for cam in CAMS)
    hash_file = root / 'audit' / (ep + '.sha256'); hash_file.parent.mkdir(exist_ok=True)
    hash_file.write_text(''.join(f'{h}  {name}\n' for name, h in sorted(hashes)))
    result = {'episode_id': ep, 'split': split, 'archive': str(archive.resolve()),
              'archive_sha256': expected_sha, 'archive_bytes': archive.stat().st_size,
              'members': members, 'uncompressed_bytes': total, 'extracted_bytes': written,
              'frames': len(ids), 'first_frame': min(ids), 'last_frame': max(ids),
              'sensor_counts': dict(counts), 'six_camera_frame_alignment': True,
              'gzip_crc': 'passed', 'images_full_decode': 'passed',
              'image_shapes': [{'camera': k[0], 'width': k[1], 'height': k[2], 'mode': k[3], 'count': n}
                               for k, n in sorted(dims.items())]}
    save(root / 'audit' / (ep + '.json'), {'episodes': [result], 'failures': []})
    return result

