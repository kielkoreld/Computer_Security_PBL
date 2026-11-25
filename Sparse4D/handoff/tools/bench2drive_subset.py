#!/usr/bin/env python3
"""Select reproducible Bench2Drive V0.0.4 archives, equally per scenario.

Examples:
    python3 bench2drive_subset.py --per-scenario 1
    python3 bench2drive_subset.py --per-scenario 1 --download
    python3 bench2drive_subset.py --per-scenario 5 --download
"""

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, RepoFile, snapshot_download

REPO_ID = "rethinklab/Bench2Drive-V0.0.4"
ARCHIVE_RE = re.compile(
    r"^(?P<scenario>.+?)_Town\d+(?:HD)?_Route\d+_Weather\d+_.+\.tar\.gz$"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select N reproducible .tar.gz archives per Bench2Drive scenario."
    )
    parser.add_argument("--per-scenario", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("~/datasets/Bench2Drive-V0.0.4"))
    parser.add_argument("--revision", default="main", help="Branch, tag, or pinned full commit SHA")
    parser.add_argument("--download", action="store_true", help="Download after manifest/size preview")
    args = parser.parse_args()

    if not 1 <= args.per_scenario <= 25:
        parser.error("--per-scenario must be between 1 and 25")

    output = args.output.expanduser().resolve()
    api = HfApi()
    revision = api.repo_info(REPO_ID, repo_type="dataset", revision=args.revision).sha
    if not revision:
        raise RuntimeError("Could not resolve the repository revision.")

    grouped = defaultdict(list)
    unmatched = []
    for entry in api.list_repo_tree(
        REPO_ID, repo_type="dataset", recursive=True, revision=revision
    ):
        if not isinstance(entry, RepoFile) or not entry.path.endswith(".tar.gz"):
            continue
        match = ARCHIVE_RE.fullmatch(Path(entry.path).name)
        if match is None:
            unmatched.append(entry.path)
            continue
        grouped[match.group("scenario")].append(entry)

    if unmatched:
        raise RuntimeError(f"Unrecognized archive names: {unmatched[:5]} (total {len(unmatched)})")
    if len(grouped) != 44 or any(len(v) != 25 for v in grouped.values()):
        counts = {k: len(v) for k, v in sorted(grouped.items())}
        raise RuntimeError(f"Unexpected dataset structure; expected 44 x 25: {counts}")

    selected = []
    for scenario in sorted(grouped):
        # Stable random ranking: increasing N retains all already selected routes.
        candidates = sorted(grouped[scenario], key=lambda f: f.path)
        random.Random(f"{args.seed}:{scenario}").shuffle(candidates)
        selected.extend((scenario, file) for file in candidates[: args.per_scenario])

    selected.sort(key=lambda item: (item[0], item[1].path))
    total_bytes = sum(file.size for _, file in selected)
    output.mkdir(parents=True, exist_ok=True)

    manifest = output / f"selected_{args.per_scenario}_per_scenario.txt"
    manifest.write_text("".join(file.path + "\n" for _, file in selected), encoding="utf-8")

    details = {
        "repo_id": REPO_ID,
        "revision_sha": revision,
        "seed": args.seed,
        "per_scenario": args.per_scenario,
        "scenarios": len(grouped),
        "archive_count": len(selected),
        "compressed_bytes": total_bytes,
        "files": [
            {"scenario": scenario, "path": file.path, "compressed_bytes": file.size}
            for scenario, file in selected
        ],
    }
    manifest_json = output / f"selected_{args.per_scenario}_per_scenario.json"
    manifest_json.write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    missing = [
        file
        for _, file in selected
        if not (output / file.path).is_file()
        or (output / file.path).stat().st_size != file.size
    ]
    need_bytes = sum(file.size for file in missing)
    free_bytes = shutil.disk_usage(output).free

    print(f"Dataset revision: {revision}")
    print(f"Scenarios: {len(grouped)} | Archives: {len(selected)} | Seed: {args.seed}")
    print(f"Selected compressed size: {total_bytes / 1e9:.2f} GB ({total_bytes / 2**30:.2f} GiB)")
    print(f"Absent or size-mismatched files: {len(missing)} ({need_bytes / 1e9:.2f} GB)")
    print(f"Free disk space: {free_bytes / 1e9:.2f} GB")
    print(f"Manifest: {manifest}")
    print(f"Manifest details: {manifest_json}")
    print("Selection by scenario:")
    for scenario, file in selected:
        print(f"  {scenario:35s} {file.size / 1e6:8.1f} MB  {file.path}")

    if not args.download:
        print("PREVIEW ONLY: add --download to start downloading these exact archives.")
        return
    if need_bytes > free_bytes:
        raise RuntimeError("Not enough free disk space for the not-yet-downloaded archives.")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[file.path for _, file in selected],
        local_dir=str(output),
        max_workers=4,
    )
    print(f"Download finished. Archives: {output}")


if __name__ == "__main__":
    main()
