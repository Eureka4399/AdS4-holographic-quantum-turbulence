"""Inventory or verify MAT files without loading arrays or running MATLAB.

Usage:
    python tools/data_inventory.py --refresh
    python tools/data_inventory.py --verify
    python tools/data_inventory.py --verify --included-only

The standard library is sufficient. Hashing reads each file sequentially.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'data/metadata/manifest.json'
INCLUDED = {
    'data/reference/mu=5,nz=30,vy=3.8_20.mat',
    'data/reference/mu=6,nz=30,vy=7_20.mat',
    'data/reference/mu=8,nz=30,vy=13.5_20.mat',
    'data/processed/average2.mat',
    'data/processed/structure_factor.mat',
}


def sha256(path: Path) -> str:
    """Hash one file in bounded-memory chunks without deserializing its contents."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def refresh() -> None:
    """Record the local MAT collection and explicitly rewrite its two manifests."""
    records = []
    # Read only the MAT header for the format; array data are used solely for hashing.
    for path in sorted((ROOT / 'data').rglob('*.mat')):
        relative = path.relative_to(ROOT).as_posix()
        with path.open('rb') as stream:
            header = stream.read(116).decode('ascii', errors='replace').strip()
        record = {
            'path': relative,
            'original_name': path.name,
            'category': path.parent.name,
            'size_bytes': path.stat().st_size,
            'sha256': sha256(path),
            'mat_format': '7.3 (HDF5)' if header.startswith('MATLAB 7.3') else '5.0',
            'mat_header': header,
            'distribution': 'git' if relative in INCLUDED else 'separate_data_archive',
        }
        records.append(record)
        print(f"Inventoried {relative} ({record['size_bytes']:,} bytes)", flush=True)
    payload = {
        'schema_version': 1,
        'hash_algorithm': 'SHA-256',
        'total_files': len(records),
        'total_size_bytes': sum(item['size_bytes'] for item in records),
        'files': records,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    columns = ['path', 'category', 'size_bytes', 'sha256', 'mat_format', 'distribution']
    with MANIFEST.with_suffix('.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    print(f"Manifest: {len(records)} files, {payload['total_size_bytes']:,} bytes.")


def verify(included_only: bool) -> int:
    """Compare local file size and content hashes against the existing inventory."""
    payload = json.loads(MANIFEST.read_text(encoding='utf-8'))
    failed = False
    for item in payload['files']:
        if included_only and item['distribution'] != 'git':
            continue
        path = (ROOT / item['path']).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Manifest path escapes the repository.')
        if not path.is_file():
            print(f"Missing: {item['path']}")
            failed = True
        elif path.stat().st_size != item['size_bytes'] or sha256(path) != item['sha256']:
            print(f"Mismatch: {item['path']}")
            failed = True
        else:
            print(f"OK: {item['path']}", flush=True)
    return int(failed)


def main() -> int:
    """Select either read-only verification or an explicit manifest refresh."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--refresh', action='store_true', help='Write a new inventory of local files.')
    mode.add_argument('--verify', action='store_true', help='Compare local files with the inventory.')
    parser.add_argument('--included-only', action='store_true', help='Verify only the five files included in Git.')
    arguments = parser.parse_args()
    if arguments.included_only and not arguments.verify:
        parser.error('--included-only requires --verify')
    if arguments.refresh:
        refresh()
        return 0
    return verify(arguments.included_only)


if __name__ == '__main__':
    sys.exit(main())
