"""Check local reproduction inputs against the audited archive fingerprints."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--kind', choices=['clean', 'noise', 'noise_2k', 'all'], default='all')
    args = parser.parse_args()
    root = args.data_root.resolve()
    with (ROOT / 'manifests/input_checksums.csv').open() as handle:
        rows = [r for r in csv.DictReader(handle) if args.kind == 'all' or r['kind'] == args.kind]
    failures = []
    for row in rows:
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            failures.append({'path': row['path'], 'reason': 'missing or outside root'})
        elif path.stat().st_size != int(row['bytes']) or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            failures.append({'path': row['path'], 'reason': 'content differs'})
    print(json.dumps({'checked': len(rows), 'matched': len(rows) - len(failures), 'failures': failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
