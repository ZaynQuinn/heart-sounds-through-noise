"""Download manifest-listed PhysioNet clean recordings and verify their hashes.

Read the data preparation instructions in README.md and the publisher's terms before downloading. No audio is uploaded.
"""
import argparse
import csv
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://physionet.org/files/challenge-2016/1.0.0/'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True, help='New audio data root')
    parser.add_argument('--subsets', nargs='+', choices=list('abcdef'), default=list('abcdef'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true',
                        help='Verify and reuse existing files; download only missing files')
    args = parser.parse_args()
    if args.output_root.exists() and not args.resume:
        parser.error('Output root must not exist; existing data are never overwritten')
    if args.output_root.exists() and not args.output_root.is_dir():
        parser.error('Output root must be a directory')
    if not 1 <= args.workers <= 4:
        parser.error('Use between 1 and 4 workers')
    output = args.output_root.resolve()
    with (ROOT / 'manifests/input_checksums.csv').open() as handle:
        rows = [r for r in csv.DictReader(handle) if r['kind'] == 'clean'
                and Path(r['path']).parent.name[-1] in args.subsets]
    for row in rows:
        rel = Path(row['path'])
        if not (output / rel).resolve().is_relative_to(output):
            raise ValueError('Manifest path escapes output')
        rel.relative_to('data/raw')
    output.mkdir(parents=True, exist_ok=args.resume)

    def download(row):
        relative = Path(row['path']).relative_to('data/raw').as_posix()
        url = BASE + relative
        try:
            target = output / row['path']
            if target.exists():
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
                if target.stat().st_size == int(row['bytes']) and digest == row['sha256']:
                    return {'path': row['path'], 'status': 'verified', 'source': 'existing'}
                return {'path': row['path'], 'status': 'existing_mismatch',
                        'received_sha256': digest}
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read(int(row['bytes']) + 1)
            digest = hashlib.sha256(payload).hexdigest()
            if len(payload) != int(row['bytes']) or digest != row['sha256']:
                return {'path': row['path'], 'status': 'mismatch', 'received_sha256': digest}
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as handle:
                handle.write(payload)
            return {'path': row['path'], 'status': 'verified', 'source': 'downloaded'}
        except Exception as error:
            return {'path': row['path'], 'status': 'error', 'error': str(error)}

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(download, rows):
            results.append(result)
            if len(results) % 50 == 0:
                print(f'Checked {len(results)}/{len(rows)} recordings', flush=True)
    failed = [r for r in results if r['status'] != 'verified']
    report = {'source': BASE, 'requested': len(rows), 'verified': len(rows)-len(failed),
              'reused': sum(r.get('source') == 'existing' for r in results),
              'downloaded': sum(r.get('source') == 'downloaded' for r in results),
              'failures': failed,
              'scope': 'Downloaded WAV bytes checked against the audited archive SHA-256 manifest'}
    (output / 'download_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
