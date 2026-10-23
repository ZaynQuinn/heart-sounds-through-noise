"""Recompute historical float32 per-mel statistics from training rows only.

Requires about 2 GiB of free RAM for the preserved training set.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    data, output = args.data_root.resolve(), args.output_dir.resolve()
    if output.is_relative_to(data) or data.is_relative_to(output):
        parser.error('Input and output trees must be separate')
    if output.exists():
        parser.error('Output directory already exists')
    manifest = ROOT / 'manifests/segments.csv'
    with manifest.open() as handle:
        rows = [r for r in csv.DictReader(handle) if r['split'] == 'train']
    if not rows:
        raise ValueError('No training rows')
    paths = [(data / r['path']).resolve() for r in rows]
    if any(not p.is_relative_to(data) for p in paths):
        raise ValueError('Feature path escapes data root')
    # Preserve manifest order, float32 reduction and axis tuple from preprocess.py.
    first = np.load(paths[0])
    features = np.empty((len(rows), *first.shape), dtype=first.dtype)
    for index, path in enumerate(paths):
        value = np.load(path)
        if value.shape != first.shape or value.dtype != first.dtype or not np.isfinite(value).all():
            raise ValueError('Invalid or inconsistent feature: ' + str(path))
        features[index] = value
    mean = features.mean(axis=(0, 2))
    std = features.std(axis=(0, 2)) + 1e-6
    output.mkdir(parents=True, exist_ok=False)
    comparison = {}
    for name, value in [('mean', mean), ('std', std)]:
        np.save(output / (name + '.npy'), value)
        historical = np.load(ROOT / 'normalization' / (name + '.npy'))
        comparison[name] = {'array_equal': bool(np.array_equal(value, historical)),
                            'max_abs_error': float(np.max(np.abs(value - historical)))}
    report = {'training_segments': len(rows), 'training_recordings': len({r['source'] for r in rows}),
              'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
              'numpy': np.__version__, 'dtype': str(features.dtype),
              'method': 'mean(axis=(0,2)); std(axis=(0,2)) + 1e-6; manifest training order',
              'scope': 'Recomputed from existing training features, not regenerated training WAV features',
              'comparison': comparison}
    (output / 'normalization_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
