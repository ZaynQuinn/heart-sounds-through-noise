"""Recreate clean log-Mel features using the preserved recording/segment split."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True,
                        help='Root containing data/raw/training-*/*.wav')
    parser.add_argument('--output-root', type=Path, required=True,
                        help='New directory for data/features; must not already exist')
    parser.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='test')
    parser.add_argument('--compare-existing', action='store_true',
                        help='Compare every regenerated feature with the data-root feature')
    args = parser.parse_args()
    data = args.data_root.resolve()
    output = args.output_root.resolve()
    if output.is_relative_to(data) or data.is_relative_to(output):
        parser.error('Input and output directory trees must be separate')
    if output.exists():
        parser.error('Output directory already exists; choose a new directory')
    config = json.loads((ROOT / 'configs/experiment.json').read_text())
    with (ROOT / 'manifests/segments.csv').open() as handle:
        rows = [r for r in csv.DictReader(handle)
                if args.split == 'all' or r['split'] == args.split]
    groups = defaultdict(list)
    for row in rows:
        source = (data / row['source']).resolve()
        target = (output / row['path']).resolve()
        if not source.is_relative_to(data) or not target.is_relative_to(output):
            raise ValueError('Manifest path escapes root')
        if not source.is_file():
            raise FileNotFoundError(source)
        groups[source].append(row)
    output.mkdir(parents=True, exist_ok=False)
    sr = config['sample_rate']
    size = int(sr * config['segment_seconds'])
    hop = int(sr * config['hop_seconds'])
    f = config['features']
    exact = 0
    max_error = 0.0
    for source, records in groups.items():
        audio, _ = librosa.load(source, sr=sr, mono=True)
        audio = audio.astype(np.float32)
        for row in records:
            start = int(row['segment']) * hop
            segment = audio[start:start + size]
            if len(segment) < size:
                if start != 0 or len(audio) >= size:
                    raise ValueError('Manifest requests a segment beyond the historical window rule')
                segment = np.pad(segment, (0, size - len(segment)))
            mel = librosa.feature.melspectrogram(
                y=segment, sr=sr, n_fft=f['n_fft'], hop_length=f['hop_length'],
                n_mels=f['n_mels'], fmin=f['fmin'], fmax=f['fmax'],
                win_length=f['win_length'], window=f['window'], center=f['center'],
                pad_mode=f['pad_mode'], power=f['power'], htk=f['htk'],
                norm=f['norm'], dtype=np.dtype(f['dtype']))
            feature = np.log(mel + f['log_epsilon']).astype(np.float32)
            target = output / row['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, feature)
            if args.compare_existing:
                existing = np.load(data / row['path'])
                if existing.shape != feature.shape:
                    raise ValueError('Feature shapes differ')
                exact += int(np.array_equal(feature, existing))
                max_error = max(max_error, float(np.max(np.abs(feature - existing))))
    report = {'split': args.split, 'recordings': len(groups), 'segments': len(rows),
              'librosa': librosa.__version__, 'numpy': np.__version__,
              'scope': 'Raw clean WAV to features; preserved manifest and normalization; no split regeneration',
              'comparison': {'exact_features': exact, 'max_abs_error': max_error}
              if args.compare_existing else None}
    (output / 'feature_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
