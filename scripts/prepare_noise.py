"""Resample only noise recordings referenced by the preserved mixing recipes."""
import argparse
import csv
import json
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--noise-root', type=Path, required=True, help='Extracted MUSAN noise directory')
    parser.add_argument('--output-root', type=Path, required=True, help='New directory for 2kHz noise')
    parser.add_argument('--compare-root', type=Path, help='Optional historical musan_noise_2k directory')
    args = parser.parse_args()
    source, output = args.noise_root.resolve(), args.output_root.resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        parser.error('Output must be a new directory separate from input')
    with (ROOT / 'manifests/noise_mixing.csv').open() as handle:
        names = sorted({r['noise_rel'] for r in csv.DictReader(handle)})
    for name in names:
        path = (source / name).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError('Missing input or unsafe noise path: ' + name)
    output.mkdir(parents=True, exist_ok=False)
    checks = []
    for name in names:
        audio, _ = librosa.load(source / name, sr=2000, mono=True)
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(target, audio.astype(np.float32), 2000, subtype='PCM_16')
        if args.compare_root:
            actual, sr = sf.read(target, dtype='float32')
            stored, sr2 = sf.read(args.compare_root / name, dtype='float32')
            if actual.shape != stored.shape or sr != sr2:
                raise ValueError('Resampled audio shape/rate differs: ' + name)
            checks.append({'path': name, 'exact': bool(np.array_equal(actual, stored)),
                           'max_abs_error': float(np.max(np.abs(actual - stored)))})
    report = {'noise_recordings': len(names), 'sample_rate': 2000, 'subtype': 'PCM_16',
              'librosa': librosa.__version__, 'soundfile': sf.__version__, 'checks': checks}
    (output / 'resampling_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'noise_recordings': len(names), 'compared': len(checks),
                      'exact': sum(r['exact'] for r in checks),
                      'max_abs_error': max((r['max_abs_error'] for r in checks), default=None)}))


if __name__ == '__main__':
    main()
