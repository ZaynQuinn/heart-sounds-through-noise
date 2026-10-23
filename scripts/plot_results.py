"""Plot fixed-variant metrics from evaluated CSV or JSON, without embedded scores."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--clean-result', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--title', default='Heart-sound classification by input SNR')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output directory already exists')
    if args.metrics.suffix == '.json':
        obj = json.loads(args.metrics.read_text())
        rows = [{'pipeline': r['branch'], 'snr_db': r['snr'], 'n': r['segments'],
                 'acc': r['accuracy'], 'macro_f1': r['macro_f1']} for r in obj['metrics']]
        scope = obj.get('scope', 'Scope not supplied')
    else:
        with args.metrics.open() as handle:
            rows = list(csv.DictReader(handle))
        scope = 'Evaluation of preserved features / ANC segments'
    groups = defaultdict(list)
    identities = set()
    for r in rows:
        key = (r['pipeline'], float(r['snr_db']))
        if key in identities:
            raise ValueError('Duplicate branch/SNR result')
        identities.add(key)
        if int(r['n']) <= 0 or not all(0 <= float(r[k]) <= 1 for k in ['acc', 'macro_f1']):
            raise ValueError('Invalid metric value or denominator')
        groups[r['pipeline']].append(r)
    if not groups:
        raise ValueError('Empty results')
    clean = json.loads(args.clean_result.read_text()) if args.clean_result else None
    if clean and any(int(r['n']) != clean['test_segments'] for r in rows):
        raise ValueError('Clean baseline denominator differs; omit baseline for subset figures')
    names = {'noisy': 'Noisy', 'anc': 'ANC', 'wavelet_d1': 'Wavelet D1',
             'wavelet_d1d2': 'Wavelet D1+D2', 'cascade_d1': 'Cascade D1', 'cascade_d1d2': 'Cascade D1+D2'}
    if set(groups) - set(names):
        raise ValueError('Unknown processing branch')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for name in names:
        if name not in groups:
            continue
        points = sorted(groups[name], key=lambda r: float(r['snr_db']))
        for ax, key, label in zip(axes, ['acc', 'macro_f1'], ['Accuracy (%)', 'Macro F1 (%)']):
            ax.plot([float(r['snr_db']) for r in points], [100*float(r[key]) for r in points],
                    marker='o', markersize=4, linewidth=1.7, label=names[name])
            ax.set_xlabel('Input SNR (dB)')
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
    if clean:
        for ax, key in zip(axes, ['acc', 'macro_f1']):
            ax.axhline(100*clean[key], color='black', linestyle='--', linewidth=1, label='Clean baseline')
    axes[0].legend(fontsize=8, loc='lower right')
    fig.suptitle(args.title, fontsize=13)
    denominators = sorted({int(r['n']) for r in rows})
    fig.text(0.5, 0.015, 'Segments per condition: ' + ', '.join(map(str, denominators)), ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .96))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fig.savefig(args.output_dir / 'classification.png', dpi=180)
    fig.savefig(args.output_dir / 'classification.svg')
    plt.close(fig)
    metadata = {'metrics_sha256': hashlib.sha256(args.metrics.read_bytes()).hexdigest(),
                'scope': scope, 'conditions': len(rows), 'denominators': denominators,
                'fixed_variants': list(groups), 'clean_baseline': clean is not None}
    (args.output_dir / 'figure_provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()
