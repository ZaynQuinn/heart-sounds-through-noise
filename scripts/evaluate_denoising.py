"""Evaluate all generated denoising branches with the preserved test segmentation."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from src.models.tiny_cnn import MicroHeartCNN

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--waveform-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.waveform_root.resolve(), args.output_dir.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        parser.error('Input and output trees must be separate')
    if output.exists():
        parser.error('Output directory already exists')
    config = json.loads((ROOT / 'configs/experiment.json').read_text())
    with (source / 'selected_recipes.csv').open() as handle:
        recipes = list(csv.DictReader(handle))
    if not recipes:
        parser.error('Selected recipe manifest is empty')
    with (ROOT / 'manifests/segments.csv').open() as handle:
        segments = defaultdict(list)
        for row in csv.DictReader(handle):
            if row['split'] == 'test':
                segments[str(Path(row['source']).relative_to('data/raw'))].append(row)
    aliases = json.loads((ROOT / 'checkpoints/aliases.json').read_text())
    checkpoint = ROOT / aliases['canonical_checkpoint']
    model = MicroHeartCNN()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    torch.set_num_threads(2)
    mean = np.load(ROOT / 'normalization/mean.npy')[:, None]
    std = np.load(ROOT / 'normalization/std.npy')[:, None]
    size = int(config['sample_rate'] * config['segment_seconds'])
    hop = int(config['sample_rate'] * config['hop_seconds'])
    f = config['features']
    matrices = defaultdict(lambda: np.zeros((2, 2), dtype=int))
    predictions = []
    seen = set()
    for recipe in recipes:
        rel = recipe['clean_rel']
        snr = int(float(recipe['snr_primary_db']))
        if (rel, snr) in seen or rel not in segments:
            raise ValueError('Duplicate recipe or recording absent from test split')
        seen.add((rel, snr))
        tag = f'snr{snr}'
        paths = {'noisy': source / 'mix' / recipe['primary_rel'],
                 'anc': source / 'anc' / tag / 'anc' / rel}
        paths.update({name: source / name / tag / rel for name in
                      ['wavelet_d1', 'wavelet_d1d2', 'cascade_d1', 'cascade_d1d2']})
        for branch, path in paths.items():
            if not path.resolve().is_relative_to(source):
                raise ValueError('Waveform path escapes input root')
            audio, sr = sf.read(path, dtype='float32')
            if sr != config['sample_rate'] or audio.ndim != 1 or not np.isfinite(audio).all():
                raise ValueError('Expected finite mono audio at configured sample rate')
            records = segments[rel]
            features = []
            for row in records:
                start = int(row['segment']) * hop
                clip = audio[start:start + size]
                if len(clip) < size:
                    if start != 0 or len(audio) >= size:
                        raise ValueError('Waveform is shorter than preserved segment range')
                    clip = np.pad(clip, (0, size - len(clip)))
                mel = librosa.feature.melspectrogram(y=clip, sr=sr, n_fft=f['n_fft'],
                    hop_length=f['hop_length'], n_mels=f['n_mels'], fmin=f['fmin'], fmax=f['fmax'],
                win_length=f['win_length'], window=f['window'], center=f['center'],
                pad_mode=f['pad_mode'], power=f['power'], htk=f['htk'],
                norm=f['norm'], dtype=np.dtype(f['dtype']))
                feature = np.log(mel + f['log_epsilon']).astype(np.float32)
                features.append((feature - mean) / std)
            with torch.inference_mode():
                predicted = model(torch.from_numpy(np.stack(features)[:, None])).argmax(1).tolist()
            for row, pred in zip(records, predicted):
                truth = int(row['label'] == '1')
                matrices[(branch, snr)][truth, pred] += 1
                predictions.append({'branch': branch, 'snr': snr, 'recording': rel,
                                    'segment': int(row['segment']), 'label_id': truth, 'prediction': pred})
    metrics = []
    for (branch, snr), m in sorted(matrices.items()):
        f1 = [2 * m[k, k] / (m[k, :].sum() + m[:, k].sum())
              if m[k, :].sum() + m[:, k].sum() else 0.0 for k in range(2)]
        metrics.append({'branch': branch, 'snr': snr, 'segments': int(m.sum()),
                        'accuracy': float(m.trace() / m.sum()), 'macro_f1': float(np.mean(f1)),
                        'confusion_matrix': m.tolist()})
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'predictions.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    report = {'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'recordings': len({r['clean_rel'] for r in recipes}), 'recipe_count': len(recipes),
              'scope': 'All preserved test segments for selected recipes; generated PCM16 waveforms to log-Mel to classification',
              'class_order': [-1, 1], 'confusion_matrix_axes': ['true', 'predicted'],
              'metrics': metrics}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'conditions': len(metrics), 'predictions': len(predictions), 'recordings': report['recordings']}))


if __name__ == '__main__':
    main()
