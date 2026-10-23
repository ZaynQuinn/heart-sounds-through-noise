"""Evaluate the canonical checkpoint using existing clean test features."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from src.models.tiny_cnn import MicroHeartCNN

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path,
                        help='Directory containing data/features referenced by the manifest')
    parser.add_argument('--output', required=True, type=Path,
                        help='New JSON output file (existing files are not overwritten)')
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    if output.is_relative_to(data_root):
        parser.error('Output must be outside the input data directory')
    if output.exists():
        parser.error('Output already exists; choose a new path')
    aliases = json.loads((ROOT / 'checkpoints/aliases.json').read_text())
    checkpoint = ROOT / aliases['canonical_checkpoint']
    torch.set_num_threads(2)
    model = MicroHeartCNN()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    with (ROOT / 'manifests/segments.csv').open() as handle:
        rows = [row for row in csv.DictReader(handle) if row['split'] == 'test']
    if not rows:
        raise ValueError('No test segments in manifest')
    if any(row['label'] not in ('-1', '1') for row in rows):
        raise ValueError('Expected labels -1 and 1')
    paths = [(data_root / row['path']).resolve() for row in rows]
    if any(not path.is_relative_to(data_root) for path in paths):
        raise ValueError('Feature paths must remain inside data root')
    mean = np.load(ROOT / 'normalization/mean.npy')[:, None]
    std = np.load(ROOT / 'normalization/std.npy')[:, None]
    matrix = np.zeros((2, 2), dtype=int)
    with torch.inference_mode():
        for start in range(0, len(rows), 128):
            features = np.stack([(np.load(path).astype(np.float32) - mean) / std
                                 for path in paths[start:start + 128]])[:, None]
            predictions = model(torch.from_numpy(features)).argmax(1).tolist()
            for row, prediction in zip(rows[start:start + 128], predictions):
                matrix[int(row['label'] == '1'), prediction] += 1
    f1 = []
    for k in range(2):
        denominator = int(matrix[k, :].sum() + matrix[:, k].sum())
        f1.append(2 * int(matrix[k, k]) / denominator if denominator else 0.0)
    result = {
        'run': aliases['canonical_id'],
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'test_segments': len(rows), 'acc': float(matrix.trace() / matrix.sum()),
        'macro_f1': float(np.mean(f1)), 'confusion_matrix': matrix.tolist(),
        'class_order': [-1, 1], 'matrix_axes': ['true', 'predicted'],
        'params': sum(p.numel() for p in model.parameters()),
        'torch': torch.__version__, 'numpy': np.__version__,
        'scope': 'CPU inference on existing clean features; no training or feature regeneration',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
