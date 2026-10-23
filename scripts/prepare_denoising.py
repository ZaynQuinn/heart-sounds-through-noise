"""Replay recorded synthetic mixtures and denoising into a new output tree."""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'provenance/original_sources' / relative)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def inside(root, relative):
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError('Path escapes root: ' + relative)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True,
                        help='Root containing data/raw and musan_noise_2k')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--recordings', nargs='+', help='Optional clean_rel values, e.g. training-a/a0012.wav')
    parser.add_argument('--snrs', nargs='+', type=int, default=[-10, -5, 0, 5, 10, 15, 20])
    args = parser.parse_args()
    data, out = args.data_root.resolve(), args.output_root.resolve()
    if out.is_relative_to(data) or data.is_relative_to(out):
        parser.error('Input and output trees must be separate')
    if out.exists():
        parser.error('Output root already exists')
    config = json.loads((ROOT / 'configs/experiment.json').read_text())
    if not set(args.snrs) <= set(config['snr_db']):
        parser.error('SNR is not present in the experiment configuration')
    with (ROOT / 'manifests/noise_mixing.csv').open() as handle:
        recipes = [r for r in csv.DictReader(handle)
                   if int(float(r['snr_primary_db'])) in args.snrs
                   and (args.recordings is None or r['clean_rel'] in args.recordings)]
    if not recipes:
        parser.error('No matching recipes')
    if args.recordings and set(args.recordings) != {r['clean_rel'] for r in recipes}:
        parser.error('Some requested recordings have no recipe')
    for r in recipes:
        for path in [inside(data / 'data/raw', r['clean_rel']), inside(data / 'musan_noise_2k', r['noise_rel'])]:
            if not path.is_file():
                raise FileNotFoundError(path)
        for key in ('primary_rel', 'reference_rel'):
            inside(out / 'mix', r[key])
    mix = module('repro_mix', 'tools/make_noisy_test_only.py')
    anc = module('repro_anc', 'tools/anc/run_anc.py')
    wavelet = module('repro_wavelet', 'tools/wavelet_denoise_test.py')
    out.mkdir(parents=True, exist_ok=False)
    for r in recipes:
        clean = mix.load_audio(data / 'data/raw' / r['clean_rel'], 2000)
        noise = mix.load_audio(data / 'musan_noise_2k' / r['noise_rel'], 2000)
        if len(clean) != int(r['clean_len']) or len(noise) != int(r['noise_len']):
            raise ValueError('Audio length differs from preserved mixing recipe')
        if r['noise_seg_mode'] == 'slice':
            start = int(r['noise_start'])
            segment = noise[start:start + len(clean)]
        elif r['noise_seg_mode'] == 'tile':
            segment = np.tile(noise, int(r['noise_tile_reps']))[:len(clean)]
        else:
            raise ValueError('Unknown noise segment mode')
        if len(segment) != len(clean):
            raise ValueError('Recipe does not supply enough noise samples')
        for snr_key, path_key in [('snr_primary_db', 'primary_rel'), ('snr_reference_db', 'reference_rel')]:
            coefficient = mix.scale_noise_to_snr(clean, segment, float(r[snr_key]))
            audio, _, _, _ = mix.anti_clip((clean + coefficient * segment).astype(np.float32), float(r['headroom']))
            target = out / 'mix' / r[path_key]
            target.parent.mkdir(parents=True, exist_ok=True)
            sf.write(target, audio, 2000, subtype='PCM_16')
    anc.IN_ROOT, anc.OUT_ROOT = out / 'mix', out / 'anc'
    anc.SNR_TAGS = [f'snr{s}' for s in sorted(set(args.snrs))]
    for name, key in [('PROTECT_HZ','protect_hz'), ('SUBBANDS','subbands'), ('FILTER_LEN','filter_len'),
                      ('BASE_MU','base_mu'), ('LEAK','leak'), ('DELTA','delta'),
                      ('USE_DELAY_ALIGN','delay_align'), ('MAX_LAG_SAMPLES','max_lag_samples'), ('HIGH_MODE','high_mode')]:
        setattr(anc, name, config['anc'][key])
    anc.main()
    for name, key in [('WAVELET','wavelet'), ('LEVEL','level'), ('K','k'), ('THR_RULE','thr_rule'),
                      ('THR_MODE','thr_mode'), ('EXT_MODE','ext_mode'), ('CLIP_VAL','clip_val')]:
        setattr(wavelet, name, config['wavelet'][key])
    for r in recipes:
        tag = f"snr{int(float(r['snr_primary_db']))}"
        for stage, source in [('wavelet', out / 'mix' / r['primary_rel']),
                              ('cascade', out / 'anc' / tag / 'anc' / r['clean_rel'])]:
            audio, sr = sf.read(source, dtype='float32')
            if sr != 2000:
                raise ValueError('Unexpected sample rate')
            for variant, settings in config['variants'].items():
                wavelet.TOP_BANDS = settings['top_bands']
                denoised = wavelet.wavelet_denoise_best(audio)
                target = out / f'{stage}_{variant}' / tag / r['clean_rel']
                target.parent.mkdir(parents=True, exist_ok=True)
                sf.write(target, denoised, sr, subtype='PCM_16')
    with (out / 'selected_recipes.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(recipes[0]))
        writer.writeheader()
        writer.writerows(recipes)
    report = {'recordings': len({r['clean_rel'] for r in recipes}), 'mixtures': len(recipes),
              'snrs': sorted(set(args.snrs)), 'waveform_files': len(list(out.rglob('*.wav'))),
              'config': config, 'scope': 'Preserved 2kHz noise and recipes to PCM16 waveforms; no original MUSAN resampling'}
    (out / 'denoising_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'config'}, indent=2))


if __name__ == '__main__':
    main()
