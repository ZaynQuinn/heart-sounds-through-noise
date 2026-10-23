"""Plot matched recording/SNR/window comparisons from reconstructed waveforms."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--waveform-root', type=Path, required=True)
    p.add_argument('--recording', required=True)
    p.add_argument('--snr', type=int, required=True)
    p.add_argument('--variant', choices=['d1', 'd1d2'], required=True)
    p.add_argument('--start', type=float, default=0)
    p.add_argument('--duration', type=float, default=3)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists() or a.start < 0 or a.duration <= 0:
        p.error('Use a new output directory and a positive valid time window')
    root = a.waveform_root.resolve()
    out = a.output_dir.resolve()
    if out.is_relative_to(root) or root.is_relative_to(out):
        p.error('Input and output trees must be separate')
    tag = f'snr{a.snr}'
    paths = {'Noisy': root/'mix'/tag/'primary'/a.recording,
             'ANC': root/'anc'/tag/'anc'/a.recording,
             'Wavelet '+a.variant.upper(): root/('wavelet_'+a.variant)/tag/a.recording,
             'Cascade '+a.variant.upper(): root/('cascade_'+a.variant)/tag/a.recording}
    entries = []
    for name, path in paths.items():
        if not path.resolve().is_relative_to(root):
            raise ValueError('Recording path escapes root')
        audio, sr = sf.read(path, dtype='float32')
        if sr != 2000 or audio.ndim != 1:
            raise ValueError('Expected 2kHz mono audio')
        start, length = round(a.start*sr), round(a.duration*sr)
        if length < 4 or start+length > len(audio):
            raise ValueError('Requested window exceeds recording or is too short')
        entries.append((name, audio[start:start+length], sr, hashlib.sha256(path.read_bytes()).hexdigest()))
    fig, axes = plt.subplots(4, 2, figsize=(11, 8), sharex='col')
    peak = max(float(np.max(np.abs(y))) for _, y, _, _ in entries)
    for row, (name, y, sr, _) in enumerate(entries):
        t = a.start + np.arange(len(y))/sr
        axes[row, 0].plot(t, y, linewidth=.65)
        axes[row, 0].set_ylim(-max(peak, 1e-6)*1.05, max(peak, 1e-6)*1.05)
        axes[row, 0].set_ylabel(name+'\nAmplitude')
        window = np.hanning(len(y))
        amplitude = np.abs(np.fft.rfft(y*window))*2/window.sum()
        amplitude[0] *= .5
        if len(y) % 2 == 0:
            amplitude[-1] *= .5
        spectrum = 20*np.log10(np.maximum(amplitude, 1e-10))
        axes[row, 1].plot(np.fft.rfftfreq(len(y), 1/sr), spectrum, linewidth=.8)
        axes[row, 1].set_xlim(0, 1000)
        axes[row, 1].set_ylim(-100, 0)
        axes[row, 1].set_ylabel('dB re unit amplitude')
        for ax in axes[row]:
            ax.grid(alpha=.2)
    axes[-1, 0].set_xlabel('Time in recording (s)')
    axes[-1, 1].set_xlabel('Frequency (Hz)')
    fig.suptitle(f'{a.recording} | input SNR {a.snr} dB | {a.variant.upper()} | matched window', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96))
    out.mkdir(parents=True, exist_ok=False)
    fig.savefig(out/'waveform_comparison.png', dpi=160)
    fig.savefig(out/'waveform_comparison.svg')
    plt.close(fig)
    report = {'recording':a.recording, 'snr':a.snr, 'variant':a.variant,
              'start_seconds':a.start, 'duration_seconds':a.duration,
              'sample_rate':2000, 'samples_per_branch':len(entries[0][1]),
              'spectrum':'Hann-windowed one-sided amplitude FFT, 20log10, common absolute reference; not PSD',
              'input_sha256':{name:digest for name,_,_,digest in entries},
              'scope':'New same-condition illustration, not exact reconstruction of paper Figures 7/8; no S1/S2 annotations inferred'}
    (out/'figure_provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
