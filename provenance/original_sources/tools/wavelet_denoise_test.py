# tools/wavelet_denoise_final.py
"""
Final fixed-parameter wavelet denoising for mainline evaluation.

Fixed best config (locked):
- wavelet: sym6
- level: 4
- top2 (threshold d1 + d2 only)
- thr_rule: universal
- thr_mode: soft
- k: 1.2
- ext_mode: symmetric
- output: wav with same folder layout
- IMPORTANT: no peak-normalize; only clip

Update:
- IN_ROOT now points to your *actual* input: data/anc_joint (full wavs)
- Only denoise PRIMARY wavs: data/anc_joint/snrX/primary/**.wav
- Output mirrors it: data/denoised_test/snrX/primary/**.wav

Run:
  python tools/wavelet_denoise_final.py
"""

from pathlib import Path
import json

import numpy as np
import soundfile as sf
import pywt
from tqdm import tqdm

EPS = 1e-12

# =============================
# Locked BEST config
# =============================
WAVELET = "sym6"
LEVEL = 4
TOP_BANDS = 2          # d1 + d2
K = 1.2
THR_RULE = "universal"
THR_MODE = "soft"
EXT_MODE = "symmetric"

# =============================
# I/O (updated for your project)
# =============================
SNR_LIST = ["snr-10", "snr-5","snr0","snr5", "snr10", "snr15", "snr20"]

# FULL wav root (NOT segments)
IN_ROOT = Path("data/anc_joint")          # snrX/primary/**.wav and snrX/reference/**.wav
OUT_ROOT = Path("data/wavelet_d1only_denoised_test")     # snrX/primary/**.wav (denoised)

PROCESS_ONLY_PRIMARY = True               # set False if you also want reference denoised
WRITE_META = True
WAV_SUBTYPE = "PCM_16"                    # keep consistent with your pipeline
CLIP_VAL = 0.98                           # headroom to avoid clipping

# =============================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def mad_sigma(x: np.ndarray) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + EPS
    return float(mad / 0.6745)


def universal_thr(sigma: float, n: int, k: float) -> float:
    return float(k * sigma * np.sqrt(2.0 * np.log(max(n, 2))))


def wavelet_denoise_best(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32)

    max_level = pywt.dwt_max_level(len(y), pywt.Wavelet(WAVELET).dec_len)
    level = max(1, min(LEVEL, max_level))

    coeffs = pywt.wavedec(y, wavelet=WAVELET, level=level, mode=EXT_MODE)
    new_coeffs = coeffs.copy()

    # threshold d1 and d2 only
    if level >= 1:
        cD1 = coeffs[-1]
        sigma1 = mad_sigma(cD1)
        thr1 = universal_thr(sigma1, len(cD1), K)
        new_coeffs[-1] = pywt.threshold(cD1, value=thr1, mode=THR_MODE)

    if level >= 2 and TOP_BANDS >= 2:
        cD2 = coeffs[-2]
        sigma2 = mad_sigma(cD2)
        thr2 = universal_thr(sigma2, len(cD2), K)
        new_coeffs[-2] = pywt.threshold(cD2, value=thr2, mode=THR_MODE)

    y_hat = pywt.waverec(new_coeffs, wavelet=WAVELET, mode=EXT_MODE)

    # length align
    if len(y_hat) > len(y):
        y_hat = y_hat[: len(y)]
    elif len(y_hat) < len(y):
        y_hat = np.pad(y_hat, (0, len(y) - len(y_hat)), mode="constant")

    # anti-clip
    y_hat = np.clip(y_hat.astype(np.float32), -CLIP_VAL, CLIP_VAL)
    return y_hat


def denoise_one_snr(snr_tag: str):
    in_dir = IN_ROOT / snr_tag
    if not in_dir.exists():
        print(f"[SKIP] missing input: {in_dir}")
        return

    # decide which subfolders to process
    subdirs = ["primary"] if PROCESS_ONLY_PRIMARY else ["primary", "reference"]

    for sub in subdirs:
        sub_in = in_dir / sub
        if not sub_in.exists():
            print(f"[SKIP] missing: {sub_in}")
            continue

        sub_out = OUT_ROOT / snr_tag / sub
        ensure_dir(sub_out)

        if WRITE_META:
            meta = {
                "wavelet": WAVELET,
                "level": LEVEL,
                "top_bands": TOP_BANDS,
                "k": K,
                "thr_rule": THR_RULE,
                "thr_mode": THR_MODE,
                "ext_mode": EXT_MODE,
                "normalize": "none",
                "clip": True,
                "clip_val": CLIP_VAL,
                "in_dir": str(sub_in),
                "out_dir": str(sub_out),
                "process_only_primary": PROCESS_ONLY_PRIMARY,
            }
            with open(sub_out / "params.json", "w") as f:
                json.dump(meta, f, indent=2)

        wavs = sorted([p for p in sub_in.rglob("*.wav") if p.is_file()])
        if not wavs:
            print(f"[WARN] no wavs under: {sub_in}")
            continue

        for p in tqdm(wavs, desc=f"denoise {snr_tag}/{sub}", leave=False):
            rel = p.relative_to(sub_in)          # training-a/a0012.wav ...
            out_p = sub_out / rel
            ensure_dir(out_p.parent)

            y, sr = sf.read(p, dtype="float32", always_2d=False)
            if y.ndim == 2:
                y = y.mean(axis=1).astype(np.float32)

            y_hat = wavelet_denoise_best(y)
            sf.write(out_p, y_hat, int(sr), subtype=WAV_SUBTYPE)

        print(f"[DONE] {snr_tag}/{sub} -> {sub_out}")


def main():
    print("[INFO] Fixed best config:")
    print(f"  wavelet={WAVELET}, lvl={LEVEL}, top={TOP_BANDS}, k={K}, rule={THR_RULE}, mode={THR_MODE}, clip={CLIP_VAL}")
    print(f"[INFO] IN_ROOT : {IN_ROOT}")
    print(f"[INFO] OUT_ROOT: {OUT_ROOT}")
    print(f"[INFO] SNR_LIST: {SNR_LIST}")
    print(f"[INFO] PROCESS_ONLY_PRIMARY: {PROCESS_ONLY_PRIMARY}")

    for snr in SNR_LIST:
        denoise_one_snr(snr)

    print("[DONE] all requested SNRs processed.")


if __name__ == "__main__":
    main()
