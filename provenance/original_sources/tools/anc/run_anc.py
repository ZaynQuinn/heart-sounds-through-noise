# tools/anc/run_task_aware_anc.py
"""
Task-aware dual-microphone ANC (subband NLMS with protected heart-sound band)

Goal:
- NOT to maximize global SNR.
- Instead, constrain ANC to preserve task-relevant components (heart sounds),
  and only adaptively cancel noise in less task-critical frequency bands.

Rationale (heart sound @ 2 kHz):
- Heart sounds (S1/S2) are mainly concentrated in low frequencies.
- If ANC adapts on full-band, it may learn and cancel heart-sound structure (bad for classification).
- Therefore:
  1) Low band [0, PROTECT_HZ] is preserved from primary (no ANC).
  2) ANC runs only on selected subbands (e.g., 150–800 Hz), with per-band step size (mu)
     to control aggressiveness.
  3) Remaining frequencies (outside processed bands) are passed through from primary.

Input layout (from your mix step):
data/anc/
  snr5/
    primary/...wav
    reference/...wav
  snr10/...
  snr15/...
  snr20/...

Output layout:
data/anc_out_taskaware/
  snr5/anc/...wav
  snr10/anc/...wav
  snr15/anc/...wav
  snr20/anc/...wav
  manifest_anc_taskaware.csv

Run:
  python tools/anc/run_task_aware_anc.py

Dependencies:
  pip install numpy soundfile pandas (pandas only if you want; not required here)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Iterable, List

import numpy as np
import soundfile as sf

EPS = 1e-12

# =========================
# CONFIG (edit here)
# =========================
IN_ROOT = Path("data/anc_joint")
OUT_ROOT = Path("data/anc_out")
SNR_TAGS = [ "snr-10", "snr-5","snr0", "snr5","snr10", "snr15", "snr20"]

PRIMARY_SUBDIR = "primary"
REFERENCE_SUBDIR = "reference"
OUT_SUBDIR = "anc"

TARGET_SR = 2000

# --- Task-aware frequency design ---
# Preserve low band (heart-sound dominant) from primary directly
PROTECT_HZ = 250.0

# Subbands where ANC is allowed to adapt: (low_hz, high_hz, mu_scale)
# mu_scale multiplies BASE_MU (smaller => more conservative, less risk to signal)
# You can tune these. Start conservative.
SUBBANDS = [(250, 400, 0.5), (400, 900, 1.0)]

# Frequencies above the last band:
# - "pass": pass-through from primary
# - "mild": also run ANC with small mu
HIGH_MODE = "pass"         # "pass" or "mild"
HIGH_MU_SCALE = 0.20       # used if HIGH_MODE == "mild"
HIGH_HZ_MAX = None         # None => up to Nyquist

# --- NLMS params ---
FILTER_LEN = 512          # start smaller to reduce learning signal structure
BASE_MU = 0.01            # base step size (scaled per band)
LEAK = 1e-5                # leaky NLMS reduces long-term memorization of heart structure
DELTA = 1e-6               # NLMS stability term

# --- Delay alignment (recommended) ---
USE_DELAY_ALIGN = True
MAX_LAG_SAMPLES = 120      # at 2kHz => 60ms

# --- Output ---
WAV_SUBTYPE = "PCM_16"
MANIFEST_NAME = "manifest_anc_taskaware.csv"

# IMPORTANT: do NOT peak-normalize outputs; only clip
CLIP_OUTPUT = True
# =========================


@dataclass
class NLMSConfig:
    filter_len: int = FILTER_LEN
    mu: float = BASE_MU
    leak: float = LEAK
    delta: float = DELTA


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_mono(path: Path) -> Tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1).astype(np.float32)
    return x.astype(np.float32), int(sr)


def nlms_filter(d: np.ndarray, x: np.ndarray, cfg: NLMSConfig) -> np.ndarray:
    """
    Leaky NLMS adaptive filter. Returns residual e = d - y_hat.
    """
    N = min(len(d), len(x))
    d = d[:N].astype(np.float32)
    x = x[:N].astype(np.float32)

    L = int(cfg.filter_len)
    w = np.zeros(L, dtype=np.float32)
    xbuf = np.zeros(L, dtype=np.float32)
    e = np.zeros(N, dtype=np.float32)

    for n in range(N):
        xbuf[1:] = xbuf[:-1] #构造输入向量，参考麦历史窗口
        xbuf[0] = x[n]

        y_hat = float(np.dot(w, xbuf))
        err = float(d[n] - y_hat)
        e[n] = err

        norm = float(np.dot(xbuf, xbuf)) + float(cfg.delta)
        step = float(cfg.mu) / norm #归一化步长

        if cfg.leak > 0:
            w *= (1.0 - float(cfg.leak))

        w += (step * err) * xbuf #权重更新

    return e

def anti_clip(
    x: np.ndarray,
    headroom: float = 0.98,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Global anti-clipping by uniform scaling (no waveform distortion).

    Returns:
      y           : scaled signal
      write_scale : applied scale factor (<= 1.0)
      peak_before : max|x| before scaling
      peak_after  : max|y| after scaling
    """
    peak_before = float(np.max(np.abs(x)) + eps)

    if peak_before <= headroom:
        return x.astype(np.float32), 1.0, peak_before, peak_before

    write_scale = float(headroom / peak_before)
    y = (x * write_scale).astype(np.float32)
    peak_after = float(np.max(np.abs(y)) + eps)

    return y, write_scale, peak_before, peak_after



def estimate_delay(ref: np.ndarray, pri: np.ndarray, max_lag: int) -> int:
    """
    Estimate lag (samples) maximizing dot correlation (limited window).
    Positive lag => ref delayed relative to pri (shift right).
    """
    N = min(len(ref), len(pri))
    a = ref[:N].astype(np.float64)
    b = pri[:N].astype(np.float64)
    a -= a.mean()
    b -= b.mean()

    best_lag = 0
    best_val = -1e18
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            aa = a[lag:]
            bb = b[:len(aa)]
        else:
            aa = a[:lag]
            bb = b[-lag:]
        v = float(np.dot(aa, bb))
        if v > best_val:
            best_val = v
            best_lag = lag
    return best_lag


def apply_delay(x: np.ndarray, lag: int) -> np.ndarray:
    if lag == 0:
        return x
    if lag > 0:
        return np.concatenate([np.zeros(lag, dtype=x.dtype), x[:-lag]])
    lag = -lag
    return np.concatenate([x[lag:], np.zeros(lag, dtype=x.dtype)])


def rfft_masks(sr: int, n: int, bands: List[Tuple[float, float]]) -> List[np.ndarray]:
    """
    Create boolean masks (rfft bins) for each band.
    """
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    masks = []
    for lo, hi in bands:
        masks.append((freqs >= lo) & (freqs < hi))
    return masks


def apply_mask_time(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply rfft mask and return time-domain reconstruction (same length).
    """
    X = np.fft.rfft(x.astype(np.float64))
    X *= mask
    y = np.fft.irfft(X, n=len(x)).astype(np.float32)
    return y


def iter_pairs(primary_root: Path, reference_root: Path) -> Iterable[Tuple[Path, Path, Path]]:
    prim_files = sorted([p for p in primary_root.rglob("*.wav") if p.is_file()])
    for p_primary in prim_files:
        rel = p_primary.relative_to(primary_root)
        p_ref = reference_root / rel
        if not p_ref.exists():
            raise FileNotFoundError(f"Missing reference wav for {rel}: {p_ref}")
        yield p_primary, p_ref, rel


def main():
    ensure_dir(OUT_ROOT)
    manifest_path = OUT_ROOT / MANIFEST_NAME

    with open(manifest_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow([
        "snr_tag",
        "primary_rel", "reference_rel", "anc_rel",
        "sr",
        "filter_len", "base_mu", "leak",
        "protect_hz",
        "subbands",
        "use_delay_align", "max_lag_samples",
        "high_mode", "high_mu_scale",
        "write_scale", "peak_before", "peak_after"
    ])


        for tag in SNR_TAGS:
            primary_root = IN_ROOT / tag / PRIMARY_SUBDIR
            reference_root = IN_ROOT / tag / REFERENCE_SUBDIR
            if not primary_root.exists():
                raise FileNotFoundError(f"Primary dir not found: {primary_root}")
            if not reference_root.exists():
                raise FileNotFoundError(f"Reference dir not found: {reference_root}")

            out_root = OUT_ROOT / tag / OUT_SUBDIR
            ensure_dir(out_root)

            count = 0
            for p_primary, p_ref, rel in iter_pairs(primary_root, reference_root):
                pri, sr1 = read_mono(p_primary)
                ref, sr2 = read_mono(p_ref)
                if sr1 != sr2 or sr1 != TARGET_SR:
                    raise RuntimeError(f"SR mismatch for {rel}: pri={sr1}, ref={sr2}, target={TARGET_SR}")

                # Build frequency masks using current length
                n = len(pri)
                nyq = TARGET_SR / 2.0
                # Low protect band
                band_list = [(0.0, PROTECT_HZ)]
                # ANC subbands
                anc_band_edges = [(lo, hi) for (lo, hi, _) in SUBBANDS]
                band_list += anc_band_edges

                # High residual band
                high_lo = max([hi for (lo, hi, _) in SUBBANDS] + [PROTECT_HZ])
                high_hi = nyq if HIGH_HZ_MAX is None else min(HIGH_HZ_MAX, nyq)
                if high_hi > high_lo:
                    band_list.append((high_lo, high_hi))

                masks = rfft_masks(TARGET_SR, n, band_list)
                m_low = masks[0]
                m_subs = masks[1:1+len(SUBBANDS)]
                m_high = masks[1+len(SUBBANDS)] if (len(masks) > 1+len(SUBBANDS)) else None

                # Decompose
                pri_low = apply_mask_time(pri, m_low)
                ref_low = apply_mask_time(ref, m_low)  # not used, but available

                out = pri_low.copy()

                # Optional delay alignment (on an "ANC-operating proxy" band: union of subbands)
                if USE_DELAY_ALIGN:
                    # create union mask for subbands
                    union = m_subs[0].copy()
                    for mm in m_subs[1:]:
                        union |= mm
                    pri_u = apply_mask_time(pri, union)
                    ref_u = apply_mask_time(ref, union)
                    lag = estimate_delay(ref_u, pri_u, MAX_LAG_SAMPLES)
                    ref = apply_delay(ref, lag)

                # Process each ANC subband
                for (lo, hi, mu_scale), m in zip(SUBBANDS, m_subs):
                    pri_b = apply_mask_time(pri, m)
                    ref_b = apply_mask_time(ref, m)

                    cfg = NLMSConfig(
                        filter_len=FILTER_LEN,
                        mu=BASE_MU * float(mu_scale),
                        leak=LEAK,
                        delta=DELTA,
                    )
                    anc_b = nlms_filter(d=pri_b, x=ref_b, cfg=cfg)
                    out += anc_b

                # High band: pass-through or mild ANC
                if m_high is not None:
                    pri_h = apply_mask_time(pri, m_high)
                    if HIGH_MODE == "pass":
                        out += pri_h
                    elif HIGH_MODE == "mild":
                        ref_h = apply_mask_time(ref, m_high)
                        cfg = NLMSConfig(
                            filter_len=FILTER_LEN,
                            mu=BASE_MU * float(HIGH_MU_SCALE),
                            leak=LEAK,
                            delta=DELTA,
                        )
                        anc_h = nlms_filter(d=pri_h, x=ref_h, cfg=cfg)
                        out += anc_h
                    else:
                        raise ValueError(f"Unknown HIGH_MODE: {HIGH_MODE}")

                if CLIP_OUTPUT:
                    out, write_scale, peak_before, peak_after = anti_clip(
                        out,
                        headroom=0.98,
                    )   

                out_path = out_root / rel
                ensure_dir(out_path.parent)
                sf.write(str(out_path), out, TARGET_SR, subtype=WAV_SUBTYPE)

                wcsv.writerow([
                    tag,
                    (p_primary.relative_to(IN_ROOT)).as_posix(),
                    (p_ref.relative_to(IN_ROOT)).as_posix(),
                    (out_path.relative_to(OUT_ROOT)).as_posix(),
                    TARGET_SR,
                    FILTER_LEN, BASE_MU, LEAK,
                    PROTECT_HZ,
                    str(SUBBANDS),
                    int(USE_DELAY_ALIGN), MAX_LAG_SAMPLES,
                    HIGH_MODE, HIGH_MU_SCALE,
                    write_scale, peak_before, peak_after
                ])


                count += 1
                if count % 50 == 0:
                    print(f"[INFO] {tag}: processed {count} files")

            print(f"[DONE] {tag}: total {count} files -> {out_root}")

    print(f"[DONE] Outputs saved under: {OUT_ROOT}")
    print(f"[DONE] manifest: {manifest_path}")
    print(f"[PARAM] filter_len={FILTER_LEN} base_mu={BASE_MU} leak={LEAK}")
    print(f"[PARAM] protect_hz={PROTECT_HZ} subbands={SUBBANDS} high_mode={HIGH_MODE}")


if __name__ == "__main__":
    main()
