# tools/anc/make_primary_and_reference_joint.py
from __future__ import annotations

import csv
import glob
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

EPS = 1e-12

# -------------------------
# CONFIG
# -------------------------
PROJECT_ROOT = Path(".")

METADATA_PATH = PROJECT_ROOT / "data/features/metadata.csv"  # contains split + source
RAW_ROOT = PROJECT_ROOT / "data/raw"
MUSAN_ROOT = PROJECT_ROOT / "musan_noise_2k"

OUT_ROOT = PROJECT_ROOT / "data/anc_joint"
OUT_MANIFEST = OUT_ROOT / "manifest_primary_ref.csv"

TARGET_SR = 2000
PRIMARY_SNR_LIST = [-10,-5, 0, 5, 10, 15, 20]

REF_MODE = "fixed"     # "fixed" or "negate"
REF_SNR_FIXED = -20.0

GLOBAL_SEED = 42
HEADROOM = 0.98
WAV_SUBTYPE = "PCM_16"
# -------------------------


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def rms(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    return float(np.sqrt(np.mean(x * x) + EPS))


def collect_wavs(root: Path) -> List[Path]:
    return sorted([Path(p) for p in glob.glob(str(root / "**" / "*.wav"), recursive=True)])


def load_audio(path: Path, sr: int) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def pick_noise_segment_with_start(noise: np.ndarray, length: int, rng: random.Random) -> Tuple[np.ndarray, int, str, int]:
    n = int(len(noise))
    if n <= 0:
        raise RuntimeError("Noise wav has zero length.")
    if n >= length:
        start = rng.randint(0, n - length)
        seg = noise[start:start + length].astype(np.float32)
        return seg, int(start), "slice", 0
    reps = int(np.ceil(length / n))
    seg = np.tile(noise, reps)[:length].astype(np.float32)
    return seg, 0, "tile", int(reps)


def scale_noise_to_snr(clean: np.ndarray, noise_seg: np.ndarray, snr_db: float) -> float:
    rc = rms(clean)
    rn = rms(noise_seg)
    return float(rc / (rn * (10.0 ** (snr_db / 20.0)) + EPS))


def anti_clip(x: np.ndarray, headroom: float) -> Tuple[np.ndarray, float, float, float]:
    peak_before = float(np.max(np.abs(x)) + EPS)
    if peak_before <= headroom:
        return x.astype(np.float32), 1.0, peak_before, peak_before
    s = float(headroom / peak_before)
    y = (x * s).astype(np.float32)
    peak_after = float(np.max(np.abs(y)) + EPS)
    return y, s, peak_before, peak_after


def safe_rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def main():
    rng = random.Random(GLOBAL_SEED)
    ensure_dir(OUT_ROOT)
    ensure_dir(OUT_MANIFEST.parent)

    # ---- get test sources (recording-level) from metadata.csv ----
    import pandas as pd
    df = pd.read_csv(METADATA_PATH)
    if "split" not in df.columns or "source" not in df.columns:
        raise RuntimeError("metadata.csv must contain split/source.")
    test_sources = sorted(set(df[df["split"] == "test"]["source"].astype(str).tolist()))
    if not test_sources:
        raise RuntimeError("No test sources found in metadata.csv")

    noise_wavs = collect_wavs(MUSAN_ROOT)
    if not noise_wavs:
        raise RuntimeError(f"No noise wavs found under: {MUSAN_ROOT}")

    with open(OUT_MANIFEST, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "clean_rel",
            "noise_rel",
            "snr_primary_db",
            "snr_reference_db",
            "noise_start",
            "clean_len",
            "noise_len",
            "noise_seg_mode",
            "noise_tile_reps",
            "primary_rel",
            "reference_rel",
            "primary_write_scale",
            "ref_write_scale",
            "headroom",
            "sr",
            "seed",
        ])

        for src in tqdm(test_sources, desc="joint mix (primary+ref)"):
            clean_p = Path(src)
            if not clean_p.exists():
                cand = RAW_ROOT / clean_p
                if cand.exists():
                    clean_p = cand
            if not clean_p.exists():
                raise FileNotFoundError(f"Clean wav not found: {src}")

            clean = load_audio(clean_p, TARGET_SR)
            L = int(len(clean))
            folder = clean_p.parent.name
            stem = clean_p.stem

            noise_p = rng.choice(noise_wavs)
            noise = load_audio(noise_p, TARGET_SR)
            noise_len = int(len(noise))

            noise_seg, noise_start, mode, tile_reps = pick_noise_segment_with_start(noise, L, rng)

            clean_rel = safe_rel(clean_p, RAW_ROOT)
            noise_rel = safe_rel(noise_p, MUSAN_ROOT)

            for snr_p in PRIMARY_SNR_LIST:
                snr_r = (-float(snr_p)) if REF_MODE == "negate" else float(REF_SNR_FIXED)

                a_p = scale_noise_to_snr(clean, noise_seg, float(snr_p))
                a_r = scale_noise_to_snr(clean, noise_seg, float(snr_r))

                primary_f32 = (clean + a_p * noise_seg).astype(np.float32)
                ref_f32 = (clean + a_r * noise_seg).astype(np.float32)

                primary, p_scale, _, _ = anti_clip(primary_f32, HEADROOM)
                ref, r_scale, _, _ = anti_clip(ref_f32, HEADROOM)

                tag = f"snr{int(snr_p)}"
                out_p = OUT_ROOT / tag / "primary" / folder / f"{stem}.wav"
                out_r = OUT_ROOT / tag / "reference" / folder / f"{stem}.wav"
                ensure_dir(out_p.parent)
                ensure_dir(out_r.parent)

                sf.write(str(out_p), primary, TARGET_SR, subtype=WAV_SUBTYPE)
                sf.write(str(out_r), ref, TARGET_SR, subtype=WAV_SUBTYPE)

                w.writerow([
                    clean_rel,
                    noise_rel,
                    f"{float(snr_p):.2f}",
                    f"{float(snr_r):.2f}",
                    int(noise_start),
                    int(L),
                    int(noise_len),
                    mode,
                    int(tile_reps),
                    safe_rel(out_p, OUT_ROOT),
                    safe_rel(out_r, OUT_ROOT),
                    f"{p_scale:.10f}",
                    f"{r_scale:.10f}",
                    f"{HEADROOM:.3f}",
                    int(TARGET_SR),
                    int(GLOBAL_SEED),
                ])

    print(f"[DONE] joint primary+reference -> {OUT_ROOT}")
    print(f"[DONE] manifest -> {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
