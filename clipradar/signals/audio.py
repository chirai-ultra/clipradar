"""Audio excitement: loudness, sudden onsets (spectral flux) and percussive punch.

Gameplay highlights (kills, explosions, crowd/caster hype) and talk highlights (laughter,
raised voices) both show up as energy spikes, so this works with no speech model at all.
Pure numpy: no librosa needed.
"""
from __future__ import annotations

import subprocess

import numpy as np

SR = 16000


def load_mono(path, sr: int = SR) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _norm(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = np.percentile(x, 5), np.percentile(x, 99)
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def analyse(y: np.ndarray, duration: float, sr: int = SR) -> dict[str, np.ndarray]:
    n_sec = max(1, int(np.ceil(duration)))
    hop = sr // 10                      # 100 ms frames
    win = 1024
    n_frames = max(1, (len(y) - win) // hop + 1)
    if len(y) < win:
        y = np.pad(y, (0, win - len(y)))
        n_frames = 1
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = y[idx] * np.hanning(win)[None, :]
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)

    spec = np.abs(np.fft.rfft(frames, axis=1))
    logspec = np.log1p(spec)
    flux = np.maximum(0, np.diff(logspec, axis=0, prepend=logspec[:1])).sum(axis=1)
    freqs = np.fft.rfftfreq(win, 1 / sr)
    low = spec[:, freqs < 200].sum(axis=1)           # bass hits / explosions
    high = spec[:, freqs > 3000].sum(axis=1)         # crowd, cheers, gunfire crack
    total = spec.sum(axis=1) + 1e-9

    def per_sec(a: np.ndarray, how=np.mean) -> np.ndarray:
        out = np.zeros(n_sec, dtype=np.float32)
        for s in range(n_sec):
            seg = a[s * 10:(s + 1) * 10]
            out[s] = how(seg) if seg.size else 0
        return out

    loud = per_sec(db)
    # "Excitement" = loud relative to the local context, not just loud overall
    k = 30
    local = np.convolve(loud, np.ones(k) / k, mode="same")
    rel = loud - local
    silence = per_sec(db, np.min)

    return {
        "loudness": _norm(loud),
        "surprise": _norm(rel),
        "onsets": _norm(per_sec(flux, np.max)),
        "punch": _norm(per_sec(low / total)),
        "bright": _norm(per_sec(high / total)),
        "silence_db": silence,           # raw dB, used for boundary snapping
        "frame_db": db,                  # 100 ms resolution
    }
