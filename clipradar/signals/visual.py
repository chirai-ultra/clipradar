"""Visual signals from a low-res pass: scene cuts, motion intensity and where the action is.

Reads frames at 4 fps and 160 px wide with ffmpeg, so a 1-hour video takes a minute or so.
Also returns a per-second horizontal "action centre" used for smart vertical cropping.
"""
from __future__ import annotations

import subprocess

import numpy as np

W, FPS = 160, 4


def _frames(path, src_w: int, src_h: int):
    h = max(2, int(round(W * src_h / max(src_w, 1) / 2) * 2))
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-an", "-vf", f"fps={FPS},scale={W}:{h},format=gray",
           "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = W * h
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(h, W).astype(np.float32)
    finally:
        p.stdout.close()
        p.wait()


def _norm(x):
    if x.size == 0:
        return x
    lo, hi = np.percentile(x, 5), np.percentile(x, 99)
    return np.zeros_like(x) if hi - lo < 1e-9 else np.clip((x - lo) / (hi - lo), 0, 1)


def analyse(path, duration: float, src_w: int, src_h: int) -> dict[str, np.ndarray]:
    n_sec = max(1, int(np.ceil(duration)))
    motion = np.zeros(n_sec * FPS + FPS, dtype=np.float32)
    cutscore = np.zeros_like(motion)
    centre = np.full_like(motion, 0.5)
    prev = None
    prev_hist = None
    i = 0
    for f in _frames(path, src_w, src_h):
        if i >= motion.size:
            break
        hist = np.histogram(f, bins=32, range=(0, 255))[0].astype(np.float32)
        hist /= hist.sum() + 1e-9
        if prev is not None:
            d = np.abs(f - prev)
            motion[i] = d.mean()
            col = d.sum(axis=0)
            if col.sum() > 1e-6:
                xs = np.arange(col.size)
                centre[i] = float((col * xs).sum() / col.sum() / col.size)
            cutscore[i] = 0.5 * np.abs(hist - prev_hist).sum()
        prev, prev_hist = f, hist
        i += 1

    cuts = np.where(cutscore > 0.45)[0] / FPS
    m = motion[: n_sec * FPS].reshape(n_sec, FPS)
    c = centre[: n_sec * FPS].reshape(n_sec, FPS)
    weights = m + 1e-6
    centre_sec = (c * weights).sum(axis=1) / weights.sum(axis=1)
    cut_density = np.zeros(n_sec, dtype=np.float32)
    for t in cuts:
        s = int(t)
        if s < n_sec:
            cut_density[s] += 1
    cut_density = np.convolve(cut_density, np.ones(5), mode="same")
    return {
        "motion": _norm(m.mean(axis=1)),
        "cuts_density": _norm(cut_density),
        "cuts": cuts.astype(np.float32),
        "action_x": centre_sec.astype(np.float32),
    }
