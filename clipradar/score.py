"""Fuse every signal into one "highlight curve", then pick the best non-overlapping clips."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Weights per content type. Heatmap (real viewer replays) is the strongest evidence when present.
WEIGHTS = {
    "gaming": dict(heatmap=0.30, surprise=0.18, onsets=0.14, punch=0.08, bright=0.06, loudness=0.06,
                   motion=0.08, cuts_density=0.02, keywords=0.06, speech=0.02),
    "talk":   dict(heatmap=0.32, surprise=0.10, onsets=0.04, punch=0.00, bright=0.04, loudness=0.08,
                   motion=0.04, cuts_density=0.02, keywords=0.26, speech=0.10),
    "sports": dict(heatmap=0.30, surprise=0.16, onsets=0.10, punch=0.04, bright=0.12, loudness=0.08,
                   motion=0.10, cuts_density=0.02, keywords=0.06, speech=0.02),
}
PRESET_LEN = {"gaming": (15, 45), "talk": (25, 60), "sports": (12, 40)}


@dataclass
class Clip:
    start: float
    end: float
    score: float
    title: str = ""
    reason: str = ""
    signals: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start


def heatmap_curve(heatmap: list[dict], n_sec: int) -> np.ndarray:
    h = np.zeros(n_sec, dtype=np.float32)
    for p in heatmap or []:
        a, b = int(p.get("start_time", 0)), int(np.ceil(p.get("end_time", 0)))
        h[max(0, a):min(n_sec, max(a + 1, b))] = max(0.0, float(p.get("value", 0)))
    if h.max() > 0:
        # values are relative; emphasise the replay spikes, not the intro bump
        h = (h - np.median(h)).clip(0)
        h = h / (h.max() or 1)
    return h


def detect_content(speech_density: np.ndarray, motion: np.ndarray, bright: np.ndarray) -> str:
    talkiness = float((speech_density > 0.25).mean()) if speech_density.size else 0
    if talkiness > 0.45 and float(motion.mean()) < 0.35:
        return "talk"
    if float(bright.mean()) > 0.45 and talkiness > 0.2:
        return "sports"
    return "gaming" if talkiness < 0.45 else "talk"


def fuse(sig: dict[str, np.ndarray], content: str) -> tuple[np.ndarray, dict]:
    w = dict(WEIGHTS[content])
    n = len(next(iter(sig.values())))
    if not sig.get("heatmap", np.zeros(1)).any():
        w.pop("heatmap")
    if not sig.get("keywords", np.zeros(1)).any():
        w.pop("keywords", None)
        w.pop("speech", None)
    total = sum(w.values()) or 1
    curve = np.zeros(n, dtype=np.float32)
    used = {}
    for k, wt in w.items():
        arr = sig.get(k)
        if arr is None or len(arr) != n:
            continue
        curve += (wt / total) * arr
        used[k] = round(wt / total, 3)
    # gentle smoothing so a single loud frame does not win
    curve = np.convolve(curve, np.array([0.25, 0.5, 1, 0.5, 0.25]) / 2.5, mode="same")
    # dead air never makes a highlight
    if "silence_db" in sig:
        curve[sig["silence_db"] < -55] *= 0.3
    return curve, used


def _snap(t: float, anchors: np.ndarray, radius: float) -> float:
    if anchors.size == 0:
        return t
    d = np.abs(anchors - t)
    i = int(d.argmin())
    return float(anchors[i]) if d[i] <= radius else t


def boundaries(sig: dict, words, cuts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Good places to start (scene cuts, pauses before speech) and end (sentence ends, pauses)."""
    starts, ends = [], []
    db = sig.get("frame_db")
    if db is not None and db.size:
        quiet = db < (np.percentile(db, 20) + 3)
        edges = np.where(np.diff(quiet.astype(int)) != 0)[0]
        ts = edges / 10.0
        starts += list(ts)
        ends += list(ts)
    starts += list(cuts)
    for i, w in enumerate(words or []):
        gap = (words[i + 1].start - w.end) if i + 1 < len(words) else 1.0
        if w.text.endswith((".", "!", "?")) or gap > 0.6:
            ends.append(w.end + 0.15)
            if i + 1 < len(words):
                starts.append(max(0.0, words[i + 1].start - 0.15))
    return np.array(sorted(starts), dtype=np.float32), np.array(sorted(ends), dtype=np.float32)


def pick(curve: np.ndarray, n: int, min_len: float, max_len: float, duration: float,
         starts: np.ndarray, ends: np.ndarray, gap: float = 3.0) -> list[Clip]:
    if duration <= min_len + 1:
        return [Clip(0.0, duration, float(curve.mean()) if curve.size else 0)]
    work = curve.copy()
    clips: list[Clip] = []
    target = (min_len + max_len) / 2
    kernel = np.ones(max(1, int(min_len))) / max(1, int(min_len))
    half = max(1, int(min_len) // 2)
    for _ in range(10 * len(curve) + 50):          # hard cap: never loop forever
        if len(clips) >= n:
            break
        dens = np.convolve(work, kernel, mode="same")
        if dens.max() <= 1e-6:
            break
        peak = int(dens.argmax())
        level = float(dens[peak])
        # grow around the peak while the action continues
        thr = 0.6 * level
        a = b = peak
        while a > 0 and (b - a) < max_len and dens[a - 1] >= thr:
            a -= 1
        while b < len(dens) - 1 and (b - a) < max_len and dens[b + 1] >= thr:
            b += 1
        length = float(np.clip(b - a, min_len, max_len))
        if b - a < target * 0.7:
            length = max(min_len, min(max_len, target * 0.8))
        # hook early: put the strongest second about a third of the way in
        hot = a + int(np.argmax(curve[a:b + 1])) if b > a else peak
        s = max(0.0, hot - length * 0.35)
        e = min(duration, s + length)
        s = max(0.0, e - length)
        s = _snap(s, starts, 2.0)
        e = _snap(e, ends, 2.5)
        if e - s < min_len:
            e = min(duration, s + min_len)
        if e - s > max_len + 3:
            e = s + max_len
        if any(not (e + gap <= c.start or s >= c.end + gap) for c in clips):
            work[max(0, peak - half):peak + half + 1] = 0   # whole kernel, or the same peak returns
            continue
        seg = curve[int(s):max(int(s) + 1, int(e))]
        clips.append(Clip(round(s, 2), round(e, 2), float(seg.mean() * 0.6 + seg.max() * 0.4)))
        work[max(0, int(s - gap)):int(e + gap) + 1] = 0
    clips.sort(key=lambda c: -c.score)
    if clips:
        top = clips[0].score or 1
        for c in clips:
            c.score = round(100 * c.score / top, 1)
    return clips


def explain(clip: Clip, sig: dict, used: dict) -> str:
    a, b = int(clip.start), max(int(clip.start) + 1, int(clip.end))
    contrib = {k: float(sig[k][a:b].mean()) * w for k, w in used.items() if k in sig}
    labels = {"heatmap": "most-replayed by viewers", "surprise": "sudden audio spike", "onsets": "rapid hits",
              "punch": "big bass impacts", "bright": "crowd/cheer energy", "loudness": "loud moment",
              "motion": "fast action", "cuts_density": "quick cuts", "keywords": "keyword/trend match",
              "speech": "dense dialogue"}
    top = sorted(contrib.items(), key=lambda kv: -kv[1])[:3]
    clip.signals = {k: round(v, 3) for k, v in contrib.items()}
    return ", ".join(labels.get(k, k) for k, _ in top)
