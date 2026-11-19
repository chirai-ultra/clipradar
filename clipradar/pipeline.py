"""End-to-end: fetch -> analyse -> score -> pick -> render. Used by the CLI, the web app and the API."""
from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from . import ingest, keywords, llm, render, score, transcribe
from .config import Options, llm_settings, work_dir
from .signals import audio, visual

Progress = Callable[[str, float, str], None]   # (stage, 0..1, message)


def _slug(s: str, n: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return (s[:n].rstrip("-") or "video")


def _text(words, a: float, b: float) -> str:
    return " ".join(w.text for w in words if a <= w.start < b)


def _auto_title(words, vocab, a, b) -> str:
    """A hook title from the clip's own speech: the most on-topic complete short sentence.
    Returns "" rather than a fragment cut mid-thought (set an LLM for punchier titles)."""
    ws = [w for w in words if a <= w.start < b]
    sentences, cur = [], []
    for w in ws:
        cur.append(w)
        if w.text.endswith((".", "!", "?")):
            sentences.append(cur)
            cur = []
    best, best_s = "", 0.0
    for i, s in enumerate(sentences[:8]):
        if not 4 <= len(s) <= 8:
            continue
        sc = sum(vocab.get(re.sub(r"[^a-z0-9']", "", x.text.lower()), 0) for x in s)
        sc += 0.6 if s[-1].text.endswith(("!", "?")) else 0
        sc += 0.4 if i == 0 else 0
        if sc > best_s:
            best, best_s = " ".join(x.text for x in s).strip(" ,."), sc
    return best if len(best) > 14 else ""


def _srt(words, a, b, path: Path):
    def ts(t):
        t = max(0, t)
        return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d},{int(t * 1000 % 1000):03d}"
    ws = [w for w in words if a <= w.start < b]
    out, i, n = [], 0, 1
    while i < len(ws):
        g = ws[i:i + 7]
        out.append(f"{n}\n{ts(g[0].start - a)} --> {ts(g[-1].end - a)}\n{' '.join(x.text for x in g)}\n")
        i += 7
        n += 1
    if out:
        path.write_text("\n".join(out), encoding="utf-8")


def run(opts: Options, progress: Progress | None = None, log=print) -> dict:
    o = opts.normalised()
    t0 = time.time()
    p = progress or (lambda stage, f, msg: None)

    def say(stage, f, msg):
        p(stage, f, msg)
        log(msg)

    out_root = work_dir(o.out_dir)
    say("fetch", 0.02, "Getting the video…")
    media = ingest.fetch(o.source, out_root / ".work", o.max_download_height, o.cookies, log=log)
    if media.duration <= 0:
        raise ValueError("Could not read the video's duration.")
    job = out_root / f"{_slug(media.title or Path(media.path).stem)}-{time.strftime('%Y%m%d-%H%M%S')}"
    job.mkdir(parents=True, exist_ok=True)
    n_sec = int(np.ceil(media.duration))

    say("audio", 0.12, "Listening for excitement (loudness, impacts, crowd)…")
    sig: dict[str, np.ndarray] = {}
    y = np.zeros(0, dtype=np.float32)
    if media.has_audio:
        y = audio.load_mono(media.path)
        sig.update(audio.analyse(y, media.duration))
    else:
        for k in ("loudness", "surprise", "onsets", "punch", "bright"):
            sig[k] = np.zeros(n_sec, dtype=np.float32)

    say("visual", 0.25, "Scanning motion and scene cuts…")
    vis = visual.analyse(media.path, media.duration, media.width, media.height)
    cuts = vis.pop("cuts")
    action_x = vis.pop("action_x")
    sig.update(vis)
    sig["heatmap"] = score.heatmap_curve(media.heatmap, n_sec)

    words = []
    if o.asr != "none" and media.has_audio:
        say("speech", 0.38, "Transcribing speech…")
        words = transcribe.transcribe(y, audio.SR, o.asr, o.asr_model, o.language, log=log)

    say("keywords", 0.55, "Matching keywords and trends…")
    meta = keywords.topic_terms(media.title, media.tags, media.description, media.chapters)
    trend_list: list[str] = []
    if o.trends:
        topic = " ".join(o.keywords[:2]) or " ".join(t for t, _ in meta.most_common(3))
        trend_list = keywords.trend_terms(topic)
    vocab = keywords.build_vocab(meta, o.keywords, trend_list)
    sig["keywords"], sig["speech"] = keywords.score_transcript(words, vocab, n_sec)
    for k in list(sig):
        if k not in ("frame_db",) and sig[k].ndim == 1 and len(sig[k]) != n_sec:
            sig[k] = np.resize(sig[k], n_sec) if len(sig[k]) else np.zeros(n_sec, dtype=np.float32)

    content = o.content if o.content != "auto" else score.detect_content(sig["speech"], sig["motion"], sig["bright"])
    lo, hi = score.PRESET_LEN[content]
    min_len = o.min_len or lo
    max_len = max(o.max_len or hi, min_len + 1)
    say("score", 0.62, f"Content looks like {content}. Finding the best {o.clips} moments…")
    curve, used = score.fuse(sig, content)
    starts, ends = score.boundaries(sig, words, cuts)
    cfg = llm_settings() if o.llm else None
    want = min(10, o.clips * 2) if cfg else o.clips
    clips = score.pick(curve, want, min_len, max_len, media.duration, starts, ends)
    # short video and many clips asked for: shrink the length window until they fit
    tries = 0
    while len(clips) < o.clips and tries < 3 and not (o.min_len or o.max_len):
        min_len, max_len = max(6.0, min_len * 0.65), max(10.0, max_len * 0.65)
        clips = score.pick(curve, want, min_len, max_len, media.duration, starts, ends, gap=1.0)
        tries += 1
    if len(clips) < o.clips:
        log(f"Only {len(clips)} distinct moments fit in this video.")
    for c in clips:
        c.reason = score.explain(c, sig, used)
        c.title = _auto_title(words, vocab, c.start, c.end)
    if cfg and clips:
        say("llm", 0.66, "Asking the LLM to re-rank…")
        llm.rerank(cfg, media.title, clips, [_text(words, c.start, c.end) for c in clips], log=log)
        clips.sort(key=lambda c: -c.score)
    clips = clips[: o.clips]

    media.crop = ingest.detect_borders(media)
    pw, ph = (media.crop[0], media.crop[1]) if media.crop else (media.width, media.height)
    if media.crop:
        log(f"Removing black bars: picture is {pw}x{ph} inside {media.width}x{media.height}.")
    W, H = render.out_size(o.layout, pw, ph)
    framing = o.framing
    if framing == "auto":
        src_r, tgt_r = pw / max(1, ph), W / H
        framing = "smart" if (content == "talk" or abs(src_r - tgt_r) < 0.35) else "fit"
    results = []
    for i, c in enumerate(clips, 1):
        frac = 0.68 + 0.3 * (i - 1) / max(1, len(clips))
        say("render", frac, f"Rendering clip {i}/{len(clips)} in HD ({W}x{H})…")
        name = f"clip{i:02d}_{_slug(c.title, 30) if c.title else int(c.start)}"
        track = None
        if framing == "smart" and pw / max(1, ph) > W / H:
            track = render.face_track(media.path, c.start, c.end, media.width, media.height)
            if track is None:
                a, b = int(c.start), max(int(c.start) + 1, int(np.ceil(c.end)))
                track = action_x[a:b].copy()
            if media.crop:   # full-frame position -> position inside the de-barred picture
                track = np.clip((track * media.width - media.crop[2]) / pw, 0, 1)
        ass = None
        if o.captions or (o.hook_title and c.title):
            ass_p = job / f"{name}.ass"
            if render.write_ass(ass_p, words if o.captions else [], c.start, c.end, W, H,
                                c.title if o.hook_title else ""):
                ass = ass_p
        mp4 = job / f"{name}.mp4"
        render.render(media.path, mp4, c.start, c.end, W=W, H=H, src_w=pw, src_h=ph,
                      framing=framing, track=track, ass=ass, fps=o.fps or media.fps,
                      has_audio=media.has_audio, quality=o.quality, pre_crop=media.crop)
        render.thumbnail(mp4, job / f"{name}.jpg", min(c.duration * 0.33, 3))
        if words:
            _srt(words, c.start, c.end, job / f"{name}.srt")
        if ass:
            ass.unlink(missing_ok=True)
        results.append({"rank": i, "file": mp4.name, "thumb": f"{name}.jpg", "start": c.start, "end": c.end,
                        "duration": round(c.duration, 2), "score": c.score, "title": c.title, "why": c.reason,
                        "signals": c.signals, "transcript": _text(words, c.start, c.end)[:600]})

    manifest = {
        "source": o.source, "title": media.title, "duration": media.duration, "content": content,
        "layout": f"{W}x{H}", "framing": framing, "removed_bars": bool(media.crop), "used_heatmap": bool(media.heatmap),
        "trends": trend_list, "signal_weights": used, "options": asdict(o),
        "clips": results, "seconds": round(time.time() - t0, 1),
        "curve": [round(float(v), 3) for v in curve[:: max(1, n_sec // 400)]],
    }
    (job / "clips.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if ingest.is_url(o.source) and not o.keep_source:
        shutil.rmtree(out_root / ".work" / "source", ignore_errors=True)
    say("done", 1.0, f"Done: {len(results)} clips in {manifest['seconds']}s -> {job}")
    manifest["dir"] = str(job)
    return manifest
