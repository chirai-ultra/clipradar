"""Speech to word-level timestamps. Pluggable backends:

* faster-whisper  (default, 99 languages; pip install faster-whisper)
* sherpa          (sherpa-onnx + NVIDIA Parakeet, English, very fast on CPU;
                   pip install sherpa-onnx; model auto-downloads from GitHub)
* none            (skip speech; ClipRadar still works from audio/visual/heatmap)
"""
from __future__ import annotations

import os
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SHERPA_MODEL = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
SHERPA_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{SHERPA_MODEL}.tar.bz2"


@dataclass
class Word:
    start: float
    end: float
    text: str


def _cache() -> Path:
    p = Path(os.getenv("CLIPRADAR_CACHE", Path.home() / ".cache" / "clipradar"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def available_backends() -> list[str]:
    out = []
    try:
        import faster_whisper  # noqa: F401
        out.append("faster-whisper")
    except ImportError:
        pass
    try:
        import sherpa_onnx  # noqa: F401
        out.append("sherpa")
    except ImportError:
        pass
    return out


def transcribe(y: np.ndarray, sr: int, backend: str = "auto", model: str = "small",
               language: str | None = None, log=print) -> list[Word]:
    if backend == "none" or y.size == 0:
        return []
    order = [backend] if backend != "auto" else (
        ["sherpa", "faster-whisper"] if os.getenv("CLIPRADAR_SHERPA_MODEL") else ["faster-whisper", "sherpa"])
    errors = []
    for b in order:
        try:
            if b == "faster-whisper":
                return _whisper(y, sr, model, language, log)
            if b == "sherpa":
                return _sherpa(y, sr, log)
        except Exception as e:  # try the next backend
            errors.append(f"{b}: {e}")
    log("Speech recognition unavailable, continuing without captions (" + "; ".join(errors) + ")")
    return []


def _whisper(y, sr, model, language, log) -> list[Word]:
    from faster_whisper import WhisperModel
    device = os.getenv("CLIPRADAR_DEVICE", "auto")
    compute = "int8" if device in ("auto", "cpu") else "float16"
    log(f"Transcribing with faster-whisper ({model})…")
    wm = WhisperModel(model, device=device, compute_type=compute)
    segs, _ = wm.transcribe(y, language=language, word_timestamps=True, vad_filter=True)
    words = []
    for s in segs:
        for w in s.words or []:
            words.append(Word(float(w.start), float(w.end), w.word.strip()))
    return words


def _sherpa_dir(log) -> Path:
    env = os.getenv("CLIPRADAR_SHERPA_MODEL")
    if env:
        return Path(env)
    d = _cache() / SHERPA_MODEL
    if (d / "tokens.txt").exists():
        return d
    log("Downloading Parakeet speech model (~600 MB, one time)…")
    tmp = _cache() / (SHERPA_MODEL + ".tar.bz2")
    urllib.request.urlretrieve(SHERPA_URL, tmp)
    with tarfile.open(tmp) as t:
        t.extractall(_cache(), filter="data")
    tmp.unlink(missing_ok=True)
    return d


def _sherpa(y, sr, log) -> list[Word]:
    import sherpa_onnx
    d = _sherpa_dir(log)
    log("Transcribing with sherpa-onnx Parakeet…")
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
        joiner=str(d / "joiner.int8.onnx"), tokens=str(d / "tokens.txt"),
        model_type="nemo_transducer", num_threads=max(1, (os.cpu_count() or 2) - 1))
    chunk = 30 * sr
    words: list[Word] = []
    for off in range(0, len(y), chunk):
        part = y[off: off + chunk]
        if part.size < sr // 2:
            continue
        st = rec.create_stream()
        st.accept_waveform(sr, part)
        rec.decode_stream(st)
        r = st.result
        base = off / sr
        cur: list[list] = []
        for tok, ts in zip(r.tokens, r.timestamps):
            if tok.startswith(" ") or not cur:
                cur.append([tok.strip(), base + ts])
            else:
                cur[-1][0] += tok
        for i, (txt, s) in enumerate(cur):
            nxt = cur[i + 1][1] if i + 1 < len(cur) else s + 0.5
            if txt:
                words.append(Word(round(s, 2), round(min(nxt, s + 0.9), 2), txt))
    return words
