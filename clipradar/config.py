"""Options shared by the CLI, the web app and the Python API."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

MAX_CLIPS = 10
LAYOUTS = ("auto", "vertical", "landscape", "square")
FRAMINGS = ("auto", "stream", "smart", "fit", "center")
CONTENT_TYPES = ("auto", "gaming", "talk", "sports")


@dataclass
class Options:
    source: str = ""                       # URL or local file path
    clips: int = 5                         # how many clips (1..10)
    min_len: float = 0.0                   # 0 = preset default
    max_len: float = 0.0
    content: str = "auto"                  # auto | gaming | talk | sports
    layout: str = "vertical"               # vertical 1080x1920 | landscape 1920x1080 | square 1080x1080 | auto
    framing: str = "auto"                  # stream (facecam top + gameplay) | smart (follow faces/motion) | fit (blurred background) | center | auto
    captions: bool = True                  # burn word-by-word captions when speech exists
    hook_title: bool = True                # big title for the first 3 seconds when one is available
    quality: str = "fast"                  # fast (veryfast x264) | best (slow x264, smaller + sharper)
    keywords: list[str] = field(default_factory=list)   # user keywords to favour
    trends: bool = True                    # look up live search suggestions for the topic
    asr: str = "auto"                      # auto | faster-whisper | sherpa | none
    asr_model: str = "small"               # faster-whisper size, or a sherpa model folder
    language: str | None = None
    llm: bool = True                       # use an LLM to re-rank if one is configured
    fps: int = 0                           # 0 = keep source (capped at 60)
    out_dir: str = "clipradar_output"
    max_download_height: int = 1080
    cookies: str | None = None             # cookies.txt for yt-dlp (age/region-locked videos)
    keep_source: bool = False

    def normalised(self) -> "Options":
        o = Options(**asdict(self))
        o.clips = max(1, min(MAX_CLIPS, int(o.clips)))
        if o.layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}")
        if o.framing not in FRAMINGS:
            raise ValueError(f"framing must be one of {FRAMINGS}")
        if o.content not in CONTENT_TYPES:
            raise ValueError(f"content must be one of {CONTENT_TYPES}")
        o.keywords = [k.strip().lower() for k in o.keywords if k and k.strip()]
        return o


def llm_settings() -> dict | None:
    """An OpenAI-compatible endpoint (OpenAI, OpenRouter, Groq, Gemini's OpenAI API, Ollama, LM Studio...)."""
    base = os.getenv("CLIPRADAR_LLM_BASE_URL")
    key = os.getenv("CLIPRADAR_LLM_API_KEY", "")
    model = os.getenv("CLIPRADAR_LLM_MODEL")
    if not base or not model:
        return None
    return {"base_url": base.rstrip("/"), "api_key": key, "model": model}


def work_dir(out_dir: str) -> Path:
    p = Path(out_dir).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p
