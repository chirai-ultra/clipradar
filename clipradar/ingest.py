"""Get the video: download a link with yt-dlp (1,000+ sites) or use a local file.

Also collects the metadata ClipRadar scores with: title, tags, chapters, description,
and YouTube's "Most Replayed" heatmap when the site exposes it.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Media:
    path: Path
    title: str = ""
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 30.0
    has_audio: bool = True
    tags: list[str] = field(default_factory=list)
    description: str = ""
    chapters: list[dict] = field(default_factory=list)
    heatmap: list[dict] = field(default_factory=list)   # [{start_time, end_time, value}]
    url: str | None = None
    uploader: str = ""
    crop: tuple[int, int, int, int] | None = None   # (w, h, x, y) of the picture inside black bars


def is_url(s: str) -> bool:
    return bool(re.match(r"^https?://", s.strip(), re.I))


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def _fill_from_probe(m: Media) -> Media:
    info = probe(m.path)
    v = next((s for s in info["streams"] if s.get("codec_type") == "video"), None)
    if v is None:
        raise ValueError("The file has no video stream.")
    m.has_audio = any(s.get("codec_type") == "audio" for s in info["streams"])
    m.width, m.height = int(v.get("width", 0)), int(v.get("height", 0))
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = abs(int(sd["rotation"]))
    if rot in (90, 270):
        m.width, m.height = m.height, m.width
    num, _, den = (v.get("avg_frame_rate") or "30/1").partition("/")
    try:
        m.fps = float(num) / float(den or 1) or 30.0
    except (ValueError, ZeroDivisionError):
        m.fps = 30.0
    m.duration = float(info["format"].get("duration") or v.get("duration") or 0)
    return m


def detect_borders(m: Media, samples: int = 6) -> tuple[int, int, int, int] | None:
    """Find baked-in black bars (letterbox/pillarbox) with ffmpeg cropdetect at a few points."""
    found = []
    for i in range(samples):
        t = m.duration * (i + 1) / (samples + 1)
        r = subprocess.run(["ffmpeg", "-v", "info", "-ss", f"{t:.2f}", "-i", str(m.path), "-t", "1", "-an",
                            "-vf", "cropdetect=limit=24:round=2:reset=0", "-f", "null", "-"],
                           capture_output=True, text=True)
        hits = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", r.stderr)
        if hits:
            found.append(tuple(int(v) for v in hits[-1]))
    if not found:
        return None
    # union of what is ever visible, so we never cut real picture
    x0 = min(f[2] for f in found)
    y0 = min(f[3] for f in found)
    x1 = max(f[2] + f[0] for f in found)
    y1 = max(f[3] + f[1] for f in found)
    w, h = (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2
    if w <= 0 or h <= 0 or w * h > 0.93 * m.width * m.height or w < m.width * 0.25 or h < m.height * 0.25:
        return None
    return (w, h, x0, y0)


def fetch(source: str, work: Path, max_height: int = 1080, cookies: str | None = None, log=print) -> Media:
    if not is_url(source):
        p = Path(source).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"No such file: {p}")
        return _fill_from_probe(Media(path=p, title=p.stem))

    try:
        import yt_dlp
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("yt-dlp is required for links: pip install yt-dlp") from e

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not installed or not on PATH.")
    dl = work / "source"
    dl.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": f"bv*[height<={max_height}]+ba/b[height<={max_height}]/bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(dl / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    if cookies:
        opts["cookiefile"] = cookies
    log("Downloading video…")
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(source, download=True)
        path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
        if not path.exists():
            cands = sorted(dl.glob(f"{info['id']}.*"))
            path = cands[0] if cands else path
    m = Media(
        path=path, url=source,
        title=info.get("title") or "",
        tags=[t.lower() for t in (info.get("tags") or [])],
        description=(info.get("description") or "")[:5000],
        chapters=info.get("chapters") or [],
        heatmap=info.get("heatmap") or [],
        uploader=info.get("uploader") or "",
    )
    _fill_from_probe(m)
    if m.heatmap:
        log(f"Found YouTube 'Most Replayed' data ({len(m.heatmap)} points).")
    return m
