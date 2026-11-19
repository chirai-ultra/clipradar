"""Render a clip in HD: reframe (smart crop / blurred fit / centre), burn captions, upscale cleanly,
normalise loudness to -14 LUFS (the TikTok/Reels/Shorts target) and write H.264 + AAC MP4."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).parent / "assets"
FONT_NAME = "Inter Black"
# Palette (ink, paint red, cadmium yellow, leaf green, cobalt) shared with the web UI
PALETTE = {"ink": "#151E11", "red": "#E0301E", "yellow": "#F7D51B", "green": "#3B6E2C", "blue": "#3E6DB0"}
SIZES = {"vertical": (1080, 1920), "landscape": (1920, 1080), "square": (1080, 1080)}


def out_size(layout: str, src_w: int, src_h: int) -> tuple[int, int]:
    if layout != "auto":
        return SIZES[layout]
    r = src_w / max(src_h, 1)
    return SIZES["vertical"] if r < 0.8 else SIZES["square"] if r < 1.2 else SIZES["landscape"]


# ---------- framing ----------

YUNET_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")


def _face_detector():
    """YuNet (OpenCV Zoo, MIT, ~230 KB, auto-downloaded) with the classic Haar cascade as fallback."""
    import os
    import urllib.request
    import cv2
    cache = Path(os.getenv("CLIPRADAR_CACHE", Path.home() / ".cache" / "clipradar"))
    cache.mkdir(parents=True, exist_ok=True)
    model = cache / "face_detection_yunet_2023mar.onnx"
    try:
        if not model.exists() or model.stat().st_size < 10000:
            urllib.request.urlretrieve(YUNET_URL, model)
        det = cv2.FaceDetectorYN.create(str(model), "", (320, 320), 0.7)

        def yunet(img):
            det.setInputSize((img.shape[1], img.shape[0]))
            _, faces = det.detect(img)
            return [] if faces is None else [tuple(f[:4]) for f in faces]
        return yunet
    except Exception:
        pass
    xml = Path(getattr(getattr(cv2, "data", None), "haarcascades", "")) / "haarcascade_frontalface_default.xml"
    if xml.exists():
        casc = cv2.CascadeClassifier(str(xml))
        return lambda img: list(casc.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.15, 5, minSize=(28, 28)))
    return None


def face_track(path, start: float, end: float, src_w: int, src_h: int) -> np.ndarray | None:
    """Horizontal face centre (0..1) per second. None when there are no faces (then motion is followed)."""
    import cv2
    detect = _face_detector()
    if detect is None:
        return None
    cap = cv2.VideoCapture(str(path))
    n = max(1, int(np.ceil(end - start)))
    xs = np.full(n, np.nan, dtype=np.float32)
    for s in range(n):
        cap.set(cv2.CAP_PROP_POS_MSEC, (start + s + 0.5) * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        scale = 480 / max(fr.shape[1], 1)
        small = cv2.resize(fr, None, fx=scale, fy=scale)
        faces = detect(small)
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            xs[s] = (x + w / 2) / small.shape[1]
    cap.release()
    if np.isnan(xs).mean() > 0.6:
        return None
    idx = np.arange(n)
    good = ~np.isnan(xs)
    return np.interp(idx, idx[good], xs[good]).astype(np.float32)


def _smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    if x.size < 3:
        return x
    pad = np.pad(x, (k, k), mode="edge")
    sm = np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]
    # dead-zone: ignore tiny wobbles so the camera holds still like a human operator
    out = sm.copy()
    for i in range(1, out.size):
        if abs(out[i] - out[i - 1]) < 0.02:
            out[i] = out[i - 1]
    return out


def crop_x_expr(track: np.ndarray, crop_w: int, src_w: int) -> str:
    """Piecewise-linear ffmpeg expression for the crop's x position over clip time t."""
    xs = np.clip(track * src_w - crop_w / 2, 0, max(0, src_w - crop_w)).astype(int)
    if xs.size == 1 or np.ptp(xs) < 4:
        return str(int(np.median(xs)))
    parts = []
    for i in range(xs.size - 1):
        a, b = int(xs[i]), int(xs[i + 1])
        parts.append(f"between(t\\,{i}\\,{i + 1})*({a}+({b - a})*(t-{i}))" if a != b else f"between(t\\,{i}\\,{i + 1})*{a}")
    # between() is inclusive at both ends; tiny double-count at integer t is invisible
    expr = "+".join(parts)
    return f"if(gte(t\\,{xs.size - 1})\\,{int(xs[-1])}\\,{expr})"


def video_filter(framing: str, W: int, H: int, src_w: int, src_h: int, track: np.ndarray | None) -> str:
    target_r = W / H
    src_r = src_w / max(src_h, 1)
    up = f"scale={W}:{H}:flags=lanczos"
    if abs(src_r - target_r) < 0.02:
        return f"{up},unsharp=5:5:0.4"
    if framing == "fit":
        return (f"split=2[bg][fg];[bg]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
                f"boxblur=24:2,eq=brightness=-0.08[bgb];"
                f"[fg]scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos,unsharp=5:5:0.4[fgs];"
                f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2")
    # crop to the target ratio
    if src_r > target_r:  # source wider: crop width
        cw, ch = int(src_h * target_r) // 2 * 2, src_h
        x = crop_x_expr(_smooth(track), cw, src_w) if (framing == "smart" and track is not None) else str((src_w - cw) // 2)
        return f"crop={cw}:{ch}:{x}:0,{up},unsharp=5:5:0.4"
    cw, ch = src_w, int(src_w / target_r) // 2 * 2
    return f"crop={cw}:{ch}:0:{(src_h - ch) // 2},{up},unsharp=5:5:0.4"


# ---------- captions ----------

def _ass_time(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _esc(s: str) -> str:
    return s.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def write_ass(path: Path, words, start: float, end: float, W: int, H: int, title: str = "") -> bool:
    ws = [w for w in words if w.end > start and w.start < end]
    if not ws and not title:
        return False
    fs = int(W * 0.068)
    margin_v = int(H * (0.28 if H > W else 0.12))
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "WrapStyle: 0", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic,"
        " Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL,"
        " MarginR, MarginV, Encoding",
        f"Style: Cap,{FONT_NAME},{fs},&H00FFFFFF,&H001BD5F7,&H00111E15,&H64000000,-1,0,0,0,100,100,1,0,1,"
        f"{max(4, fs // 11)},2,2,60,60,{margin_v},1",
        f"Style: Hook,{FONT_NAME},{int(fs * 0.8)},&H00FFFFFF,&H00FFFFFF,&H001E30E0,&H001E30E0,-1,0,0,0,100,100,0,0,3,"
        f"{int(fs * 0.25)},0,8,80,80,{int(H * 0.09)},1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    if title:
        lines.append(f"Dialogue: 1,{_ass_time(0)},{_ass_time(min(3.2, end - start))},Hook,,0,0,0,,"
                     f"{{\\fad(120,250)}}{_esc(title.upper())}")
    # groups of up to 3 words; the word being spoken turns yellow and pops
    i = 0
    while i < len(ws):
        grp = [ws[i]]
        while (len(grp) < 3 and i + len(grp) < len(ws) and ws[i + len(grp)].start - grp[-1].end < 0.35
               and not grp[-1].text.endswith((".", "!", "?", ","))):
            grp.append(ws[i + len(grp)])
        nxt = ws[i + len(grp)].start if i + len(grp) < len(ws) else end
        for j, w in enumerate(grp):
            a = max(0.0, w.start - start)
            # never overlap the next group, or libass stacks two lines out of order
            b = (grp[j + 1].start if j + 1 < len(grp) else min(w.end + 0.25, nxt, end)) - start
            if b <= a:
                continue
            txt = " ".join(
                (r"{\c&H001BD5F7&\fscx112\fscy112}" + _esc(g.text.upper()) + r"{\r}") if k == j else _esc(g.text.upper())
                for k, g in enumerate(grp))
            lines.append(f"Dialogue: 0,{_ass_time(a)},{_ass_time(b)},Cap,,0,0,0,,{txt}")
        i += len(grp)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def _ff_path(p: Path) -> str:
    return str(p).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


# ---------- render ----------

def render(src, out: Path, start: float, end: float, *, W: int, H: int, src_w: int, src_h: int,
           framing: str, track, ass: Path | None, fps: float, has_audio: bool, quality: str = "fast",
           pre_crop: tuple[int, int, int, int] | None = None) -> Path:
    """src_w/src_h are the picture size after pre_crop (black bars removed) when one is given."""
    vf = video_filter(framing, W, H, src_w, src_h, track)
    if pre_crop:
        cw, ch, cx, cy = pre_crop
        vf = f"crop={cw}:{ch}:{cx}:{cy}," + vf
    if ass is not None:
        vf += f",subtitles='{_ff_path(ass)}':fontsdir='{_ff_path(ASSETS)}'"
    dur = end - start
    vf += f",fps={min(60, round(fps) or 30)},format=yuv420p"
    vf += f",fade=t=in:st=0:d=0.15,fade=t=out:st={max(0, dur - 0.25):.2f}:d=0.25"
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
           "-filter_complex", f"[0:v]{vf}[v]", "-map", "[v]"]
    if has_audio:
        cmd += ["-map", "0:a:0", "-af",
                f"loudnorm=I=-14:TP=-1.5:LRA=11,afade=t=in:d=0.1,afade=t=out:st={max(0, dur - 0.3):.2f}:d=0.3",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
    preset = {"best": "slow", "fast": "veryfast"}.get(quality, "veryfast")
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", "18" if quality == "best" else "20",
            "-profile:v", "high", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    return out


def thumbnail(video: Path, out: Path, at: float) -> Path:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{at:.2f}", "-i", str(video), "-frames:v", "1",
                    "-vf", "scale=540:-2", "-q:v", "3", str(out)], check=True)
    return out
