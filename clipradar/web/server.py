"""Web app: FastAPI backend + a single-page, mobile-first UI (installable as a PWA).

Run `clipradar serve`, then open the printed address on any phone, tablet or computer on
the same network (or deploy the Docker image anywhere).
"""
from __future__ import annotations

import io
import os
import queue
import re
import shutil
import socket
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..config import MAX_CLIPS, Options

STATIC = Path(__file__).parent / "static"


class JobIn(BaseModel):
    source: str
    clips: int = Field(5, ge=1, le=MAX_CLIPS)
    content: str = "auto"
    layout: str = "vertical"
    framing: str = "auto"
    captions: bool = True
    hook_title: bool = True
    keywords: list[str] = []
    min_len: float = 0
    max_len: float = 0
    quality: str = "fast"


@dataclass
class Job:
    id: str
    opts: Options
    status: str = "queued"           # queued | running | done | error
    stage: str = "queued"
    progress: float = 0.0
    message: str = "Waiting in line…"
    result: dict | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    log: list[str] = field(default_factory=list)


def lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def create_app(out_dir: str = "clipradar_output") -> FastAPI:
    out_root = Path(out_dir).expanduser().resolve()
    uploads = out_root / ".uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    jobs: dict[str, Job] = {}
    q: "queue.Queue[str]" = queue.Queue()
    max_upload = int(os.getenv("CLIPRADAR_MAX_UPLOAD_MB", "4096")) * 1024 * 1024

    def worker():
        from ..pipeline import run
        while True:
            jid = q.get()
            job = jobs[jid]
            job.status = "running"

            def prog(stage, frac, msg):
                job.stage, job.progress, job.message = stage, round(frac, 3), msg

            try:
                job.result = run(job.opts, progress=prog, log=lambda m: job.log.append(m))
                job.status, job.progress = "done", 1.0
            except Exception as e:  # report, keep the worker alive
                job.status, job.error, job.message = "error", str(e)[:500], "Something went wrong."
            finally:
                if job.opts.source.startswith(str(uploads)):
                    shutil.rmtree(Path(job.opts.source).parent, ignore_errors=True)
                q.task_done()

    threading.Thread(target=worker, daemon=True).start()

    app = FastAPI(title="ClipRadar", version=__version__)

    def _new(opts: Options) -> Job:
        opts.out_dir = str(out_root)
        jid = uuid.uuid4().hex[:10]
        jobs[jid] = Job(jid, opts.normalised())
        q.put(jid)
        return jobs[jid]

    def _view(job: Job) -> dict:
        ahead = sum(1 for j in jobs.values() if j.status == "queued" and j.created < job.created)
        d = {"id": job.id, "status": job.status, "stage": job.stage, "progress": job.progress,
             "message": job.message if job.status != "queued" or not ahead else f"Waiting: {ahead} job(s) ahead",
             "error": job.error}
        if job.result:
            folder = Path(job.result["dir"]).name
            r = {k: v for k, v in job.result.items() if k not in ("options", "dir")}
            for c in r["clips"]:
                c["url"] = f"/files/{folder}/{c['file']}"
                c["thumb_url"] = f"/files/{folder}/{c['thumb']}"
            r["zip"] = f"/api/jobs/{job.id}/zip"
            d["result"] = r
        return d

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/manifest.webmanifest")
    def manifest():
        return FileResponse(STATIC / "manifest.webmanifest", media_type="application/manifest+json")

    @app.get("/sw.js")
    def sw():
        return FileResponse(STATIC / "sw.js", media_type="text/javascript")

    @app.get("/icon.svg")
    def icon():
        return FileResponse(STATIC / "icon.svg", media_type="image/svg+xml")

    @app.get("/font.otf")
    def font():
        return FileResponse(Path(__file__).parent.parent / "assets" / "Inter-Black.otf", media_type="font/otf")

    @app.get("/api/info")
    def info():
        port = os.getenv("CLIPRADAR_PORT", "8765")
        url = f"http://{lan_ip()}:{port}"
        qr = None
        try:
            import qrcode
            import qrcode.image.svg
            buf = io.BytesIO()
            qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage, box_size=8).save(buf)
            qr = buf.getvalue().decode()
        except Exception:
            pass
        return {"version": __version__, "lan_url": url, "qr_svg": qr, "max_clips": MAX_CLIPS}

    @app.post("/api/jobs")
    def create(body: JobIn):
        if not re.match(r"^https?://", body.source.strip()):
            raise HTTPException(400, "Paste a full video link starting with http(s)://, or upload a file.")
        try:
            job = _new(Options(**body.model_dump()))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return _view(job)

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...), clips: int = Form(5), content: str = Form("auto"),
                     layout: str = Form("vertical"), framing: str = Form("auto"), captions: bool = Form(True),
                     keywords: str = Form("")):
        orig = Path(file.filename or "video.mp4").name
        safe = re.sub(r"[^A-Za-z0-9._ -]+", "_", orig)[:80] or "video.mp4"
        holder = uploads / uuid.uuid4().hex
        holder.mkdir()
        dest = holder / safe
        size = 0
        with dest.open("wb") as f:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > max_upload:
                    f.close()
                    shutil.rmtree(holder, ignore_errors=True)
                    raise HTTPException(413, "File too large.")
                f.write(chunk)
        try:
            job = _new(Options(source=str(dest), clips=clips, content=content, layout=layout, framing=framing,
                               captions=captions, keywords=[k for k in keywords.split(",") if k.strip()]))
        except ValueError as e:
            shutil.rmtree(holder, ignore_errors=True)
            raise HTTPException(400, str(e))
        return _view(job)

    @app.get("/api/jobs/{jid}")
    def status(jid: str):
        if jid not in jobs:
            raise HTTPException(404, "Unknown job")
        return _view(jobs[jid])

    @app.get("/api/jobs/{jid}/zip")
    def zip_all(jid: str):
        job = jobs.get(jid)
        if not job or not job.result:
            raise HTTPException(404, "Not ready")
        folder = Path(job.result["dir"])
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
            for p in sorted(folder.iterdir()):
                if p.suffix in (".mp4", ".srt", ".json", ".jpg"):
                    z.write(p, p.name)
        buf.seek(0)
        return StreamingResponse(buf, media_type="application/zip",
                                 headers={"Content-Disposition": f'attachment; filename="{folder.name}.zip"'})

    @app.get("/files/{folder}/{name}")
    def files(folder: str, name: str, dl: int = 0):
        p = (out_root / folder / name).resolve()
        if out_root not in p.parents or not p.is_file():
            raise HTTPException(404, "Not found")
        return FileResponse(p, filename=name if dl else None)

    @app.get("/healthz")
    def health():
        return JSONResponse({"ok": True, "ffmpeg": bool(shutil.which("ffmpeg"))})

    return app


def start(host: str = "0.0.0.0", port: int = 8765, out_dir: str = "clipradar_output"):
    import uvicorn
    os.environ["CLIPRADAR_PORT"] = str(port)
    print(f"\n  ClipRadar is running:\n    this computer:  http://localhost:{port}\n"
          f"    phone/tablet:   http://{lan_ip()}:{port}  (same Wi-Fi)\n")
    uvicorn.run(create_app(out_dir), host=host, port=port, log_level="warning")
