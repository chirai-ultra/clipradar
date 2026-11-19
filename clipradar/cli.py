"""Terminal interface.

  clipradar "https://youtube.com/watch?v=..." -n 5
  clipradar gameplay.mp4 -n 10 --content gaming --layout vertical
  clipradar serve            # web app for phone / tablet / desktop
"""
from __future__ import annotations

import sys
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import MAX_CLIPS, Options

app = typer.Typer(add_completion=False, help="ClipRadar: paste a link, get the best moments as HD shorts.")
con = Console()


@app.command("clip")
def clip(
    source: str = typer.Argument(..., help="Video URL (YouTube, Twitch VOD, Kick, X, TikTok, 1000+ sites) or a local file"),
    clips: int = typer.Option(5, "-n", "--clips", min=1, max=MAX_CLIPS, help="How many clips (1-10)"),
    content: str = typer.Option("auto", help="auto | gaming | talk | sports"),
    layout: str = typer.Option("vertical", help="vertical 1080x1920 | landscape | square | auto"),
    framing: str = typer.Option("auto", help="auto | stream (facecam on top, gameplay below) | smart (follow faces/action) | fit (blurred background) | center"),
    min_len: float = typer.Option(0, help="Minimum clip seconds (0 = preset)"),
    max_len: float = typer.Option(0, help="Maximum clip seconds (0 = preset)"),
    keyword: Optional[List[str]] = typer.Option(None, "-k", "--keyword", help="Favour moments mentioning this (repeatable)"),
    captions: bool = typer.Option(True, help="Burn animated captions"),
    hook_title: bool = typer.Option(True, help="Show a hook title for the first 3 s"),
    trends: bool = typer.Option(True, help="Use live search trends for the topic"),
    asr: str = typer.Option("auto", help="auto | faster-whisper | sherpa | none"),
    asr_model: str = typer.Option("small", help="faster-whisper model size (tiny/base/small/medium/large-v3)"),
    language: Optional[str] = typer.Option(None, help="Speech language code, e.g. en, hi (auto if empty)"),
    llm: bool = typer.Option(True, help="Re-rank with an LLM if CLIPRADAR_LLM_* is set"),
    quality: str = typer.Option("fast", help="fast | best"),
    out: str = typer.Option("clipradar_output", "-o", "--out", help="Output folder"),
    cookies: Optional[str] = typer.Option(None, help="cookies.txt for yt-dlp"),
):
    """Find the best moments in a video and export them as HD clips."""
    opts = Options(source=source, clips=clips, content=content, layout=layout, framing=framing,
                   min_len=min_len, max_len=max_len, keywords=keyword or [], captions=captions,
                   hook_title=hook_title, trends=trends, asr=asr, asr_model=asr_model, language=language,
                   llm=llm, quality=quality, out_dir=out, cookies=cookies)
    from .pipeline import run
    try:
        res = run(opts, log=lambda m: con.print(f"[bold #E0301E]>[/] {m}"))
    except Exception as e:
        con.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1)
    t = Table(title=f"{res['title'] or 'clips'}  ({res['content']}, {res['layout']})", header_style="bold #3E6DB0")
    for col in ("#", "time", "len", "score", "title / why", "file"):
        t.add_column(col)
    for c in res["clips"]:
        t.add_row(str(c["rank"]), f"{c['start']:.0f}-{c['end']:.0f}s", f"{c['duration']:.0f}s", f"{c['score']:.0f}",
                  (c["title"] + "\n" if c["title"] else "") + f"[dim]{c['why']}[/]", c["file"])
    con.print(t)
    con.print(f"[bold #3B6E2C]Saved to[/] {res['dir']}")


@app.command("serve")
def serve(host: str = typer.Option("0.0.0.0", help="0.0.0.0 lets phones on your Wi-Fi connect"),
          port: int = typer.Option(8765), out: str = typer.Option("clipradar_output", "-o", "--out")):
    """Start the web app (works on any device with a browser)."""
    from .web.server import start
    start(host, port, out)


@app.command("doctor")
def doctor():
    """Check that ffmpeg and optional extras are installed."""
    import shutil
    from .transcribe import available_backends
    from .config import llm_settings
    ok = lambda b: "[green]yes[/]" if b else "[red]no[/]"
    con.print(f"ClipRadar {__version__}  Python {sys.version.split()[0]}")
    con.print(f"ffmpeg: {ok(shutil.which('ffmpeg'))}   ffprobe: {ok(shutil.which('ffprobe'))}")
    b = available_backends()
    con.print(f"speech backends: {', '.join(b) or '[yellow]none (pip install clipradar[asr])[/]'}")
    cfg = llm_settings()
    con.print(f"LLM re-rank: {cfg['model'] + ' @ ' + cfg['base_url'] if cfg else '[dim]off (optional)[/]'}")


def main():
    # `clipradar <url>` without a sub-command means `clipradar clip <url>`
    cmds = {"clip", "serve", "doctor", "--help", "-h", "--version"}
    if len(sys.argv) > 1 and sys.argv[1] not in cmds:
        sys.argv.insert(1, "clip")
    if len(sys.argv) > 1 and sys.argv[1] == "--version":
        print(__version__)
        return
    app()


if __name__ == "__main__":
    main()
