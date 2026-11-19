"""End-to-end on a synthetic 'gameplay' video: quiet stretches with three loud, busy bursts."""
import json
import subprocess
from pathlib import Path

import pytest

from clipradar import Options, run

BURSTS = (40, 95, 150)


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    d = tmp_path_factory.mktemp("vid")
    out = d / "game.mp4"
    vol = "+".join(f"between(t,{b},{b + 6})*0.9" for b in BURSTS)
    vis = "+".join(f"between(t,{b},{b + 6})" for b in BURSTS)
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=0x3B6E2C:size=1280x720:rate=24:duration=180",
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=24:duration=180",
        "-f", "lavfi", "-i", "anoisesrc=d=180:c=pink:a=0.5",
        "-filter_complex",
        f"[0:v][1:v]overlay=enable='{vis}'[v];"
        f"[2:a]volume='0.03+{vol}':eval=frame[a]",
        "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(out)],
        check=True)
    return out


def test_pipeline_finds_the_bursts(video, tmp_path):
    res = run(Options(source=str(video), clips=3, asr="none", trends=False, llm=False,
                      content="gaming", out_dir=str(tmp_path)), log=lambda m: None)
    assert len(res["clips"]) == 3
    hits = sum(any(c["start"] <= b + 3 <= c["end"] for c in res["clips"]) for b in BURSTS)
    assert hits >= 2, res["clips"]
    folder = Path(res["dir"])
    manifest = json.loads((folder / "clips.json").read_text())
    for c in manifest["clips"]:
        mp4 = folder / c["file"]
        assert mp4.exists() and (folder / c["thumb"]).exists()
        info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(mp4)],
                                         capture_output=True, text=True).stdout)
        v = next(s for s in info["streams"] if s["codec_type"] == "video")
        assert (v["width"], v["height"]) == (1080, 1920)


def test_web_api(video, tmp_path):
    from fastapi.testclient import TestClient
    from clipradar.web.server import create_app
    c = TestClient(create_app(str(tmp_path)))
    assert c.get("/").status_code == 200
    assert c.post("/api/jobs", json={"source": "not a link"}).status_code == 400
    assert c.post("/api/jobs", json={"source": "https://x.com/a", "clips": 11}).status_code == 422
    assert c.get("/files/../../etc/passwd").status_code == 404
