import re
import subprocess

import numpy as np
import pytest

from clipradar import Options, keywords, render, score
from clipradar.transcribe import Word


def test_options_clamp_to_ten():
    assert Options(source="x", clips=50).normalised().clips == 10
    assert Options(source="x", clips=0).normalised().clips == 1
    with pytest.raises(ValueError):
        Options(source="x", layout="tall").normalised()


def test_pick_finds_peaks_without_overlap():
    curve = np.zeros(600, dtype=np.float32)
    for c in (60, 200, 350, 500):
        curve[c - 5:c + 5] = 1.0
    clips = score.pick(curve, 4, 15, 30, 600.0, np.array([]), np.array([]))
    assert len(clips) == 4
    clips.sort(key=lambda c: c.start)
    for a, b in zip(clips, clips[1:]):
        assert a.end <= b.start
    for c, peak in zip(clips, (60, 200, 350, 500)):
        assert c.start <= peak <= c.end
        assert 15 <= c.duration <= 33


def test_pick_terminates_when_video_too_short_for_request():
    # regression: used to loop forever when the request could not fit
    curve = np.random.default_rng(0).random(120).astype(np.float32)
    clips = score.pick(curve, 10, 25, 60, 120.0, np.array([]), np.array([]))
    assert 1 <= len(clips) <= 10


def test_heatmap_curve_highlights_replay_spike():
    hm = [{"start_time": i * 10, "end_time": (i + 1) * 10, "value": 0.1} for i in range(30)]
    hm[17]["value"] = 1.0
    h = score.heatmap_curve(hm, 300)
    assert h.argmax() // 10 == 17 and h.max() == pytest.approx(1.0)


def test_keyword_scoring_prefers_user_terms():
    vocab = keywords.build_vocab(keywords.topic_terms("Epic clutch round", ["valorant"], "", []), ["clutch"], [])
    words = [Word(1, 1.3, "nothing"), Word(5, 5.4, "clutch!"), Word(9, 9.3, "ok")]
    words[1].text = "clutch"
    kw, dens = keywords.score_transcript(words, vocab, 12)
    assert kw.argmax() in (4, 5, 6) and dens[1] > 0


def test_captions_never_overlap(tmp_path):
    words = [Word(i * 0.3, i * 0.3 + 0.28, w) for i, w in enumerate("so it is simple go to the site now".split())]
    p = tmp_path / "c.ass"
    assert render.write_ass(p, words, 0, 4, 1080, 1920, title="Hook")
    ev = []
    for line in p.read_text().splitlines():
        if line.startswith("Dialogue: 0,"):
            a, b = line.split(",")[1:3]
            ev.append((a, b))
    t = lambda s: sum(float(x) * m for x, m in zip(s.split(":"), (3600, 60, 1)))
    ev = sorted((t(a), t(b)) for a, b in ev)
    for (a1, b1), (a2, b2) in zip(ev, ev[1:]):
        assert b1 <= a2 + 1e-6


def test_crop_expression_is_valid_for_ffmpeg(tmp_path):
    track = np.linspace(0.2, 0.8, 6).astype(np.float32)
    expr = render.crop_x_expr(track, 608, 1920)
    out = tmp_path / "o.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=1920x1080:rate=10:duration=6",
                    "-vf", f"crop=608:1080:{expr}:0", "-c:v", "libx264", "-preset", "ultrafast", str(out)], check=True)
    assert out.stat().st_size > 1000


def test_out_size():
    assert render.out_size("vertical", 1920, 1080) == (1080, 1920)
    assert render.out_size("auto", 1080, 1920) == (1080, 1920)
    assert render.out_size("auto", 1920, 1080) == (1920, 1080)


def test_facecam_rules():
    from clipradar import facecam
    W, H = 1920, 1080
    corner = [(180 + i % 3, 840, 140, 180) for i in range(20)]
    assert facecam.choose(corner, 24, W, H) is not None                   # small, steady, in a corner
    centre = [(900, 400, 140, 180)] * 20
    assert facecam.choose(centre, 24, W, H) is None                       # person in the middle of a room
    assert facecam.choose(centre, 24, W, H, require_corner=False) is not None  # unless asked for explicitly
    big = [(700, 200, 500, 600)] * 20
    assert facecam.choose(big, 24, W, H) is None                          # full-screen talking head
    assert facecam.choose(corner[:5], 24, W, H) is None                   # face only now and then


def test_stream_layout_keeps_facecam_out_of_gameplay():
    from clipradar import facecam
    cam, game, top = facecam.layout(facecam.Box(180, 840, 140, 180), 1920, 1080, 1080, 1920)
    assert top == 672
    assert cam[0] / cam[1] == pytest.approx(1080 / 672, rel=0.02)
    assert game[2] >= cam[2] + cam[0]                                     # gameplay starts right of the cam
    assert game[0] / game[1] == pytest.approx(1080 / (1920 - 672), rel=0.02)
    for (w, h, x, y) in (cam, game):
        assert x >= 0 and y >= 0 and x + w <= 1920 and y + h <= 1080
