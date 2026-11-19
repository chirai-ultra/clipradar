---
name: clipradar
description: Find the best moments in a long video (YouTube/Twitch/Kick/X/TikTok link or local file) and export up to 10 HD vertical clips with captions. Use when asked to clip, make shorts/reels/TikToks from, or pull highlights out of a video.
---

# ClipRadar skill

ClipRadar scores every second of a video by fusing YouTube "Most Replayed" data, audio excitement,
motion, scene cuts, speech keywords and live search trends, then renders the top moments in HD.

## Setup (once)

```bash
pip install "clipradar[asr] @ git+https://github.com/chirai-ultra/clipradar"   # needs ffmpeg on PATH
clipradar doctor                                                       # checks ffmpeg + speech backend
```

## Run

```bash
clipradar "<url or file>" -n <1-10> [options] -o clipradar_output
```

Pick options from what the user asked for:

| User wants | Flag |
|---|---|
| number of clips | `-n 5` (max 10) |
| gameplay / podcast / sports | `--content gaming\|talk\|sports` (default auto) |
| TikTok/Reels/Shorts | `--layout vertical` (default) |
| YouTube / X landscape | `--layout landscape` |
| keep the whole game screen | `--framing fit` |
| follow the speaker | `--framing smart` |
| specific moments ("the goal", "when he laughs") | `-k goal -k laugh` |
| exact lengths | `--min-len 20 --max-len 45` |
| no captions | `--no-captions` |
| non-English speech | `--language hi` (faster-whisper) |
| sharper/smaller files | `--quality best` |

## Read the result

The command prints a table and writes `<out>/<video>-<time>/`:
- `clipNN_<title>.mp4`: 1080x1920 H.264, -14 LUFS audio
- `clipNN_<title>.jpg` thumbnail, `.srt` captions
- `clips.json`: start/end, score (0-100), title, `why` (which signals fired), transcript

Report each clip's time range, score and `why` to the user, then send the mp4 files.

## Notes

- Links download with yt-dlp; for age/region-locked videos pass `--cookies cookies.txt`.
- No speech model available? Use `--asr none`; scoring still works from audio/visual/heatmap.
- Optional LLM re-rank + titles: set `CLIPRADAR_LLM_BASE_URL`, `CLIPRADAR_LLM_MODEL`, `CLIPRADAR_LLM_API_KEY`.
- Only clip videos the user owns or has permission (e.g. a clipping campaign) to use.
