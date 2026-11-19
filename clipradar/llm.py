"""Optional LLM re-rank + titles via any OpenAI-compatible endpoint.

Works with OpenAI, OpenRouter, Groq, Together, Gemini (OpenAI-compatible API), and local
Ollama / LM Studio. Configure with env vars:

  CLIPRADAR_LLM_BASE_URL   e.g. http://localhost:11434/v1   (Ollama)
  CLIPRADAR_LLM_MODEL      e.g. llama3.1
  CLIPRADAR_LLM_API_KEY    (blank for local servers)

Without these, ClipRadar uses its signal scores and makes titles from the transcript.
"""
from __future__ import annotations

import json
import re
import urllib.request

PROMPT = """You pick short-form clips (TikTok/Reels/Shorts). Video title: "{title}".
Below are candidate clips with transcripts. For each, rate 0-10 how likely it is to perform as a
standalone short (strong hook in first 3s, payoff, emotion, no missing context), and write a
punchy on-screen title of at most 6 words (no hashtags, no emojis).
Reply with JSON only: {{"clips":[{{"id":0,"score":7.5,"title":"..."}}]}}

{items}"""


def _chat(cfg: dict, prompt: str, timeout: float = 90) -> str:
    body = json.dumps({"model": cfg["model"], "temperature": 0.2,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(cfg["base_url"] + "/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {cfg['api_key']}"} if cfg["api_key"] else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def rerank(cfg: dict, title: str, clips, texts: list[str], log=print) -> bool:
    items = "\n".join(f"[{i}] {c.start:.0f}s-{c.end:.0f}s: {t[:700] or '(no speech: gameplay/music)'}"
                      for i, (c, t) in enumerate(zip(clips, texts)))
    try:
        raw = _chat(cfg, PROMPT.format(title=title[:120], items=items))
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0) if m else raw)
    except Exception as e:
        log(f"LLM re-rank skipped ({e}).")
        return False
    for row in data.get("clips", []):
        i = int(row.get("id", -1))
        if 0 <= i < len(clips):
            llm = float(row.get("score", 5)) * 10
            clips[i].score = round(0.55 * clips[i].score + 0.45 * llm, 1)
            if row.get("title"):
                clips[i].title = str(row["title"])[:60]
    log(f"LLM re-ranked {len(clips)} candidates with {cfg['model']}.")
    return True
