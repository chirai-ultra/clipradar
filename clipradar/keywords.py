"""Keyword + trend signal.

Builds a weighted vocabulary from:
  * the video's own title, tags, chapter titles and description (what the video is about)
  * keywords the user typed (what they want)
  * live search-autocomplete trends for the topic (what people are searching right now)
  * a built-in hype lexicon (gaming, sports, reactions) that marks peak moments in speech
Then scores each second of the transcript against it.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from collections import Counter

import numpy as np

STOP = set("""a an the and or but if then so to of in on at by for with from up down out over under is are was were be
been being am i you he she it we they me him her us them my your his its our their this that these those what which who
whom whose when where why how all any both each few more most other some such no nor not only own same than too very can
will just don should now get got go going im youre dont thats its oh yeah like know really one two video official new
watch full part episode ep vs ft feat lyrics hd 4k 1080p subscribe channel http https www com""".split())

HYPE = {
    # reactions
    "wow": 1.0, "omg": 1.0, "insane": 1.0, "crazy": 0.9, "unbelievable": 1.0, "no way": 1.0, "what": 0.3,
    "oh my god": 1.0, "holy": 0.8, "let's go": 1.0, "lets go": 1.0, "yes": 0.4, "clutch": 1.0, "wait": 0.5,
    "bro": 0.4, "dude": 0.4, "amazing": 0.8, "incredible": 0.9, "huge": 0.6, "massive": 0.6, "epic": 0.8,
    "funny": 0.6, "laugh": 0.6, "hilarious": 0.9, "shocking": 0.9, "secret": 0.7, "never": 0.4, "best": 0.5,
    "worst": 0.6, "biggest": 0.6, "first time": 0.6, "finally": 0.6, "important": 0.5, "the truth": 0.8,
    # gaming
    "headshot": 1.0, "ace": 0.9, "kill": 0.6, "killed": 0.6, "win": 0.6, "won": 0.6, "victory": 0.8,
    "boss": 0.6, "one shot": 0.9, "triple": 0.8, "quad": 0.9, "penta": 1.0, "squad wipe": 1.0, "noob": 0.5,
    "gg": 0.7, "rage": 0.7, "glitch": 0.7, "speedrun": 0.6, "world record": 1.0, "chicken dinner": 1.0,
    # sports
    "goal": 1.0, "score": 0.6, "touchdown": 1.0, "dunk": 0.9, "save": 0.6, "six": 0.6, "wicket": 0.9,
    "knockout": 1.0, "penalty": 0.7, "buzzer": 0.9, "overtime": 0.6,
    # value / storytelling hooks
    "how to": 0.5, "tip": 0.5, "mistake": 0.6, "money": 0.5, "million": 0.6, "story": 0.4, "problem": 0.4,
}


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9']+", (text or "").lower()) if len(t) > 2 and t not in STOP]


def topic_terms(title: str, tags: list[str], description: str, chapters: list[dict]) -> Counter:
    c: Counter = Counter()
    for t in tokens(title):
        c[t] += 3
    for tag in tags[:30]:
        for t in tokens(tag):
            c[t] += 2
    for ch in chapters:
        for t in tokens(ch.get("title", "")):
            c[t] += 1.5
    for t in tokens(description[:1500]):
        c[t] += 0.3
    return c


def trend_terms(query: str, timeout: float = 4.0) -> list[str]:
    """Google/YouTube autocomplete: a free, keyless proxy for what people search right now."""
    if not query.strip():
        return []
    url = ("https://suggestqueries.google.com/complete/search?client=firefox&ds=yt&q="
           + urllib.parse.quote(query[:80]))
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8", "ignore"))
        return [s for s in data[1]][:10]
    except Exception:
        return []


def build_vocab(meta_terms: Counter, user: list[str], trends: list[str]) -> dict[str, float]:
    vocab: dict[str, float] = {}
    top = meta_terms.most_common(25)
    peak = top[0][1] if top else 1
    for t, w in top:
        vocab[t] = 0.6 * w / peak
    for phrase in trends:
        for t in tokens(phrase):
            vocab[t] = max(vocab.get(t, 0), 0.7)
    for k in user:
        vocab[k.lower()] = 1.5
        for t in tokens(k):
            vocab[t] = max(vocab.get(t, 0), 1.2)
    for k, v in HYPE.items():
        vocab.setdefault(k, v)
    return vocab


def score_transcript(words, vocab: dict[str, float], n_sec: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (keyword score per second, speech density per second)."""
    kw = np.zeros(n_sec, dtype=np.float32)
    dens = np.zeros(n_sec, dtype=np.float32)
    if not words:
        return kw, dens
    norm = [re.sub(r"[^a-z0-9' ]", "", w.text.lower()) for w in words]
    phrases = {k: v for k, v in vocab.items() if " " in k}
    for i, w in enumerate(words):
        s = int(w.start)
        if s >= n_sec:
            continue
        dens[s] += 1
        v = vocab.get(norm[i], 0.0)
        if i + 1 < len(words):
            bi = norm[i] + " " + norm[i + 1]
            v = max(v, phrases.get(bi, 0.0))
        if i + 2 < len(words):
            tri = norm[i] + " " + norm[i + 1] + " " + norm[i + 2]
            v = max(v, phrases.get(tri, 0.0))
        if words[i].text.endswith(("!", "?")):
            v += 0.3
        kw[s] += v
    kw = np.convolve(kw, np.ones(3) / 3, mode="same")
    hi = np.percentile(kw, 99) if kw.any() else 1
    kw = np.clip(kw / (hi or 1), 0, 1)
    dens = np.clip(dens / 4.0, 0, 1)
    return kw, dens
