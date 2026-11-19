"""Streamer facecam detection + the split "Stream" layout (facecam on top, gameplay below).

A facecam is a face that is small relative to the frame and stays in the same place
for most of the video. A full-screen talking head (big face) or a moving person is not one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TOP_SHARE = 0.35          # facecam gets the top 35% of the vertical frame


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int
    overlay: tuple[int, int, int, int] | None = None   # the facecam rectangle itself (x, y, w, h), if measured


def detect(path, duration: float, frame_w: int, frame_h: int, samples: int = 24,
           offset: tuple[int, int] = (0, 0), require_corner: bool = True) -> Box | None:
    """Return the median facecam face box in picture coordinates (after removing `offset`), or None."""
    import cv2
    from .render import _face_detector
    find = _face_detector()
    if find is None or duration <= 0:
        return None
    cap = cv2.VideoCapture(str(path))
    boxes = []
    grays = []
    for i in range(samples):
        cap.set(cv2.CAP_PROP_POS_MSEC, duration * (i + 0.5) / samples * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        ox, oy = offset
        fr = fr[oy:oy + frame_h, ox:ox + frame_w]
        scale = 960 / max(fr.shape[1], 1)
        small = cv2.resize(fr, None, fx=scale, fy=scale)
        grays.append((cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), scale))
        faces = find(small)
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            boxes.append((x / scale, y / scale, w / scale, h / scale))
    cap.release()
    face = choose(boxes, samples, frame_w, frame_h, require_corner)
    if face and grays:
        face.overlay = find_overlay([g for g, _ in grays], grays[0][1], face, frame_w, frame_h)
    return face


def find_overlay(grays, scale: float, face: Box, frame_w: int, frame_h: int):
    """Measure the facecam rectangle. Its border sits still while gameplay edges move, so after
    averaging edge strength over many frames the border stands out as a sharp ridge."""
    import cv2
    acc = None
    for g in grays:
        gf = g.astype(np.float32)
        gx = np.abs(cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3))
        acc = (gx, gy) if acc is None else (acc[0] + gx, acc[1] + gy)
    ex, ey = acc[0] / len(grays), acc[1] / len(grays)
    H, W = ex.shape
    fx0, fy0 = int(face.x * scale), int(face.y * scale)
    fx1, fy1 = int((face.x + face.w) * scale), int((face.y + face.h) * scale)
    fw, fh = max(1, fx1 - fx0), max(1, fy1 - fy0)

    def ridge(profile, lo, hi, toward_end):
        lo, hi = max(0, lo), min(len(profile), hi)
        if hi - lo < 3:
            return None
        seg = profile[lo:hi]
        k = int(seg.argmax())
        base = float(np.median(profile)) + 1e-6
        return lo + k if seg[k] > 3.0 * base else None

    rows = slice(max(0, fy0), min(H, fy1))
    cols = slice(max(0, fx0), min(W, fx1))
    colp = ex[rows].mean(axis=0)
    rowp = ey[:, cols].mean(axis=1)
    left = ridge(colp, fx0 - 3 * fw, fx0 - 2, False)
    right = ridge(colp, fx1 + 2, fx1 + 3 * fw, True)
    top = ridge(rowp, fy0 - 2 * fh, fy0 - 2, False)
    bottom = ridge(rowp, fy1 + 2, fy1 + 2 * fh, True)
    # a frame edge counts as a border too (facecams are often flush with the screen edge)
    left = 0 if left is None and fx0 < 3 * fw else left
    right = W - 1 if right is None and W - fx1 < 3 * fw else right
    top = 0 if top is None and fy0 < 2 * fh else top
    bottom = H - 1 if bottom is None and H - fy1 < 2 * fh else bottom
    if None in (left, right, top, bottom):
        return None
    x, y = left / scale, top / scale
    w, h = (right - left) / scale, (bottom - top) / scale
    if w < face.w * 1.3 or h < face.h * 1.1 or w * h > 0.35 * frame_w * frame_h:
        return None
    return (int(x), int(y), int(w), int(h))


def choose(boxes, samples: int, frame_w: int, frame_h: int, require_corner: bool = True) -> Box | None:
    """Pure decision logic (unit-tested): is this set of per-frame faces a facecam?"""
    if len(boxes) < max(4, samples * 0.45):
        return None
    a = np.array(boxes, dtype=np.float32)
    cx = a[:, 0] + a[:, 2] / 2
    cy = a[:, 1] + a[:, 3] / 2
    # keep the dominant position (ignores the odd face that appears in gameplay)
    mx, my = np.median(cx), np.median(cy)
    near = (np.abs(cx - mx) < frame_w * 0.05) & (np.abs(cy - my) < frame_h * 0.06)
    if near.sum() < max(4, samples * 0.4):
        return None
    w = float(np.median(a[near, 2]))
    h = float(np.median(a[near, 3]))
    if w > frame_w * 0.2:          # big face = talking head / full-screen cam, not a facecam overlay
        return None
    if require_corner:
        # overlays live in a corner; a person filmed in a room sits near the middle
        side = mx < frame_w * 0.33 or mx > frame_w * 0.67
        edge = my < frame_h * 0.4 or my > frame_h * 0.6
        if not (side and edge):
            return None
    return Box(int(np.median(a[near, 0])), int(np.median(a[near, 1])), int(w), int(h))


def _even(v: float) -> int:
    """Even size (codecs need even dimensions), at least 2."""
    return max(2, int(v) // 2 * 2)


def _pos(v: float) -> int:
    """Even offset, may be 0."""
    return max(0, int(v) // 2 * 2)


def layout(face: Box, src_w: int, src_h: int, W: int, H: int) -> tuple[tuple, tuple, int]:
    """Crop rectangles (w, h, x, y) for the facecam and gameplay panels, and the top panel height."""
    top_h = _even(H * TOP_SHARE)
    bot_h = H - top_h
    cam_r = W / top_h
    fx, fy = face.x + face.w / 2, face.y + face.h / 2
    if face.overlay:
        ox, oy, ow, oh = face.overlay
        inset = max(4, ow * 0.02)                      # drop the overlay's own border
        ox, oy, ow, oh = ox + inset, oy + inset, ow - 2 * inset, oh - 2 * inset
        # largest panel-shaped crop inside the overlay, centred on the face
        cw, ch = (ow, ow / cam_r) if ow / oh < cam_r else (oh * cam_r, oh)
        cx = min(max(ox, fx - cw / 2), ox + ow - cw)
        cy = min(max(oy, fy - ch * 0.45), oy + oh - ch)
        cam_l, cam_r_edge = face.overlay[0], face.overlay[0] + face.overlay[2]
    else:
        # estimate: the face fills ~30% of the panel width, sitting slightly above centre
        cw = min(src_w, max(face.w * 3.3, 160))
        ch = cw / cam_r
        if ch > src_h:
            ch, cw = src_h, src_h * cam_r
        cx = min(max(0, fx - cw / 2), src_w - cw)
        cy = min(max(0, fy - ch * 0.45), src_h - ch)
        cam_l, cam_r_edge = cx - face.w * 0.4, cx + cw + face.w * 0.4   # pad: the estimate is rough
    cam = (_even(cw), _even(ch), _pos(cx), _pos(cy))
    # gameplay panel: full height, centred, the facecam corner usually falls outside it
    g_r = W / bot_h
    gh = src_h
    gw = min(src_w, gh * g_r)
    if gw >= src_w:
        gw, gh = src_w, src_w / g_r
    gx = (src_w - gw) / 2
    # keep the facecam overlay out of the gameplay panel: slide the crop away from it if there's room
    if cam_r_edge > gx and cam_l < gx + gw:
        if fx < src_w / 2:
            gx = min(max(gx, cam_r_edge + 4), src_w - gw)
        else:
            gx = max(min(gx, cam_l - 4 - gw), 0)
    game = (_even(gw), _even(gh), _pos(gx), _pos((src_h - gh) / 2))
    return cam, game, top_h


def video_filter(cam: tuple, game: tuple, top_h: int, W: int, H: int) -> str:
    cw, ch, cx, cy = cam
    gw, gh, gx, gy = game
    bot_h = H - top_h
    return (f"split=2[c][g];"
            f"[c]crop={cw}:{ch}:{cx}:{cy},scale={W}:{top_h}:flags=lanczos,unsharp=5:5:0.5[top];"
            f"[g]crop={gw}:{gh}:{gx}:{gy},scale={W}:{bot_h}:flags=lanczos,unsharp=5:5:0.4[bot];"
            f"[top][bot]vstack=inputs=2,drawbox=x=0:y={top_h - 4}:w={W}:h=8:color=0xF7D51B:t=fill")
