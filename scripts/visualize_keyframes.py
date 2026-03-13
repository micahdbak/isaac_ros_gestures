#!/usr/bin/env python3

"""Visualize recorded keyframes from a per-video CSV.

The video_collector_node writes CSV files with:
- 200 rows (max_frames)
- 42 columns (21 keypoints * (x,y))
- trailing all-zero rows are padding

This script loads a CSV, drops the padded rows, and animates the keypoints.

Usage:
  python3 scripts/visualize_keyframes.py /path/to/video.csv

Optional:
    --fps 30           Animation playback FPS
    --connect          Draw a simple hand skeleton
    --frames 3         Force first N frames (useful if data is all zeros)
    --save out.mp4     Save animation instead of showing (requires ffmpeg)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# A simple MediaPipe-style hand skeleton over 21 points.
_EDGES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def _load_csv(path: Path) -> np.ndarray:
    # genfromtxt is more forgiving than loadtxt for odd whitespace.
    data = np.genfromtxt(str(path), delimiter=',', dtype=np.float32)
    if data.ndim == 0:
        data = data.reshape(1, 1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def _trim_trailing_padding(rows: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Trim *trailing* all-zero padding rows.

    This matches the collector's behavior (padding added at the end). We avoid
    removing all-zero rows in the middle, since those could be valid data.
    """

    if rows.size == 0:
        return rows

    nonzero = np.any(np.abs(rows) > eps, axis=1)
    if not np.any(nonzero):
        # Cannot infer padding if everything is zero.
        return rows
    last = int(np.where(nonzero)[0][-1])
    return rows[: last + 1]


def _normalize_rows(rows: np.ndarray, expected_kpts: int = 21, max_frames: int = 200) -> np.ndarray:
    """Normalize input into a 2D array shaped (T, expected_kpts*2)."""

    row_len = expected_kpts * 2

    if rows.ndim != 2:
        raise ValueError(f'Expected 2D array after CSV load, got ndim={rows.ndim}')

    # Common cases:
    # - (200, 42): per-frame rows
    # - (T, 42): per-frame rows with variable length
    # - (1, 8400): flattened old format (max_frames*42)
    if rows.shape[1] == row_len:
        return rows

    if rows.shape[0] == 1 and rows.shape[1] == max_frames * row_len:
        return rows.reshape(max_frames, row_len)

    if rows.shape[0] == 1 and rows.shape[1] % row_len == 0:
        t = rows.shape[1] // row_len
        return rows.reshape(t, row_len)

    raise ValueError(f'Unexpected CSV shape {rows.shape}; expected (*,{row_len}) or (1,{max_frames*row_len})')


def _reshape_to_xy(rows: np.ndarray, expected_kpts: int = 21) -> np.ndarray:
    if rows.shape[1] != expected_kpts * 2:
        raise ValueError(f'Expected {expected_kpts * 2} columns, got {rows.shape[1]}')
    xy = rows.reshape(rows.shape[0], expected_kpts, 2)
    return xy


def main() -> int:
    parser = argparse.ArgumentParser(description='Animate keyframes from a collector CSV')
    parser.add_argument('csv_path', type=str, help='Path to a per-video CSV file')
    parser.add_argument('--fps', type=float, default=30.0, help='Playback FPS (default: 30)')
    parser.add_argument('--connect', action='store_true', help='Draw a simple skeleton between keypoints')
    parser.add_argument('--frames', type=int, default=0, help='Force first N frames (0 = auto)')
    parser.add_argument('--save', type=str, default='', help='Optional output path (mp4/gif) to save instead of showing')
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    raw = _load_csv(csv_path)
    rows = _normalize_rows(raw, expected_kpts=21, max_frames=200)

    if args.frames and args.frames > 0:
        rows = rows[: args.frames]
    else:
        rows = _trim_trailing_padding(rows)

    if rows.shape[0] == 0:
        print('No frames found after trimming.')
        return 0

    # Helpful diagnostics
    max_abs = float(np.nanmax(np.abs(rows))) if rows.size else 0.0
    print(f'Loaded frames={rows.shape[0]} cols={rows.shape[1]} max_abs={max_abs:.6f}')

    xy = _reshape_to_xy(rows, expected_kpts=21)  # (T, 21, 2)

    # Set up plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f'{csv_path.name}  (frames={xy.shape[0]})')

    # Coordinates are normalized to [-1, 1] in this pipeline.
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)

    scat = ax.scatter([], [], s=35)
    lines = []
    if args.connect:
        for _ in _EDGES:
            (ln,) = ax.plot([], [], linewidth=2)
            lines.append(ln)

    text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top')

    def init():
        scat.set_offsets(np.zeros((21, 2), dtype=np.float32))
        if args.connect:
            for ln in lines:
                ln.set_data([], [])
        text.set_text('')
        return [scat, text, *lines]

    def update(i: int):
        pts = xy[i]
        scat.set_offsets(pts)
        if args.connect:
            for ln, (a, b) in zip(lines, _EDGES):
                ln.set_data([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]])
        text.set_text(f'frame {i + 1}/{xy.shape[0]}')
        return [scat, text, *lines]

    interval_ms = 1000.0 / max(1e-6, float(args.fps))
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=xy.shape[0],
        interval=interval_ms,
        blit=True,
        repeat=True,
    )

    if args.save:
        out = Path(args.save)
        suffix = out.suffix.lower()
        if suffix == '.gif':
            anim.save(str(out), writer='pillow', fps=float(args.fps))
        else:
            # default to ffmpeg writer for mp4
            anim.save(str(out), writer='ffmpeg', fps=float(args.fps))
        print(f'Saved animation to {out}')
    else:
        plt.show()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
