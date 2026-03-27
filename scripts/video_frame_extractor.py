#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path

import cv2


# Edit these variables (no CLI arguments are used)
VIDEO_PATH = '/home/robot/ros_video_20260313_165625.mp4'

# Leave empty to default to: <video_dir>/<video_stem>_frames
OUT_DIR = './video_data_1'

# Must include an integer placeholder like %06d
FILENAME_PATTERN = 'frame_%d.jpg'

# Extraction controls
START_FRAME = 0  # 0-based, inclusive
END_FRAME = -1  # -1 = until end, else 0-based inclusive index
STRIDE = 1  # save every Nth frame

# Output quality
JPEG_QUALITY = 95  # 0..100

# Progress logging
PRINT_EVERY = 200  # frames read


def extract_frames(
	video_path: Path,
	out_dir: Path,
	pattern: str,
	start: int,
	end: int,
	stride: int,
	quality: int,
	print_every: int = 0,
) -> int:
	if stride < 1:
		raise ValueError('--stride must be >= 1')
	if start < 0:
		raise ValueError('--start must be >= 0')
	if end != -1 and end < start:
		raise ValueError('--end must be -1 or >= --start')
	if not (0 <= quality <= 100):
		raise ValueError('--quality must be in [0, 100]')
	if '%d' not in pattern:
		raise ValueError("--pattern must include an integer placeholder like 'frame_%06d.jpg'")

	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise RuntimeError(f'Failed to open video: {video_path}')

	out_dir.mkdir(parents=True, exist_ok=True)
	encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]

	saved = 0
	frame_idx = 0
	try:
		while True:
			ok, frame = cap.read()
			if not ok:
				break

			if print_every and frame_idx % int(print_every) == 0:
				print(f'Read frame {frame_idx}... saved {saved}')

			if frame_idx >= start and (end == -1 or frame_idx <= end):
				if (frame_idx - start) % stride == 0:
					filename = pattern % frame_idx
					out_path = out_dir / filename
					ok_write = cv2.imwrite(str(out_path), frame, encode_params)
					if not ok_write:
						raise RuntimeError(f'Failed to write image: {out_path}')
					saved += 1
			elif end != -1 and frame_idx > end:
				break

			frame_idx += 1
	finally:
		cap.release()

	return saved


def main() -> None:
	video_path = Path(os.path.expanduser(VIDEO_PATH)).resolve()
	if not video_path.exists():
		raise SystemExit(
			"Video not found. Edit VIDEO_PATH at the top of this file. "
			f"Current VIDEO_PATH resolves to: {video_path}"
		)

	if OUT_DIR:
		out_dir = Path(os.path.expanduser(OUT_DIR)).resolve()
	else:
		out_dir = video_path.parent / f'{video_path.stem}_frames'

	saved = extract_frames(
		video_path=video_path,
		out_dir=out_dir,
		pattern=FILENAME_PATTERN,
		start=int(START_FRAME),
		end=int(END_FRAME),
		stride=int(STRIDE),
		quality=int(JPEG_QUALITY),
		print_every=int(PRINT_EVERY),
	)
	print(f'Saved {saved} frame(s) to: {out_dir}')


if __name__ == '__main__':
	main()
