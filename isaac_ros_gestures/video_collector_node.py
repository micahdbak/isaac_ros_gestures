#!/usr/bin/env python3

"""Video Collector Node.

Iterates through a folder of .mp4 files, publishes frames as `sensor_msgs/Image`,
subscribes to `visualization_msgs/MarkerArray` pose markers, and writes a
fixed-length flattened keypoint array to CSV per video.

Output CSV format:
- `max_frames` rows, each row of length `expected_keypoints * 2`.
- Each received keypose contributes one row: `[x0, y0, x1, y1, ...]`.
- Frames with no valid keypoints are dropped.
- The CSV is padded with all-zero rows at the end or truncated from the end.

Assumes `handpose_decoder.py` publishes 21 markers with IDs 0..20 and uses
marker.pose.position.(x,y) as the coordinates.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2 as cv
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from visualization_msgs.msg import MarkerArray


@dataclass
class _VideoItem:
    path: Path

    @property
    def stem(self) -> str:
        return self.path.stem


class VideoCollectorNode(Node):
    """ROS 2 node that plays a directory of videos and collects pose markers."""

    def __init__(self):
        super().__init__('video_collector_node')

        # Video publishing parameters
        self.declare_parameter('width', 1920)
        self.declare_parameter('height', 960)
        self.declare_parameter('frame_id', 'theta_camera')
        self.declare_parameter('fps', 0.0)  # 0 => use source video FPS when available

        # Collector parameters
        self.declare_parameter('video_dir', '/workspaces/isaac_ros-dev/videos')
        self.declare_parameter('output_dir', '/workspaces/isaac_ros-dev/video_keyposes')
        self.declare_parameter('pose_markers_topic', 'pose_markers')
        self.declare_parameter('image_topic', 'image_raw')
        self.declare_parameter('cropped_image_topic', 'image_cropped')
        self.declare_parameter('expected_keypoints', 21)
        self.declare_parameter('max_frames', 200)
        # Drain behavior at end-of-video: allow downstream pipeline to finish producing
        # image_cropped / pose_markers before we finalize and move to next video.
        self.declare_parameter('end_drain_seconds', 0.5)  # minimum drain time after last source frame
        self.declare_parameter('drain_quiet_seconds', 0.25)  # require no new cropped frames for this long
        self.declare_parameter('max_drain_seconds', 5.0)  # safety cap

        # Cropped-video recording parameters
        self.declare_parameter('save_cropped_video', True)
        self.declare_parameter('cropped_video_suffix', '_image_cropped.mp4')
        self.declare_parameter('cropped_video_codec', 'mp4v')

        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.fps_override = float(self.get_parameter('fps').value)

        self.video_dir = Path(str(self.get_parameter('video_dir').value))
        self.output_dir = Path(str(self.get_parameter('output_dir').value))
        self.pose_markers_topic = str(self.get_parameter('pose_markers_topic').value)
        self.image_topic = str(self.get_parameter('image_topic').value)
        self.cropped_image_topic = str(self.get_parameter('cropped_image_topic').value)
        self.expected_keypoints = int(self.get_parameter('expected_keypoints').value)
        self.max_frames = int(self.get_parameter('max_frames').value)
        self.end_drain_seconds = float(self.get_parameter('end_drain_seconds').value)
        self.drain_quiet_seconds = float(self.get_parameter('drain_quiet_seconds').value)
        self.max_drain_seconds = float(self.get_parameter('max_drain_seconds').value)

        self.save_cropped_video = bool(self.get_parameter('save_cropped_video').value)
        self.cropped_video_suffix = str(self.get_parameter('cropped_video_suffix').value)
        self.cropped_video_codec = str(self.get_parameter('cropped_video_codec').value)

        if self.expected_keypoints <= 0:
            raise ValueError('expected_keypoints must be positive')
        if self.max_frames <= 0:
            raise ValueError('max_frames must be positive')

        self.row_len = self.expected_keypoints * 2

        # Publisher/subscriber
        self.image_pub = self.create_publisher(Image, self.image_topic, 10)
        self.current_video_pub = self.create_publisher(String, 'current_video', 10)
        self.bridge = CvBridge()

        self.pose_sub = self.create_subscription(
            MarkerArray,
            self.pose_markers_topic,
            self.pose_callback,
            10,
        )

        self.cropped_sub = self.create_subscription(
            Image,
            self.cropped_image_topic,
            self.cropped_callback,
            10,
        )

        # Internal state
        self._videos: List[_VideoItem] = self._discover_videos(self.video_dir)
        self._video_index: int = -1

        self._cap: Optional[cv.VideoCapture] = None
        self._publish_timer = None
        self._drain_timer = None
        self._state: str = 'idle'  # idle | playing | draining | done
        self._drain_deadline_ns: Optional[int] = None

        self._cropped_writer: Optional[cv.VideoWriter] = None
        self._cropped_writer_path: Optional[Path] = None
        self._cropped_writer_size: Optional[tuple[int, int]] = None  # (w,h)
        self._current_fps: float = 30.0

        self._last_source_frame_pub_ns: Optional[int] = None
        self._drain_started_ns: Optional[int] = None
        self._last_cropped_rx_ns: Optional[int] = None
        self._cropped_frames_written: int = 0

        self.current_video_name: str = ''
        self._current_xy: List[float] = []

        if not self._videos:
            self.get_logger().error(f'No .mp4 files found under video_dir={self.video_dir}')
            self._state = 'done'
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.get_logger().info(
            f'Found {len(self._videos)} videos in {self.video_dir}; writing CSVs to {self.output_dir}'
        )

        # Start processing
        self._start_next_video()

    def _discover_videos(self, root: Path) -> List[_VideoItem]:
        if not root.exists() or not root.is_dir():
            return []
        videos = sorted(list(root.glob('*.mp4')) + list(root.glob('*.MP4')))
        return [_VideoItem(path=p) for p in videos]

    def _effective_fps(self, cap: cv.VideoCapture) -> float:
        if self.fps_override and self.fps_override > 0:
            return float(self.fps_override)
        src_fps = float(cap.get(cv.CAP_PROP_FPS) or 0.0)
        if src_fps > 0:
            return src_fps
        return 30.0

    def _open_video(self, item: _VideoItem) -> bool:
        self._close_video()
        cap = cv.VideoCapture(str(item.path))
        if not cap.isOpened():
            self.get_logger().error(f'Failed to open video file: {item.path}')
            return False
        self._cap = cap
        return True

    def _close_video(self):
        if self._cap is not None:
            try:
                if self._cap.isOpened():
                    self._cap.release()
            finally:
                self._cap = None

    def _close_cropped_writer(self):
        if self._cropped_writer is not None:
            try:
                self._cropped_writer.release()
            except Exception:
                pass
        self._cropped_writer = None
        self._cropped_writer_path = None
        self._cropped_writer_size = None

    def _ensure_cropped_writer(self, frame_width: int, frame_height: int):
        if not self.save_cropped_video:
            return
        if self._cropped_writer is not None:
            return
        if not self.current_video_name:
            return

        out_path = self.output_dir / (Path(self.current_video_name).stem + self.cropped_video_suffix)
        fourcc = cv.VideoWriter_fourcc(*self.cropped_video_codec)
        writer = cv.VideoWriter(str(out_path), fourcc, float(self._current_fps), (int(frame_width), int(frame_height)))
        if not writer.isOpened():
            self.get_logger().error(
                f'Failed to open cropped video writer at {out_path} (codec={self.cropped_video_codec})'
            )
            return
        self._cropped_writer = writer
        self._cropped_writer_path = out_path
        self._cropped_writer_size = (int(frame_width), int(frame_height))
        self._cropped_frames_written = 0
        self.get_logger().info(f'Recording {self.cropped_image_topic} to {out_path}')

    def _cancel_timer(self, timer_attr: str):
        timer = getattr(self, timer_attr, None)
        if timer is not None:
            try:
                timer.cancel()
            except Exception:
                pass
            setattr(self, timer_attr, None)

    def _start_next_video(self):
        self._cancel_timer('_publish_timer')
        self._cancel_timer('_drain_timer')

        # Close any previous per-video resources.
        self._close_cropped_writer()

        self._last_source_frame_pub_ns = None
        self._drain_started_ns = None
        self._last_cropped_rx_ns = None

        self._video_index += 1
        if self._video_index >= len(self._videos):
            self._state = 'done'
            self.current_video_name = ''
            self.get_logger().info('All videos processed; shutting down node.')
            # Give logs a moment to flush.
            self.create_timer(0.1, self._shutdown_once)
            return

        item = self._videos[self._video_index]
        if not self._open_video(item):
            # Skip and continue.
            self._start_next_video()
            return

        self.current_video_name = item.path.name
        self.current_video_pub.publish(String(data=self.current_video_name))
        self._current_xy = []

        fps = self._effective_fps(self._cap)
        self._current_fps = float(fps)
        period = 1.0 / max(1e-6, fps)
        self.get_logger().info(f'Playing {item.path} at {fps:.3f} FPS')

        self._state = 'playing'
        self._publish_timer = self.create_timer(period, self._publish_frame)

    def _shutdown_once(self):
        if self._state != 'done':
            return
        rclpy.shutdown()

    def _publish_frame(self):
        if self._state != 'playing' or self._cap is None:
            return

        ret, frame = self._cap.read()
        if not ret:
            # End of video: stop publishing frames and allow last poses to arrive.
            self._state = 'draining'
            self._cancel_timer('_publish_timer')
            now_ns = self.get_clock().now().nanoseconds
            self._drain_started_ns = now_ns
            self._drain_deadline_ns = now_ns + int(max(0.0, self.end_drain_seconds) * 1e9)
            self._drain_timer = self.create_timer(0.05, self._drain_and_finalize)
            return

        try:
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv.resize(frame, (self.width, self.height))

            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            msg = self.bridge.cv2_to_imgmsg(frame_rgb, encoding='rgb8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            self.image_pub.publish(msg)
            self._last_source_frame_pub_ns = self.get_clock().now().nanoseconds
        except Exception as exc:
            self.get_logger().error(f'Error publishing frame: {exc}')

    def _drain_and_finalize(self):
        if self._state != 'draining':
            return

        now_ns = self.get_clock().now().nanoseconds
        # Wait at least `end_drain_seconds` AND until cropped frames have been quiet for a bit,
        # to avoid dropping late-arriving image_cropped messages.
        min_deadline_ns = self._drain_deadline_ns
        if min_deadline_ns is not None and now_ns < min_deadline_ns:
            return

        quiet_ns = int(max(0.0, self.drain_quiet_seconds) * 1e9)
        last_rx_ns = self._last_cropped_rx_ns
        if last_rx_ns is None:
            # If we never received cropped frames, consider it quiet.
            last_rx_ns = self._drain_started_ns

        if last_rx_ns is not None and quiet_ns > 0 and (now_ns - last_rx_ns) < quiet_ns:
            return

        max_ns = int(max(0.0, self.max_drain_seconds) * 1e9)
        if self._drain_started_ns is not None and max_ns > 0 and (now_ns - self._drain_started_ns) > max_ns:
            self.get_logger().warn('Max drain time exceeded; finalizing anyway.')

        self._cancel_timer('_drain_timer')

        # Finalize current video
        self._write_current_csv()
        self._close_cropped_writer()
        self._close_video()

        # Move to next
        self._start_next_video()

    def _markers_to_xy(self, msg: MarkerArray) -> Optional[List[float]]:
        markers = msg.markers
        if len(markers) != self.expected_keypoints:
            return None

        marker_map: Dict[int, object] = {int(m.id): m for m in markers}

        for i in range(self.expected_keypoints):
            if i not in marker_map:
                return None

        out: List[float] = []
        for i in range(self.expected_keypoints):
            m = marker_map[i]
            out.append(float(m.pose.position.x))
            out.append(float(m.pose.position.y))
        return out

    def pose_callback(self, msg: MarkerArray):
        if self._state not in ('playing', 'draining'):
            return

        xy = self._markers_to_xy(msg)
        if xy is None:
            return

        # Append flattened [x,y,...] for this keypose frame.
        self._current_xy.extend(xy)

    def cropped_callback(self, msg: Image):
        if self._state not in ('playing', 'draining'):
            return
        if not self.save_cropped_video:
            return

        try:
            # Prefer BGR for OpenCV VideoWriter.
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'Failed converting cropped image to cv2: {exc}')
            return

        try:
            h, w = frame_bgr.shape[:2]
            self._last_cropped_rx_ns = self.get_clock().now().nanoseconds
            self._ensure_cropped_writer(w, h)
            if self._cropped_writer is not None:
                if self._cropped_writer_size is not None:
                    ww, hh = self._cropped_writer_size
                    if (w, h) != (ww, hh):
                        frame_bgr = cv.resize(frame_bgr, (ww, hh))
                self._cropped_writer.write(frame_bgr)
                self._cropped_frames_written += 1
        except Exception as exc:
            self.get_logger().error(f'Failed writing cropped frame: {exc}')

    def _write_current_csv(self):
        if not self.current_video_name:
            return

        # Reshape flat collected stream into per-frame rows.
        data = self._current_xy
        if self.row_len <= 0:
            return

        usable_len = (len(data) // self.row_len) * self.row_len
        if usable_len != len(data):
            data = data[:usable_len]

        frames: List[List[float]] = [
            data[i : i + self.row_len] for i in range(0, len(data), self.row_len)
        ]

        # Truncate/pad in units of frames.
        if len(frames) > self.max_frames:
            frames = frames[: self.max_frames]
        elif len(frames) < self.max_frames:
            zero_row = [0.0] * self.row_len
            frames = frames + [zero_row] * (self.max_frames - len(frames))

        csv_path = self.output_dir / (Path(self.current_video_name).stem + '.csv')
        try:
            with csv_path.open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(frames)
            self.get_logger().info(
                f'Wrote {csv_path} (keypose_values={len(self._current_xy)}, rows={len(frames)}, cols={self.row_len})'
            )
        except Exception as exc:
            self.get_logger().error(f'Failed writing CSV {csv_path}: {exc}')

    def destroy_node(self):
        self._cancel_timer('_publish_timer')
        self._cancel_timer('_drain_timer')
        self._close_cropped_writer()
        self._close_video()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
