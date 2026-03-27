#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray


class SessionCollectorNode(Node):
    def __init__(self) -> None:
        super().__init__('session_collector_node')

        self.declare_parameter('button_topic', '/arduino_buttons')
        self.declare_parameter('hand_marker_topic', '/pose_markers')
        self.declare_parameter('body_marker_topic', '/fullbody_pose_markers')
        self.declare_parameter('image_topic', '/image_cropped')
        self.declare_parameter('save_dir', '/workspaces/isaac_ros-dev/src/isaac_ros_gestures/data/go')
        self.declare_parameter('video_fps', 20.0)
        self.declare_parameter('video_codec', 'mp4v')

        self.button_topic = str(self.get_parameter('button_topic').value)
        self.hand_marker_topic = str(self.get_parameter('hand_marker_topic').value)
        self.body_marker_topic = str(self.get_parameter('body_marker_topic').value)
        self.image_topic = str(self.get_parameter('image_topic').value)
        self.save_dir = Path(str(self.get_parameter('save_dir').value))
        self.video_fps = float(self.get_parameter('video_fps').value)
        self.video_codec = str(self.get_parameter('video_codec').value)

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()

        self._recording = False
        self._current_index: Optional[int] = None
        self._current_hand_path: Optional[Path] = None
        self._current_body_path: Optional[Path] = None
        self._current_video_path: Optional[Path] = None
        self._video_writer: Optional[cv2.VideoWriter] = None

        self._hand_frames: List[np.ndarray] = []
        self._body_frames: List[np.ndarray] = []
        self._expected_hand_landmarks: Optional[int] = None
        self._expected_body_landmarks: Optional[int] = None
        self._failed=False
        self.button_sub = self.create_subscription(
            String, self.button_topic, self.button_callback, 10
        )
        self.hand_marker_sub = self.create_subscription(
            MarkerArray, self.hand_marker_topic, self.hand_marker_callback, 10
        )
        self.body_marker_sub = self.create_subscription(
            MarkerArray, self.body_marker_topic, self.body_marker_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10
        )

        self.get_logger().info(
            f'Session collector ready. '
            f'button={self.button_topic}, '
            f'hand_markers={self.hand_marker_topic}, '
            f'body_markers={self.body_marker_topic}, '
            f'image={self.image_topic}, save_dir={self.save_dir}'
        )

    def _find_existing_indices(self) -> List[int]:
        indices = set()

        patterns = [
            'session_*_hand.npy',
            'session_*_body.npy',
            'session_*.mp4',
        ]

        for pattern in patterns:
            for path in self.save_dir.glob(pattern):
                try:
                    name = path.stem  # e.g. session_0003_hand
                    parts = name.split('_')
                    idx = int(parts[1])
                    indices.add(idx)
                except Exception:
                    pass

        return sorted(indices)

    def _find_last_existing_index(self) -> Optional[int]:
        indices = self._find_existing_indices()
        return indices[-1] if indices else None

    def _next_index(self) -> int:
        indices = self._find_existing_indices()
        return 0 if not indices else indices[-1] + 1

    def _release_video_writer(self) -> None:
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

    def _extract_xy_frame(self, msg: MarkerArray) -> Optional[np.ndarray]:
        frame_points = []

        for marker in msg.markers:
            if len(marker.points) > 0:
                for p in marker.points:
                    frame_points.append([p.x, p.y])
            else:
                pos = marker.pose.position
                frame_points.append([pos.x, pos.y])

        if len(frame_points) == 0:
            return None

        return np.asarray(frame_points, dtype=np.float32)

    def _save_array(self, path: Optional[Path], frames: List[np.ndarray], dims: int = 2) -> None:
        if path is None:
            return

        if len(frames) == 0:
            arr = np.empty((0, 0, dims), dtype=np.float32)
            self._failed=True
        else:
            arr = np.stack(frames, axis=0).astype(np.float32)

        np.save(path, arr)
        self.get_logger().info(f'Saved {path.name} with shape {arr.shape}')

    def start_session(self) -> None:
        if self._recording:
            self.get_logger().warn('Button 1 pressed but a session is already recording.')
            return

        idx = self._next_index()
        self._current_index = idx
        self._current_hand_path = self.save_dir / f'session_{idx:04d}_hand.npy'
        self._current_body_path = self.save_dir / f'session_{idx:04d}_body.npy'
        self._current_video_path = self.save_dir / f'session_{idx:04d}.mp4'

        self._hand_frames = []
        self._body_frames = []
        self._expected_hand_landmarks = None
        self._expected_body_landmarks = None
        self._release_video_writer()
        self._recording = True

        self.get_logger().info(
            f'Started session {idx:04d}: '
            f'{self._current_hand_path.name}, '
            f'{self._current_body_path.name}, '
            f'{self._current_video_path.name}'
        )

    def stop_session(self) -> None:
        if not self._recording:
            self.get_logger().warn('Button 1 released but no session is recording.')
            return
        try:
            self._save_array(self._current_hand_path, self._hand_frames, dims=2)
            self._save_array(self._current_body_path, self._body_frames, dims=2)
        finally:
            self._release_video_writer()

        self._recording = False
        self._current_index = None
        self._current_hand_path = None
        self._current_body_path = None
        self._current_video_path = None
        self._hand_frames = []
        self._body_frames = []
        self._expected_hand_landmarks = None
        self._expected_body_landmarks = None

    def delete_last_session(self) -> None:
        if self._recording:
            self.get_logger().warn('Button 2 pressed while recording. Ignoring delete request.')
            return

        idx = self._find_last_existing_index()
        if idx is None:
            self.get_logger().warn('No saved session found to delete.')
            return

        hand_path = self.save_dir / f'session_{idx:04d}_hand.npy'
        body_path = self.save_dir / f'session_{idx:04d}_body.npy'
        video_path = self.save_dir / f'session_{idx:04d}.mp4'

        deleted_any = False

        for path in [hand_path, body_path, video_path]:
            if path.exists():
                path.unlink()
                self.get_logger().info(f'Deleted {path.name}')
                deleted_any = True

        if not deleted_any:
            self.get_logger().warn(f'No files found for session {idx:04d}')

    def button_callback(self, msg: String) -> None:
        data = msg.data.strip()

        if data == 'button_1_pressed':
            self.start_session()
        elif data == 'button_1_released':
            self.stop_session()
            if self._failed==True:
                self.get_logger().warn("last session collected nothing, deleted")
                self.delete_last_session()
                self._failed=False

        elif data == 'button_2_pressed':
            self.delete_last_session()

    def hand_marker_callback(self, msg: MarkerArray) -> None:
        if not self._recording:
            return

        frame_array = self._extract_xy_frame(msg)
        if frame_array is None:
            return

        if self._expected_hand_landmarks is None:
            self._expected_hand_landmarks = 21#frame_array.shape[0]
            self.get_logger().info(
                f'Using {self._expected_hand_landmarks} hand landmarks per frame.'
            )
        if frame_array.shape[0] != self._expected_hand_landmarks:
            self.get_logger().warn(
                f'Skipping hand frame with {frame_array.shape[0]} landmarks; '
                f'expected {self._expected_hand_landmarks}.'
            )
            
            return

        self._hand_frames.append(frame_array)

    def body_marker_callback(self, msg: MarkerArray) -> None:
        if not self._recording:
            return

        frame_array = self._extract_xy_frame(msg)
        if frame_array is None:
            return

        if self._expected_body_landmarks is None:
            self._expected_body_landmarks = 17#frame_array.shape[0]
            self.get_logger().info(
                f'Using {self._expected_body_landmarks} body landmarks per frame.'
            )
        if frame_array.shape[0] != self._expected_body_landmarks:
            self.get_logger().warn(
                f'Skipping body frame with {frame_array.shape[0]} landmarks; '
                f'expected {self._expected_body_landmarks}.'
            )
            

            return

        self._body_frames.append(frame_array)

    def image_callback(self, msg: Image) -> None:
        if not self._recording or self._current_video_path is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image_cropped: {e}')
            return

        h, w = frame.shape[:2]

        if self._video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
            self._video_writer = cv2.VideoWriter(
                str(self._current_video_path),
                fourcc,
                self.video_fps,
                (w, h),
            )

            if not self._video_writer.isOpened():
                self.get_logger().error(
                    f'Failed to open video writer for {self._current_video_path}'
                )
                self._release_video_writer()
                return

            self.get_logger().info(
                f'Opened video writer {self._current_video_path.name} at {w}x{h}'
            )

        self._video_writer.write(frame)

    def destroy_node(self):
        if self._recording:
            self.get_logger().warn('Shutting down during recording. Finalizing session.')
            self.stop_session()

        self._release_video_writer()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SessionCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()