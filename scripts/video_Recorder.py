#!/usr/bin/env python3

from __future__ import annotations

import os
import time
from pathlib import Path
from threading import Lock

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image


class VideoRecorder(Node):
	def __init__(self) -> None:
		super().__init__('video_recorder')

		self.declare_parameter('topic', '/image_raw')
		self.declare_parameter('output', '')
		self.declare_parameter('fps', 24.0)
		self.declare_parameter('codec', 'mp4v')
		self.declare_parameter('display', True)
		self.declare_parameter('encoding', 'bgr8')
		self.declare_parameter('queue_size', 10)

		self._topic: str = self.get_parameter('topic').get_parameter_value().string_value
		self._output: str = self.get_parameter('output').get_parameter_value().string_value
		self._fps: float = self.get_parameter('fps').get_parameter_value().double_value
		self._codec: str = self.get_parameter('codec').get_parameter_value().string_value
		self._display: bool = self.get_parameter('display').get_parameter_value().bool_value
		self._encoding: str = self.get_parameter('encoding').get_parameter_value().string_value
		self._queue_size: int = int(self.get_parameter('queue_size').get_parameter_value().integer_value)

		if not self._output:
			stamp = time.strftime('%Y%m%d_%H%M%S')
			self._output = str(Path.home() / f'ros_video_{stamp}.mp4')

		out_path = Path(os.path.expanduser(self._output)).resolve()
		out_path.parent.mkdir(parents=True, exist_ok=True)
		self._output = str(out_path)

		qos = QoSProfile(
			reliability=QoSReliabilityPolicy.BEST_EFFORT,
			history=QoSHistoryPolicy.KEEP_LAST,
			depth=max(self._queue_size, 1),
		)

		self._bridge = CvBridge()
		self._writer: cv2.VideoWriter | None = None
		self._writer_lock = Lock()
		self._stopped = False
		self._frame_count = 0

		self._sub = self.create_subscription(Image, self._topic, self._on_image, qos)
		self.get_logger().info(
			f"Recording {self._topic} -> {self._output} (fps={self._fps}, codec={self._codec}, display={self._display})"
		)
		if self._display:
			self.get_logger().info("Press 'q' in the preview window to stop.")

	def _init_writer(self, width: int, height: int) -> None:
		fourcc = cv2.VideoWriter_fourcc(*self._codec)
		writer = cv2.VideoWriter(self._output, fourcc, float(self._fps), (int(width), int(height)))
		if not writer.isOpened():
			raise RuntimeError(
				f"Failed to open VideoWriter for '{self._output}'. "
				f"Try a different 'codec' (e.g. XVID for .avi, mp4v for .mp4) or output path."
			)
		self._writer = writer

	def _on_image(self, msg: Image) -> None:
		if self._stopped:
			return

		try:
			frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding=self._encoding)
		except Exception as exc:  # noqa: BLE001
			self.get_logger().warn(f'cv_bridge conversion failed: {exc}')
			return

		# Ensure BGR for most codecs.
		if frame.ndim == 2:
			frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
		elif frame.shape[2] == 4:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

		with self._writer_lock:
			if self._writer is None:
				try:
					self._init_writer(frame.shape[1], frame.shape[0])
				except Exception as exc:  # noqa: BLE001
					self.get_logger().error(str(exc))
					self.stop()
					rclpy.shutdown()
					return

			assert self._writer is not None
			self._writer.write(frame)
			self._frame_count += 1

		if self._display:
			cv2.imshow('ROS Video Recorder', frame)
			key = cv2.waitKey(1) & 0xFF
			if key in (ord('q'), ord('Q')):
				self.get_logger().info('q pressed — stopping recording')
				self.stop()
				rclpy.shutdown()

	def stop(self) -> None:
		if self._stopped:
			return
		self._stopped = True

		try:
			if hasattr(self, '_sub') and self._sub is not None:
				self.destroy_subscription(self._sub)
		except Exception:  # noqa: BLE001
			pass

		with self._writer_lock:
			if self._writer is not None:
				try:
					self._writer.release()
				except Exception:  # noqa: BLE001
					pass
				self._writer = None

		try:
			cv2.destroyAllWindows()
		except Exception:  # noqa: BLE001
			pass

		self.get_logger().info(f'Saved {self._frame_count} frames to {self._output}')


def main() -> None:
	rclpy.init()
	node = VideoRecorder()
	rclpy.get_default_context().on_shutdown(node.stop)
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.stop()
		node.destroy_node()
		if rclpy.ok():
			rclpy.shutdown()


if __name__ == '__main__':
	main()
