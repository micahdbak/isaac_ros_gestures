#!/usr/bin/env python3

"""Ultralytics YOLO inference on /image_raw.

Subscribes to a ROS 2 `sensor_msgs/Image` stream (default: /image_raw), runs an
Ultralytics YOLO `.pt` model on each frame, and visualizes the annotated output
using OpenCV.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class YoloInferenceNode(Node):
	def __init__(self):
		super().__init__('yolo_inference_node')

		default_model = str(Path(__file__).resolve().parent / 'guide-dog-hand-detector.pt')

		self.declare_parameter('image_topic', '/image_raw')
		self.declare_parameter('model_path', default_model)
		self.declare_parameter('device', '')  # '', 'cpu', '0', '0,1', etc.
		self.declare_parameter('imgsz', 640)
		self.declare_parameter('conf', 0.25)
		self.declare_parameter('iou', 0.45)
		self.declare_parameter('max_det', 50)
		self.declare_parameter('show_side_by_side', True)
		self.declare_parameter('window_name', 'YOLO26 Hand Detector')
		self.declare_parameter('drop_frames_when_busy', True)

		self.image_topic = str(self.get_parameter('image_topic').value)
		self.model_path = str(self.get_parameter('model_path').value)
		self.device = str(self.get_parameter('device').value)
		self.imgsz = int(self.get_parameter('imgsz').value)
		self.conf = float(self.get_parameter('conf').value)
		self.iou = float(self.get_parameter('iou').value)
		self.max_det = int(self.get_parameter('max_det').value)
		self.show_side_by_side = bool(self.get_parameter('show_side_by_side').value)
		self.window_name = str(self.get_parameter('window_name').value)
		self.drop_frames_when_busy = bool(self.get_parameter('drop_frames_when_busy').value)

		if not Path(self.model_path).exists():
			self.get_logger().warning(
				f'Model not found at {self.model_path}. '
				'Set --ros-args -p model_path:=/abs/path/to/model.pt'
			)

		try:
			from ultralytics import YOLO  # type: ignore
		except Exception as e:
			raise RuntimeError(
				'Failed to import ultralytics. Install it into your environment, e.g.:\n'
				'  pip install ultralytics\n\n'
				f'Import error: {e}'
			)

		self.get_logger().info(f'Loading Ultralytics model: {self.model_path}')
		self.model = YOLO(self.model_path)

		self.bridge = CvBridge()
		self._busy = False
		self._last_fps_time_s = time.perf_counter()
		self._fps_ema: Optional[float] = None

		self.sub = self.create_subscription(
			Image,
			self.image_topic,
			self._on_image,
			qos_profile_sensor_data,
		)

		cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
		self.get_logger().info(f'Subscribed to {self.image_topic}. Press q in the window to quit.')

	def _on_image(self, msg: Image):
		if self.drop_frames_when_busy and self._busy:
			return

		self._busy = True
		try:
			# Theta publishes rgb8. Convert to BGR for OpenCV/Ultralytics.
			cv_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
			cv_bgr = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2BGR)
			print(cv_bgr.shape)
			t0 = time.perf_counter()
			results_list = self.model.predict(
				source=cv_bgr,
				imgsz=self.imgsz,
				conf=self.conf,
				iou=self.iou,
				max_det=self.max_det,
				device=self.device if self.device else None,
				verbose=False,
			)
			dt_ms = (time.perf_counter() - t0) * 1000.0

			result = results_list[0]
			annotated = result.plot()  # BGR annotated frame

			# Lightweight FPS estimate
			now_s = time.perf_counter()
			dt_s = max(1e-6, now_s - self._last_fps_time_s)
			inst_fps = 1.0 / dt_s
			self._last_fps_time_s = now_s
			self._fps_ema = inst_fps if self._fps_ema is None else (0.9 * self._fps_ema + 0.1 * inst_fps)

			label = f'{(self._fps_ema or 0.0):.1f} FPS | {dt_ms:.1f} ms'
			cv2.putText(
				annotated,
				label,
				(10, 30),
				cv2.FONT_HERSHEY_SIMPLEX,
				1.0,
				(0, 255, 0),
				2,
				cv2.LINE_AA,
			)

			if self.show_side_by_side:
				# Ensure both have same height.
				h = min(cv_bgr.shape[0], annotated.shape[0])
				if cv_bgr.shape[0] != h:
					cv_bgr = cv2.resize(cv_bgr, (int(cv_bgr.shape[1] * (h / cv_bgr.shape[0])), h))
				if annotated.shape[0] != h:
					annotated = cv2.resize(annotated, (int(annotated.shape[1] * (h / annotated.shape[0])), h))
				vis = cv2.hconcat([cv_bgr, annotated])
			else:
				vis = annotated

			cv2.imshow(self.window_name, vis)
			key = cv2.waitKey(1) & 0xFF
			if key == ord('q'):
				self.get_logger().info('Quit requested; shutting down.')
				rclpy.shutdown()
		except Exception as e:
			self.get_logger().error(f'Inference/visualization error: {e}')
		finally:
			self._busy = False

	def destroy_node(self):
		try:
			cv2.destroyWindow(self.window_name)
		except Exception:
			pass
		super().destroy_node()


def main(args=None):
	rclpy.init(args=args)
	node: Optional[YoloInferenceNode] = None
	try:
		node = YoloInferenceNode()
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		if node is not None:
			node.destroy_node()
		if rclpy.ok():
			rclpy.shutdown()


if __name__ == '__main__':
	main()

