#!/usr/bin/env python3

"""Ultralytics YOLO inference on an encoded tensor topic.

Subscribes to an Isaac ROS `isaac_ros_tensor_list_interfaces/TensorList` stream
(default: `/tensor_view_handbox`) which is typically produced by
`DnnImageEncoderNode`.

Expected tensor shape: `1x3x640x640` (NCHW, RGB). The node decodes the tensor
into an image, runs an Ultralytics YOLO `.pt` model, and visualizes detections
using OpenCV.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from isaac_ros_tensor_list_interfaces.msg import TensorList
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class YoloInferenceNode(Node):
	def __init__(self):
		super().__init__('yolo_inference_tensor_node')

		default_model = str(Path(__file__).resolve().parent / 'guide-dog-hand-detector.pt')

		self.declare_parameter('tensor_topic', '/tensor_view_handbox')
		self.declare_parameter('tensor_index', 0)
		self.declare_parameter('expected_dims', [1, 3, 640, 640])
		self.declare_parameter('model_path', default_model)
		self.declare_parameter('device', '')  # '', 'cpu', '0', '0,1', etc.
		self.declare_parameter('imgsz', 640)
		self.declare_parameter('conf', 0.25)
		self.declare_parameter('iou', 0.45)
		self.declare_parameter('max_det', 50)
		self.declare_parameter('show_side_by_side', True)
		self.declare_parameter('window_name', 'YOLO26 Hand Detector (Tensor Input)')
		self.declare_parameter('drop_frames_when_busy', True)

		self.tensor_topic = str(self.get_parameter('tensor_topic').value)
		self.tensor_index = int(self.get_parameter('tensor_index').value)
		self.expected_dims = [int(x) for x in list(self.get_parameter('expected_dims').value)]
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

		self._busy = False
		self._last_fps_time_s = time.perf_counter()
		self._fps_ema: Optional[float] = None

		self.sub = self.create_subscription(
			TensorList,
			self.tensor_topic,
			self._on_tensor,
			qos_profile_sensor_data,
		)

		cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
		self.get_logger().info(f'Subscribed to {self.tensor_topic}. Press q in the window to quit.')

	def _tensor_to_rgb8(self, tensor) -> Optional[np.ndarray]:
		dims = list(getattr(getattr(tensor, 'shape', None), 'dims', []))
		if dims and self.expected_dims and [int(x) for x in dims] != self.expected_dims:
			self.get_logger().warn(f'Unexpected tensor dims: {dims} (expected {self.expected_dims})')

		if len(dims) != 4:
			return None
		n, c, h, w = (int(dims[0]), int(dims[1]), int(dims[2]), int(dims[3]))
		if n < 1 or c != 3 or h <= 0 or w <= 0:
			return None

		data = getattr(tensor, 'data', b'')
		if not data:
			return None

		expected_elems = int(np.prod(dims))
		if expected_elems <= 0:
			return None

		# DnnImageEncoder commonly outputs float32; sometimes uint8.
		if len(data) == expected_elems * 4:
			arr = np.frombuffer(data, dtype=np.float32)
		elif len(data) == expected_elems:
			arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
		else:
			self.get_logger().warn(f'Unexpected tensor byte length: {len(data)} for dims {dims}')
			return None

		arr = arr.reshape((n, c, h, w))[0]
		arr = np.transpose(arr, (1, 2, 0))  # HWC, RGB

		if not np.isfinite(arr).all():
			arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

		vmin = float(np.min(arr))
		vmax = float(np.max(arr))

		if vmin >= 0.0 and vmax <= 1.5:
			vis = np.clip(arr, 0.0, 1.0) * 255.0
		elif vmin >= 0.0 and vmax <= 255.0:
			vis = np.clip(arr, 0.0, 255.0)
		else:
			lo, hi = np.percentile(arr, [1.0, 99.0])
			if float(hi) <= float(lo) + 1e-9:
				vis = np.zeros((h, w, 3), dtype=np.float32)
			else:
				vis = (arr - float(lo)) * (255.0 / float(hi - lo))
				vis = np.clip(vis, 0.0, 255.0)

		return vis.astype(np.uint8)

	def _on_tensor(self, msg: TensorList):
		if self.drop_frames_when_busy and self._busy:
			return

		if self.tensor_index < 0 or self.tensor_index >= len(msg.tensors):
			return

		self._busy = True
		try:
			rgb = self._tensor_to_rgb8(msg.tensors[self.tensor_index])
			if rgb is None:
				return

			cv_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

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

