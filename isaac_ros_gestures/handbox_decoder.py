#!/usr/bin/env python3

"""Handbox decoder node.

Consumes YOLO26 TensorRT output run on the full /image_raw, selects the best hand
bounding box, crops that region from /image_raw, resizes to 640x640, and publishes
it to /image_cropped.

Parsing of TensorRT output is intentionally mirrored from handpose_decoder.py.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2 as cv
import numpy as np
import rclpy
from cv_bridge import CvBridge
from isaac_ros_tensor_list_interfaces.msg import TensorList
from rclpy.node import Node
from sensor_msgs.msg import Image


class HandboxDecoder(Node):
    def __init__(self):
        super().__init__('handbox_decoder')

        self.declare_parameter('score_threshold', 0.25)
        self.declare_parameter('network_size', 640.0)
        self.declare_parameter('padding_ratio', 2.0)
        self.declare_parameter('max_movement_threshold', 250.0)
        self.declare_parameter('image_topic', 'image_raw')
        self.declare_parameter('tensor_topic', 'tensor_output_handdet')

        self.score_threshold = float(self.get_parameter('score_threshold').value)
        self.network_size = float(self.get_parameter('network_size').value)
        self.padding_ratio = float(self.get_parameter('padding_ratio').value)
        self.max_movement_threshold = float(self.get_parameter('max_movement_threshold').value)
        self.image_topic = str(self.get_parameter('image_topic').value)
        self.tensor_topic = str(self.get_parameter('tensor_topic').value)

        self.bridge = CvBridge()

        self._last_image: Optional[Tuple[np.ndarray, Image]] = None  # (rgb image, msg)
        self._image_cache: Dict[int, Tuple[np.ndarray, Image]] = {}

        self._last_hand_center: Optional[Tuple[float, float]] = None
        self._last_crop_xyxy: Optional[Tuple[int, int, int, int]] = None

        self.image_pub = self.create_publisher(Image, 'image_cropped', 1)

        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.tensor_sub = self.create_subscription(TensorList, self.tensor_topic, self.tensor_callback, 10)

    def _stamp_to_ns(self, stamp) -> Optional[int]:
        try:
            return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        except Exception:
            return None

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as exc:
            self.get_logger().error(f'Failed to decode image: {exc}')
            return

        self._last_image = (cv_image, msg)

        stamp_ns = self._stamp_to_ns(getattr(getattr(msg, 'header', None), 'stamp', None))
        if stamp_ns is not None:
            self._image_cache[stamp_ns] = (cv_image, msg)
            if len(self._image_cache) > 10:
                for key in sorted(self._image_cache.keys())[:-10]:
                    del self._image_cache[key]

    def _get_image_for_tensorlist(self, msg: TensorList) -> Optional[Tuple[np.ndarray, Image]]:
        stamp_ns = self._stamp_to_ns(getattr(getattr(msg, 'header', None), 'stamp', None))
        if stamp_ns is not None and stamp_ns in self._image_cache:
            return self._image_cache[stamp_ns]
        return self._last_image

    def _letterbox_params(self, src_w: int, src_h: int) -> Tuple[float, float, float]:
        n = float(self.network_size)
        scale = min(n / float(src_w), n / float(src_h))
        resized_w = float(src_w) * scale
        resized_h = float(src_h) * scale
        pad_x = (n - resized_w) / 2.0
        pad_y = (n - resized_h) / 2.0
        return scale, pad_x, pad_y

    def _xywh_center_to_xyxy(self, x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
        return (x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0)

    def _network_box_to_image_xyxy(
        self,
        box_xywh: Tuple[float, float, float, float],
        src_w: int,
        src_h: int,
    ) -> Optional[Tuple[float, float, float, float]]:
        x, y, w, h = box_xywh

        # Heuristic: some exports emit normalized coordinates. Support both.
        if max(x, y, w, h) <= 2.0:
            n = float(self.network_size)
            x *= n
            y *= n
            w *= n
            h *= n

        # Convert xywh(center) in network coordinates -> xyxy in network coordinates.
        x1, y1, x2, y2 = self._xywh_center_to_xyxy(float(x), float(y), float(w), float(h))

        scale, pad_x, pad_y = self._letterbox_params(src_w, src_h)
        if scale <= 0:
            return None

        # Undo letterbox padding then scale back to source image.
        x1 = (x1 - pad_x) / scale
        x2 = (x2 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        y2 = (y2 - pad_y) / scale

        # Clamp to source image.
        x1 = max(0.0, min(float(src_w), x1))
        x2 = max(0.0, min(float(src_w), x2))
        y1 = max(0.0, min(float(src_h), y1))
        y2 = max(0.0, min(float(src_h), y2))

        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _expand_to_square(self, xyxy: Tuple[float, float, float, float], src_w: int, src_h: int):
        x1, y1, x2, y2 = xyxy
        bw = x2 - x1
        bh = y2 - y1
        cx = x1 + bw / 2.0
        cy = y1 + bh / 2.0

        # Movement gating (optional stabilization)
        if self._last_hand_center is not None:
            dist = ((cx - self._last_hand_center[0]) ** 2 + (cy - self._last_hand_center[1]) ** 2) ** 0.5
            if dist > self.max_movement_threshold and self._last_crop_xyxy is not None:
                return self._last_crop_xyxy

        self._last_hand_center = (cx, cy)

        side = max(bw, bh) * max(1.0, float(self.padding_ratio))
        side = max(4.0, side)

        nx1 = int(round(cx - side / 2.0))
        ny1 = int(round(cy - side / 2.0))
        nx2 = int(round(cx + side / 2.0))
        ny2 = int(round(cy + side / 2.0))

        nx1 = max(0, min(src_w - 1, nx1))
        ny1 = max(0, min(src_h - 1, ny1))
        nx2 = max(1, min(src_w, nx2))
        ny2 = max(1, min(src_h, ny2))

        # Ensure even dimensions.
        if (nx2 - nx1) % 2 != 0:
            nx2 = max(nx1 + 2, nx2 - 1)
        if (ny2 - ny1) % 2 != 0:
            ny2 = max(ny1 + 2, ny2 - 1)

        if nx2 <= nx1 or ny2 <= ny1:
            return None

        self._last_crop_xyxy = (nx1, ny1, nx2, ny2)
        return self._last_crop_xyxy

    def tensor_callback(self, msg: TensorList):
        if len(msg.tensors) == 0:
            return

        image_pair = self._get_image_for_tensorlist(msg)
        if image_pair is None:
            return

        src_rgb, src_msg = image_pair
        src_h, src_w = src_rgb.shape[:2]

        # --- Tensor parsing mirrored from handpose_decoder.py ---
        tensor = msg.tensors[0]
        data = np.frombuffer(tensor.data, dtype=np.float32)
        out = data.reshape(tuple(tensor.shape.dims))
        pred = out[0]

        conf = pred[:, 4]
        best_idx = int(np.argmax(conf))
        best_pred = pred[best_idx]

        valid = float(conf[best_idx]) >= float(self.score_threshold)

        crop_xyxy: Optional[Tuple[int, int, int, int]] = None

        if valid:
            net_box = (float(best_pred[0]), float(best_pred[1]), float(best_pred[2]), float(best_pred[3]))
            img_xyxy = self._network_box_to_image_xyxy(net_box, src_w=src_w, src_h=src_h)
            if img_xyxy is not None:
                crop_xyxy = self._expand_to_square(img_xyxy, src_w=src_w, src_h=src_h)

        if crop_xyxy is None:
            # Fallback to last known crop, if any.
            crop_xyxy = self._last_crop_xyxy

        if crop_xyxy is None:
            return

        x1, y1, x2, y2 = crop_xyxy
        if x2 <= x1 or y2 <= y1:
            return

        crop = src_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return

        resized = cv.resize(crop, (int(self.network_size), int(self.network_size)), interpolation=cv.INTER_LINEAR)

        cropped_msg = self.bridge.cv2_to_imgmsg(resized, encoding='rgb8')
        cropped_msg.header = src_msg.header

        self.image_pub.publish(cropped_msg)



def main(args=None):
    rclpy.init(args=args)
    node = HandboxDecoder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
