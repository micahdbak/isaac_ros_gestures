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
from sensor_msgs.msg import Image, CameraInfo, RegionOfInterest


class HandboxDecoder(Node):
    def __init__(self):
        super().__init__('handbox_decoder')

        self.score_threshold = 0.05
        self.network_size = 640

        self.bridge = CvBridge()
        self._last_image: Optional[Tuple[np.ndarray, Image]] = None
        self._image_cache: Dict[int, Tuple[np.ndarray, Image]] = {}
        self._last_roi: Tuple[float, float, float, float] = None

        self.image_sub = self.create_subscription(Image, 'image_raw', self.image_callback, 10)
        self.image_pub = self.create_publisher(Image, 'image_cropped', 1)
        self.tensor_sub = self.create_subscription(TensorList, 'tensor_output_handbox', self.tensor_callback, 10)
        self.roi_pub = self.create_publisher(CameraInfo, 'image_roi', 1)

        self.get_logger().info('HandboxDecoder started')

    def _net_to_src_xy(self, x_net: float, y_net: float, src_w: int, src_h: int) -> Tuple[float, float]:
        net = float(self.network_size)

        crop_side = float(min(src_w, src_h))
        crop_x0 = (float(src_w) - crop_side) / 2.0
        crop_y0 = (float(src_h) - crop_side) / 2.0

        src_per_net = crop_side / net
        return crop_x0 + x_net * src_per_net, crop_y0 + y_net * src_per_net

    # not currently used
    def _decode_box_to_src_xyxy(self, pred_row: np.ndarray, src_w: int, src_h: int) -> Tuple[int, int, int, int]:
        x = float(pred_row[0])
        y = float(pred_row[1])
        w = float(pred_row[2])
        h = float(pred_row[3])

        is_normalized = 0.0 <= x <= 1.5 and 0.0 <= y <= 1.5 and 0.0 <= w <= 1.5 and 0.0 <= h <= 1.5
        if is_normalized:
            net = float(self.network_size)
            x *= net
            y *= net
            w *= net
            h *= net

        x1_net = x - (w / 2.0)
        y1_net = y - (h / 2.0)
        x2_net = x + (w / 2.0)
        y2_net = y + (h / 2.0)

        x1_src, y1_src = self._net_to_src_xy(x1_net, y1_net, src_w, src_h)
        x2_src, y2_src = self._net_to_src_xy(x2_net, y2_net, src_w, src_h)

        x1_i = int(max(0.0, min(float(src_w - 1), x1_src)))
        y1_i = int(max(0.0, min(float(src_h - 1), y1_src)))
        x2_i = int(max(0.0, min(float(src_w), x2_src)))
        y2_i = int(max(0.0, min(float(src_h), y2_src)))
        return x1_i, y1_i, x2_i, y2_i

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
            while len(self._image_cache) > 10:
                self._image_cache.pop(min(self._image_cache))

    def _get_image_for_tensorlist(self, msg: TensorList) -> Optional[Tuple[np.ndarray, Image]]:
        stamp_ns = self._stamp_to_ns(getattr(getattr(msg, 'header', None), 'stamp', None))
        if stamp_ns is not None and stamp_ns in self._image_cache:
            return self._image_cache[stamp_ns]
        return self._last_image

    def _create_camera_info(self, x, y, width, height, stamp, frame_id) -> CameraInfo:
        roi = RegionOfInterest()
        roi.x_offset = int(x)
        roi.y_offset = int(y)
        roi.width = int(width)
        roi.height = int(height)
        roi.do_rectify = False

        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.roi = roi

        return msg

    def _use_last_roi(self):
        if not self._last_roi or not self._last_image:
            return

        x1, y1, x2, y2 = self._last_roi
        src_rgb, src_msg = self._last_image

        img_stamp = getattr(getattr(src_msg, 'header', None), 'stamp', None)
        if img_stamp is None:
            return

        crop = src_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return

        resized = cv.resize(crop, (self.network_size, self.network_size), interpolation=cv.INTER_LINEAR)
        cropped_msg = self.bridge.cv2_to_imgmsg(resized, encoding='rgb8')
        cropped_msg.header = src_msg.header

        self.image_pub.publish(cropped_msg)

        roi = self._create_camera_info(x1, y1, x2-x1, y2-y1, img_stamp, src_msg.header.frame_id)
        self.roi_pub.publish(roi)

    def tensor_callback(self, msg: TensorList):
        if len(msg.tensors) == 0:
            self.get_logger().info("NO TENSORS")
            self._use_last_roi()
            return

        tensor_stamp = getattr(getattr(msg, 'header', None), 'stamp', None)
        if tensor_stamp is None:
            self.get_logger().info("NO STAMP")
            self._use_last_roi()
            return

        image_pair = self._get_image_for_tensorlist(msg)
        if image_pair is None:
            self.get_logger().info("NO PAIR")
            self._use_last_roi()
            return

        src_rgb, src_msg = image_pair
        src_h, src_w = src_rgb.shape[:2]

        tensor = msg.tensors[0]
        data = np.frombuffer(tensor.data, dtype=np.float32)
        out = data.reshape(tuple(tensor.shape.dims))
        pred = out[0]

        conf = pred[:, 4]
        best_idx = int(np.argmax(conf))
        best_pred = pred[best_idx]

        valid = float(conf[best_idx]) >= float(self.score_threshold)
        if not valid:
            self.get_logger().info("NOT CONFIDENT")
            self._use_last_roi()
            return

        kpts = best_pred[6:].reshape(21, 3)
        xs = kpts[:, 0].astype(np.float32)
        ys = kpts[:, 1].astype(np.float32)

        is_normalized = (
            float(np.min(xs)) >= -0.5 and float(np.max(xs)) <= 1.5 and
            float(np.min(ys)) >= -0.5 and float(np.max(ys)) <= 1.5
        )
        if is_normalized:
            xs *= float(self.network_size)
            ys *= float(self.network_size)

        x1_net = float(np.min(xs))
        y1_net = float(np.min(ys))
        x2_net = float(np.max(xs))
        y2_net = float(np.max(ys))

        x1_src_f, y1_src_f = self._net_to_src_xy(x1_net, y1_net, src_w, src_h)
        x2_src_f, y2_src_f = self._net_to_src_xy(x2_net, y2_net, src_w, src_h)

        margin_x = 0.08 * float(src_w)
        margin_y = 0.08 * float(src_h)

        x1 = int(max(0.0, min(float(src_w - 1), min(x1_src_f, x2_src_f) - margin_x)))
        y1 = int(max(0.0, min(float(src_h - 1), min(y1_src_f, y2_src_f) - margin_y)))
        x2 = int(max(0.0, min(float(src_w), max(x1_src_f, x2_src_f) + margin_x)))
        y2 = int(max(0.0, min(float(src_h), max(y1_src_f, y2_src_f) + margin_y)))

        if x2 <= x1 + 1 or y2 <= y1 + 1:
            self.get_logger().info("BAD COORDS")
            self._use_last_roi()
            return

        self._last_roi = (x1, y1, x2, y2)

        crop = src_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            self.get_logger().info("CROP SIZE")
            self._use_last_roi()
            return

        resized = cv.resize(crop, (self.network_size, self.network_size), interpolation=cv.INTER_LINEAR)
        cropped_msg = self.bridge.cv2_to_imgmsg(resized, encoding='rgb8')
        cropped_msg.header = src_msg.header

        self.get_logger().info("SUCCESS")

        self.image_pub.publish(cropped_msg)

        roi = self._create_camera_info(x1, y1, x2-x1, y2-y1, tensor_stamp, src_msg.header.frame_id)
        self.roi_pub.publish(roi)


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
