#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional, Tuple

import cv2 as cv
import numpy as np
import rclpy
from cv_bridge import CvBridge
from isaac_ros_tensor_list_interfaces.msg import TensorList
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, RegionOfInterest


class HandboxDecoder(Node):
    def __init__(self):
        super().__init__('handbox_decoder')

        self.declare_parameter('image_topic', 'image_raw')
        self.declare_parameter('tensor_topic', 'tensor_output_handbox')
        self.declare_parameter('cropped_topic', 'image_cropped')
        self.declare_parameter('roi_topic', 'image_roi')

        self.declare_parameter('score_threshold', 0.10)
        self.declare_parameter('model_size', 640)
        self.declare_parameter('crop_size', 640)

        # Fixed crop size in source-image pixels
        self.declare_parameter('fixed_box_size', 320)

        # Keep only detections whose center lies in the middle portion of the image width
        # Example: 0.5 means only middle 50% of the image width is allowed
        self.declare_parameter('center_region_width_ratio', 0.6)

        self.image_topic = str(self.get_parameter('image_topic').value)
        self.tensor_topic = str(self.get_parameter('tensor_topic').value)
        self.cropped_topic = str(self.get_parameter('cropped_topic').value)
        self.roi_topic = str(self.get_parameter('roi_topic').value)

        self.score_threshold = float(self.get_parameter('score_threshold').value)
        self.model_size = int(self.get_parameter('model_size').value)
        self.crop_size = int(self.get_parameter('crop_size').value)
        self.fixed_box_size = int(self.get_parameter('fixed_box_size').value)
        self.center_region_width_ratio = float(self.get_parameter('center_region_width_ratio').value)

        self.bridge = CvBridge()
        self.last_image: Optional[np.ndarray] = None
        self.last_image_msg: Optional[Image] = None

        # Store previous crop box in source-image coordinates
        self.last_crop_box: Optional[Tuple[int, int, int, int]] = None

        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10
        )
        self.tensor_sub = self.create_subscription(
            TensorList, self.tensor_topic, self.tensor_callback, 10
        )

        self.image_pub = self.create_publisher(Image, self.cropped_topic, 10)
        self.roi_pub = self.create_publisher(CameraInfo, self.roi_topic, 10)

        self.get_logger().info('HandboxDecoder started')

    def image_callback(self, msg: Image) -> None:
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f'Failed to decode image: {e}')
            return

        self.last_image = img
        self.last_image_msg = msg

    def tensor_callback(self, msg: TensorList) -> None:
        if self.last_image is None or self.last_image_msg is None:
            return

        src_img = self.last_image
        src_msg = self.last_image_msg
        src_h, src_w = src_img.shape[:2]

        crop_box: Optional[Tuple[int, int, int, int]] = None

        if msg.tensors:
            try:
                pred = self.parse_yolo_output(msg.tensors[0])
            except Exception as e:
                self.get_logger().error(f'Failed to parse tensor output: {e}')
                pred = None

            if pred is not None:
                det_box = self.model_box_to_source_box(
                    pred=pred,
                    src_w=src_w,
                    src_h=src_h,
                    model_size=self.model_size,
                )

                if self.is_detection_valid(det_box, src_w):
                    cx = 0.5 * (det_box[0] + det_box[2])
                    cy = 0.5 * (det_box[1] + det_box[3])

                    crop_box = self.make_fixed_square_crop_box(
                        cx=cx,
                        cy=cy,
                        side=self.fixed_box_size,
                        src_w=src_w,
                        src_h=src_h,
                    )

        # Fallback to previous crop box if current detection is unusable
        if crop_box is None:
            crop_box = self.last_crop_box

        if crop_box is None:
            return

        x1, y1, x2, y2 = crop_box
        if x2 <= x1 or y2 <= y1:
            return

        crop = src_img[y1:y2, x1:x2]
        if crop.size == 0:
            return

        resized = cv.resize(
            crop, (self.crop_size, self.crop_size), interpolation=cv.INTER_LINEAR
        )

        cropped_msg = self.bridge.cv2_to_imgmsg(resized, encoding='rgb8')
        cropped_msg.header = src_msg.header
        self.image_pub.publish(cropped_msg)

        roi_msg = self.make_roi_msg(
            x=x1,
            y=y1,
            width=x2 - x1,
            height=y2 - y1,
            stamp=src_msg.header.stamp,
            frame_id=src_msg.header.frame_id,
        )
        self.roi_pub.publish(roi_msg)

        self.last_crop_box = crop_box

    def parse_yolo_output(self, tensor) -> Optional[np.ndarray]:
        dims = tuple(int(x) for x in tensor.shape.dims)
        data = np.frombuffer(tensor.data, dtype=np.float32)

        if np.prod(dims) != data.size:
            raise ValueError(f'Shape {dims} does not match data length {data.size}')

        out = data.reshape(dims)

        if len(dims) != 3 or dims[0] != 1 or dims[1] != 5:
            raise ValueError(f'Unexpected output shape: {dims}, expected (1, 5, N)')

        pred = out[0]      # (5, N)
        conf = pred[4, :]  # (N,)

        best_idx = int(np.argmax(conf))
        best_conf = float(conf[best_idx])
        self.get_logger().info(f"{best_conf}")

        if best_conf < self.score_threshold:
            return None

        return pred[:, best_idx].copy()  # [cx, cy, w, h, conf]

    def model_box_to_source_box(
        self,
        pred: np.ndarray,
        src_w: int,
        src_h: int,
        model_size: int,
    ) -> Tuple[int, int, int, int]:
        """
        Convert YOLO box from 640x640 letterboxed coordinates back to original image.

        Example:
          original 1920x960 -> resized to 640x320 -> pad top/bottom to 640x640
        """
        cx, cy, bw, bh, _ = pred.astype(np.float32)

        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0

        scale = min(float(model_size) / float(src_w), float(model_size) / float(src_h))
        new_w = float(src_w) * scale
        new_h = float(src_h) * scale
        pad_x = (float(model_size) - new_w) / 2.0
        pad_y = (float(model_size) - new_h) / 2.0

        x1 -= pad_x
        x2 -= pad_x
        y1 -= pad_y
        y2 -= pad_y

        x1 = np.clip(x1, 0.0, new_w)
        x2 = np.clip(x2, 0.0, new_w)
        y1 = np.clip(y1, 0.0, new_h)
        y2 = np.clip(y2, 0.0, new_h)

        x1 /= scale
        x2 /= scale
        y1 /= scale
        y2 /= scale

        x1 = int(np.clip(x1, 0, src_w - 1))
        y1 = int(np.clip(y1, 0, src_h - 1))
        x2 = int(np.clip(x2, 0, src_w))
        y2 = int(np.clip(y2, 0, src_h))

        return x1, y1, x2, y2

    def is_detection_valid(
        self,
        det_box: Tuple[int, int, int, int],
        src_w: int,
    ) -> bool:
        x1, y1, x2, y2 = det_box

        if x2 <= x1 or y2 <= y1:
            return False

        cx = 0.5 * (x1 + x2)

        ratio = float(self.center_region_width_ratio)
        ratio = max(0.0, min(1.0, ratio))

        allowed_w = src_w * ratio
        allowed_x1 = 0.5 * (src_w - allowed_w)
        allowed_x2 = 0.5 * (src_w + allowed_w)

        return allowed_x1 <= cx <= allowed_x2

    def make_fixed_square_crop_box(
        self,
        cx: float,
        cy: float,
        side: int,
        src_w: int,
        src_h: int,
    ) -> Tuple[int, int, int, int]:
        side = max(2, int(side))
        side = min(side, src_w, src_h)

        x1 = cx - side / 2.0
        y1 = cy - side / 2.0
        x2 = cx + side / 2.0
        y2 = cy + side / 2.0

        if x1 < 0.0:
            shift = -x1
            x1 += shift
            x2 += shift
        if y1 < 0.0:
            shift = -y1
            y1 += shift
            y2 += shift
        if x2 > float(src_w):
            shift = x2 - float(src_w)
            x1 -= shift
            x2 -= shift
        if y2 > float(src_h):
            shift = y2 - float(src_h)
            y1 -= shift
            y2 -= shift

        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        x2 = min(float(src_w), x2)
        y2 = min(float(src_h), y2)

        return (
            int(round(x1)),
            int(round(y1)),
            int(round(x2)),
            int(round(y2)),
        )

    def make_roi_msg(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        stamp,
        frame_id: str,
    ) -> CameraInfo:
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


def main(args=None):
    rclpy.init(args=args)
    node = HandboxDecoder()
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