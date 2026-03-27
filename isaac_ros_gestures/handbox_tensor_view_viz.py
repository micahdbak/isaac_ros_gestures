#!/usr/bin/env python3

"""Visualize the encoded handbox detector input tensor.

Subscribes to a TensorList (from DnnImageEncoderNode) on `tensor_view_handbox`,
decodes the first tensor into an RGB image, overlays basic stats, and publishes
as a sensor_msgs/Image.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2 as cv
import numpy as np
import rclpy
from cv_bridge import CvBridge
from isaac_ros_tensor_list_interfaces.msg import TensorList
from rclpy.node import Node
from sensor_msgs.msg import Image


class HandboxTensorViewViz(Node):
    def __init__(self):
        super().__init__('handbox_tensor_view_viz')

        self.declare_parameter('input_topic', 'tensor_view_handbox')
        self.declare_parameter('output_topic', 'handbox_detector_input_image')
        self.declare_parameter('overlay_text', True)

        input_topic = str(self.get_parameter('input_topic').value)
        output_topic = str(self.get_parameter('output_topic').value)
        self._overlay_text = bool(self.get_parameter('overlay_text').value)

        self._bridge = CvBridge()

        self._sub = self.create_subscription(TensorList, input_topic, self._callback, 10)
        self._pub = self.create_publisher(Image, output_topic, 10)

        self.get_logger().info(f'Subscribing: {input_topic} -> Publishing: {output_topic}')

    def _infer_layout(self, dims: Sequence[int]) -> Optional[Tuple[int, int, int, str]]:
        if len(dims) != 4:
            return None

        n, d1, d2, d3 = (int(dims[0]), int(dims[1]), int(dims[2]), int(dims[3]))
        if n <= 0:
            return None

        if d1 == 3:
            return d2, d3, 3, 'NCHW'
        if d3 == 3:
            return d1, d2, 3, 'NHWC'

        return None

    def _tensor_to_rgb8(self, tensor) -> Optional[np.ndarray]:
        dims = list(getattr(getattr(tensor, 'shape', None), 'dims', []))
        layout = self._infer_layout(dims)
        if layout is None:
            self.get_logger().warn(f'Unsupported tensor dims: {dims}')
            return None

        h, w, c, layout_name = layout
        if c != 3:
            self.get_logger().warn(f'Unsupported channels: {c}')
            return None

        data = getattr(tensor, 'data', b'')
        if not data:
            return None

        expected_elems = int(np.prod(dims))
        if expected_elems <= 0:
            return None

        if len(data) == expected_elems * 4:
            arr = np.frombuffer(data, dtype=np.float32)
        elif len(data) == expected_elems:
            arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
        else:
            self.get_logger().warn(f'Unexpected tensor byte length: {len(data)} for dims {dims}')
            return None

        arr = arr.reshape(tuple(dims))
        arr = arr[0]

        if layout_name == 'NCHW':
            arr = np.transpose(arr, (1, 2, 0))

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

    def _tensor_min_max(self, tensor) -> Tuple[float, float]:
        data = getattr(tensor, 'data', b'')
        if not data:
            return 0.0, 0.0

        if len(data) % 4 == 0:
            arr = np.frombuffer(data, dtype=np.float32)
        else:
            arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)

        if arr.size == 0:
            return 0.0, 0.0
        if not np.isfinite(arr).all():
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return float(np.min(arr)), float(np.max(arr))

    def _callback(self, msg: TensorList):
        if len(msg.tensors) == 0:
            return

        rgb = self._tensor_to_rgb8(msg.tensors[0])
        if rgb is None:
            return

        if self._overlay_text:
            bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
            dims = list(getattr(getattr(msg.tensors[0], 'shape', None), 'dims', []))
            vmin, vmax = self._tensor_min_max(msg.tensors[0])

            text = f'dims={dims}  min={vmin:.3f}  max={vmax:.3f}'
            cv.putText(
                bgr,
                text,
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv.LINE_AA,
            )
            rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)

        out_msg = self._bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
        out_msg.header = getattr(msg, 'header', out_msg.header)
        self._pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = HandboxTensorViewViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
