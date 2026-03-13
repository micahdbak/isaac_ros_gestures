#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from isaac_ros_tensor_list_interfaces.msg import TensorList
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo
from typing import Optional, Tuple, Dict

class HandposeDecoder(Node):
    def __init__(self):
        super().__init__('handpose_decoder')
        
        self.score_threshold = 0.05
        self.network_size = 640.0
        self.source_crop_width = 1920.0
        self.source_crop_height = 960.0
        self.source_crop_x = 480.0
        self.source_crop_y = 0.0

        self._last_roi: Optional[Tuple[float, float, float, float]] = None
        self._roi_cache: Dict[int, Tuple[float, float, float, float]] = {}

        self.tensor_sub = self.create_subscription(TensorList, 'tensor_output', self.tensor_callback, 10)
        self.roi_sub = self.create_subscription(CameraInfo, 'image_roi', self.roi_callback, 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'pose_markers', 10)

    def tensor_callback(self, msg: TensorList):
        if len(msg.tensors) == 0:
            return

        tensor = msg.tensors[0]
        data = np.frombuffer(tensor.data, dtype=np.float32)
        out = data.reshape(tuple(tensor.shape.dims))
        pred = out[0] # (1,300,69) -> (300,69)
        
        conf = pred[:, 4]
        best_idx = int(np.argmax(conf))
        best_pred = pred[best_idx]

        if float(conf[best_idx]) < float(self.score_threshold):
            # Nothing confident enough to visualize.
            self.marker_pub.publish(MarkerArray())
            return

        kpts = best_pred[6:].reshape(21, 3)
        image_roi = self._get_roi_for_tensorlist(msg)
        self.publish_keypoints(kpts, image_roi)

    def _stamp_to_ns(self, stamp) -> Optional[int]:
        try:
            return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        except Exception:
            return None

    def roi_callback(self, msg: CameraInfo):
        roi = msg.roi
        if roi.width <= 0 or roi.height <= 0:
            return

        x = float(roi.x_offset)
        y = float(roi.y_offset)
        w = float(roi.width)
        h = float(roi.height)
        self._last_roi = (x, y, w, h)

        stamp_ns = self._stamp_to_ns(getattr(getattr(msg, 'header', None), 'stamp', None))
        if stamp_ns is not None:
            self._roi_cache[stamp_ns] = (x, y, w, h)
            # prune cache to keep bounded
            if len(self._roi_cache) > 60:
                for key in sorted(self._roi_cache.keys())[:-60]:
                    del self._roi_cache[key]

    def _get_roi_for_tensorlist(self, msg: TensorList) -> Optional[Tuple[float, float, float, float]]:
        stamp_ns = self._stamp_to_ns(getattr(getattr(msg, 'header', None), 'stamp', None))
        if stamp_ns is not None and stamp_ns in self._roi_cache:
            return self._roi_cache[stamp_ns]
        return self._last_roi

    def _kpt_to_crop_norm(self, x: float, y: float, roi_xywh: Optional[Tuple[float, float, float, float]]):
        nx = float(x) / float(self.network_size) # 0..1
        ny = float(y) / float(self.network_size) # 0..1

        if roi_xywh is None:
            x_px = nx * self.source_crop_width
            y_px = ny * self.source_crop_height
        else:
            roi_x, roi_y, roi_w, roi_h = roi_xywh
            x_px = roi_x + nx * roi_w
            y_px = roi_y + ny * roi_h

        x_out = (x_px / self.source_crop_width) * 2.0 - 1.0
        y_out = (y_px / self.source_crop_height) * 2.0 - 1.0
        x_out = max(-1.0, min(1.0, x_out))
        y_out = max(-1.0, min(1.0, y_out))
        return x_out, y_out

    def publish_keypoints(self, kpts, roi_xywh: Optional[Tuple[float, float, float, float]] = None):
        marker_array = MarkerArray()

        for i, (x, y, v) in enumerate(kpts):
            x_out, y_out = self._kpt_to_crop_norm(x, y, roi_xywh)
            marker = Marker()
            marker.header.frame_id = "theta_camera"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "hand_keypoints"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.02
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            marker.pose.position.x = float(x_out)
            marker.pose.position.y = float(y_out)
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.lifetime.sec = 0
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = HandposeDecoder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
