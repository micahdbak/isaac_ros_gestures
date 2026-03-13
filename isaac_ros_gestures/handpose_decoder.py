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

HAND_KEYPOINT_NAMES = [
    'wrist',
    'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
    'index_mcp', 'index_pip', 'index_dip', 'index_tip',
    'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
    'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
    'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'
]

class HandposeDecoder(Node):
    def __init__(self):
        super().__init__('handpose_decoder')
        
        self.declare_parameter('score_threshold', 0.25)
        self.declare_parameter('network_size', 640.0)

        # Coordinate frame for output markers:
        # x/y in [-1, 1] relative to the *original* 960x960 crop of the source image.
        # -1 is the left/top edge of that crop, +1 is the right/bottom edge.
        self.declare_parameter('source_crop_size', 960.0)
        # If the original source image is 1920x960 and you center-crop to 960x960,
        # the crop origin is x=480, y=0 in the original image.
        self.declare_parameter('source_crop_origin_x', 480.0)
        self.declare_parameter('source_crop_origin_y', 0.0)
        
        self.score_threshold = self.get_parameter('score_threshold').value
        self.network_size = self.get_parameter('network_size').value
        self.source_crop_size = float(self.get_parameter('source_crop_size').value)
        self.source_crop_origin_x = float(self.get_parameter('source_crop_origin_x').value)
        self.source_crop_origin_y = float(self.get_parameter('source_crop_origin_y').value)

        # Cache latest ROI (and a small timestamp map if TensorList carries headers).
        self._last_roi: Optional[Tuple[float, float, float, float]] = None  # x,y,w,h in source image px
        self._roi_cache: Dict[int, Tuple[float, float, float, float]] = {}

        self.tensor_sub = self.create_subscription(
            TensorList,
            'tensor_output',
            self.tensor_callback,
            10
        )

        self.roi_sub = self.create_subscription(
            CameraInfo,
            'palm_roi',
            self.roi_callback,
            10
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            'pose_markers',
            10
        )

        # from sensor_msgs.msg import Image
        # import cv2
        # from cv_bridge import CvBridge
        # self.cv2 = cv2
        # self.bridge = CvBridge()
        # self.current_image = None
        # self.image_sub = self.create_subscription(
        #     Image,
        #     '/image_cropped',
        #     self.image_callback,
        #     10
        # )
    
    # def image_callback(self, msg):
    #     # Convert ROS Image to OpenCV
    #     img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    #     img = self.cv2.resize(img, (int(self.network_size), int(self.network_size)))
    #     self.current_image = img

    def tensor_callback(self, msg: TensorList):
        if len(msg.tensors) == 0:
            print("no tensors bruh")
            return

        tensor = msg.tensors[0]
        data = np.frombuffer(tensor.data, dtype=np.float32)
        out = data.reshape(tuple(tensor.shape.dims))
        pred = out[0] # (1,300,69) -> (300,69)
        
        conf = pred[:, 4]
        best_idx = np.argmax(conf)
        best_pred = pred[best_idx]

        if float(conf[best_idx]) < float(self.score_threshold):
            # Nothing confident enough to visualize.
            self.marker_pub.publish(MarkerArray())
            return

        kpts = best_pred[6:].reshape(21, 3)
        palm_roi = self._get_roi_for_tensorlist(msg)
        self.publish_keypoints(kpts, palm_roi)

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
        # Keypoints are predicted in the resized network input (e.g. 640x640) corresponding to the palm crop.
        # Map to the original 960x960 source crop, then normalize to [-1, 1].
        nx = float(x) / float(self.network_size)
        ny = float(y) / float(self.network_size)

        if roi_xywh is None:
            # Best-effort fallback: treat network input as spanning the full 960x960 crop.
            x_px = nx * self.source_crop_size
            y_px = ny * self.source_crop_size
        else:
            roi_x, roi_y, roi_w, roi_h = roi_xywh

            # Convert ROI from source-image coordinates into the 960x960 crop coordinates.
            roi_x = float(roi_x) - self.source_crop_origin_x
            roi_y = float(roi_y) - self.source_crop_origin_y

            # Clamp ROI into the crop bounds to avoid runaway values when the upstream crop is near edges.
            roi_w = max(1.0, min(float(roi_w), self.source_crop_size))
            roi_h = max(1.0, min(float(roi_h), self.source_crop_size))
            roi_x = max(0.0, min(float(roi_x), self.source_crop_size - roi_w))
            roi_y = max(0.0, min(float(roi_y), self.source_crop_size - roi_h))

            x_px = roi_x + nx * roi_w
            y_px = roi_y + ny * roi_h

        x_out = (x_px / self.source_crop_size) * 2.0 - 1.0
        y_out = (y_px / self.source_crop_size) * 2.0 - 1.0
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
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)

        # Overlay markers on image and show in OpenCV window
        # if self.current_image is not None:
        #     img = self.current_image.copy()
        #     for i, (x, y, v) in enumerate(kpts):
        #         # Optionally skip invisible keypoints
        #         # if v < 0.5:
        #         #     continue
        #         cx = int(float(x))
        #         cy = int(float(y))
        #         self.cv2.circle(img, (cx, cy), 6, (0, 255, 0), -1)
        #     self.cv2.imshow('Hand Keypoints Overlay', img)
        #     self.cv2.waitKey(1)

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
