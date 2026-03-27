#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional, Tuple, List

import cv2 as cv
import numpy as np
import rclpy
from cv_bridge import CvBridge
from isaac_ros_tensor_list_interfaces.msg import TensorList
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, RegionOfInterest
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

class FullBodyPoseDecoder(Node):
    def __init__(self):
        super().__init__('full_body_pose_decoder')

        self.declare_parameter('image_topic', 'image_raw')
        self.declare_parameter('tensor_topic', 'tensor_output_handbox')
        self.declare_parameter('cropped_topic', 'image_cropped')
        self.declare_parameter('roi_topic', 'image_roi')
        self.declare_parameter('marker_topic', 'fullbody_pose_markers')

        self.declare_parameter('score_threshold', 0.10)
        self.declare_parameter('keypoint_threshold', 0.10)
        self.declare_parameter('model_size', 640)
        self.declare_parameter('crop_size', 640)

        # Fixed crop size in source-image pixels
        self.declare_parameter('fixed_box_size', 180)

        # Keep only detections whose wrist x lies inside [roi_left_bound, roi_right_bound]
        # Negative values mean "use image boundary"
        self.declare_parameter('roi_left_bound', 400)
        self.declare_parameter('roi_right_bound', 1500)

        # COCO full-body pose convention: right wrist = 10
        self.declare_parameter('right_wrist_kpt_index', 10)

        # Visualization
        self.declare_parameter('show_visualization', True)
        self.declare_parameter('vis_window_name', 'full_body_pose_overlay')
        self.declare_parameter('max_center_jump_px', 120.0)

        self.marker_topic = str(self.get_parameter('marker_topic').value)
        self.max_center_jump_px = float(self.get_parameter('max_center_jump_px').value)
        self.image_topic = str(self.get_parameter('image_topic').value)
        self.tensor_topic = str(self.get_parameter('tensor_topic').value)
        self.cropped_topic = str(self.get_parameter('cropped_topic').value)
        self.roi_topic = str(self.get_parameter('roi_topic').value)

        self.score_threshold = float(self.get_parameter('score_threshold').value)
        self.keypoint_threshold = float(self.get_parameter('keypoint_threshold').value)
        self.model_size = int(self.get_parameter('model_size').value)
        self.crop_size = int(self.get_parameter('crop_size').value)
        self.fixed_box_size = int(self.get_parameter('fixed_box_size').value)
        self.roi_left_bound = int(self.get_parameter('roi_left_bound').value)
        self.roi_right_bound = int(self.get_parameter('roi_right_bound').value)
        self.right_wrist_kpt_index = int(self.get_parameter('right_wrist_kpt_index').value)

        self.show_visualization = bool(self.get_parameter('show_visualization').value)
        self.vis_window_name = str(self.get_parameter('vis_window_name').value)

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
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)

        # COCO-17 skeleton connections
        self.skeleton_edges = [
            (0, 1), (0, 2),
            (1, 3), (2, 4),
            (5, 6),
            (5, 7), (7, 9),
            (6, 8), (8, 10),
            (5, 11), (6, 12),
            (11, 12),
            (11, 13), (13, 15),
            (12, 14), (14, 16),
        ]

        if self.show_visualization:
            cv.namedWindow(self.vis_window_name, cv.WINDOW_NORMAL)

        self.get_logger().info(
            'FullBodyPoseDecoder started (tracking right wrist from pose output)'
        )
        self.last_person_center: Optional[Tuple[float, float]] = None



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
        published_markers = False
        src_img = self.last_image
        src_msg = self.last_image_msg
        src_h, src_w = src_img.shape[:2]

        crop_box: Optional[Tuple[int, int, int, int]] = None
        best_kpts_src: Optional[np.ndarray] = None
        wrist_point_src: Optional[Tuple[float, float]] = None
        bbox_center_src: Optional[Tuple[float, float]] = None
        accepted_bbox_center_src: Optional[Tuple[float, float]] = None
        best_score: Optional[float] = None

        rejected_for_roi = 0
        rejected_for_jump = 0
        rejected_for_kpt = 0
        rejected_for_score = 0

        if msg.tensors:
            try:
                pred, scores = self.parse_pose_output(msg.tensors[0])
            except Exception as e:
                self.get_logger().error(f'Failed to parse tensor output: {e}')
                pred, scores = None, None

            if pred is not None and scores is not None and pred.shape[0] > 0:
                sorted_indices = np.argsort(-scores)  # high to low

                for idx in sorted_indices:
                    cand = pred[int(idx)]
                    det_conf = float(cand[4])

                    if det_conf < self.score_threshold:
                        rejected_for_score += 1
                        continue

                    try:
                        (
                            box_cx_model,
                            box_cy_model,
                            box_w_model,
                            box_h_model,
                            wrist_x_model,
                            wrist_y_model,
                            wrist_conf,
                            kpts_model,
                        ) = self.candidate_to_data(cand)
                    except Exception as e:
                        self.get_logger().warn(f'Failed to parse candidate {int(idx)}: {e}')
                        continue

                    if wrist_conf < self.keypoint_threshold:
                        rejected_for_kpt += 1
                        continue

                    bbox_center_x_src, bbox_center_y_src = self.model_box_center_to_source_point(
                        cx=box_cx_model,
                        cy=box_cy_model,
                        src_w=src_w,
                        src_h=src_h,
                        model_size=self.model_size,
                    )

                    if not self.is_point_inside_roi(bbox_center_x_src, src_w):
                        rejected_for_roi += 1
                        continue

                    if not self.is_center_jump_acceptable(
                        new_center=(bbox_center_x_src, bbox_center_y_src),
                        max_jump_px=self.max_center_jump_px,
                    ):
                        rejected_for_jump += 1
                        continue

                    wrist_x_src, wrist_y_src = self.model_point_to_source_point(
                        x=wrist_x_model,
                        y=wrist_y_model,
                        src_w=src_w,
                        src_h=src_h,
                        model_size=self.model_size,
                    )

                    best_kpts_src = self.model_keypoints_to_source_keypoints(
                        kpts_model=kpts_model,
                        src_w=src_w,
                        src_h=src_h,
                        model_size=self.model_size,
                    )
                    self.publish_pose_markers(
                        kpts=best_kpts_src,
                        stamp=src_msg.header.stamp,
                        frame_id=src_msg.header.frame_id,
                        kpt_threshold=self.keypoint_threshold,
                    )
                    published_markers = True
                    bbox_center_src = (bbox_center_x_src, bbox_center_y_src)
                    accepted_bbox_center_src = bbox_center_src
                    wrist_point_src = (wrist_x_src, wrist_y_src)
                    best_score = det_conf

                    crop_box = self.make_fixed_square_crop_box(
                        cx=wrist_x_src,
                        cy=wrist_y_src,
                        side=self.fixed_box_size,
                        src_w=src_w,
                        src_h=src_h,
                    )

                    #self.get_logger().info(
                    #    f'Accepted candidate idx={int(idx)} '
                    #    f'conf={det_conf:.3f} '
                    #    f'bbox_center=({bbox_center_x_src:.1f}, {bbox_center_y_src:.1f}) '
                    #    f'wrist=({wrist_x_src:.1f}, {wrist_y_src:.1f})'
                    #)
                    break

        # Fallback to previous crop box if current detection is unusable
        if crop_box is None:
            crop_box = self.last_crop_box
        if not published_markers:
            self.clear_pose_markers(
                stamp=src_msg.header.stamp,
                frame_id=src_msg.header.frame_id,
            )
        # Visualization
        if self.show_visualization:
            vis = src_img.copy()

            left = self.roi_left_bound if self.roi_left_bound >= 0 else 0
            right = self.roi_right_bound if self.roi_right_bound >= 0 else (src_w - 1)
            left = max(0, min(left, src_w - 1))
            right = max(0, min(right, src_w - 1))
            if left > right:
                left, right = right, left

            cv.line(vis, (left, 0), (left, src_h - 1), (255, 255, 0), 2)
            cv.line(vis, (right, 0), (right, src_h - 1), (255, 255, 0), 2)
            cv.putText(
                vis,
                f'ROI x:[{left}, {right}]',
                (20, 60),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv.LINE_AA,
            )

            if best_kpts_src is not None:
                self.draw_pose_overlay(
                    image=vis,
                    kpts=best_kpts_src,
                    right_wrist_idx=self.right_wrist_kpt_index,
                    kpt_threshold=self.keypoint_threshold,
                )

            if bbox_center_src is not None:
                bx, by = bbox_center_src
                cv.circle(vis, (int(round(bx)), int(round(by))), 6, (0, 255, 255), -1)
                cv.putText(
                    vis,
                    'BBoxCenter',
                    (int(round(bx)) + 8, int(round(by)) - 8),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                    cv.LINE_AA,
                )

            if self.last_person_center is not None:
                lx, ly = self.last_person_center
                cv.circle(vis, (int(round(lx)), int(round(ly))), 6, (255, 0, 255), -1)
                cv.putText(
                    vis,
                    'LastCenter',
                    (int(round(lx)) + 8, int(round(ly)) - 8),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    2,
                    cv.LINE_AA,
                )

            if wrist_point_src is not None:
                wx, wy = wrist_point_src
                cv.circle(vis, (int(round(wx)), int(round(wy))), 8, (255, 255, 0), 2)

            if crop_box is not None:
                x1, y1, x2, y2 = crop_box
                cv.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

            if best_score is not None:
                cv.putText(
                    vis,
                    f'person conf: {best_score:.3f}',
                    (20, 30),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv.LINE_AA,
                )

            cv.putText(
                vis,
                f'max jump: {self.max_center_jump_px:.1f}px',
                (20, 90),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )

            cv.putText(
                vis,
                f'rej score:{rejected_for_score} kpt:{rejected_for_kpt} roi:{rejected_for_roi} jump:{rejected_for_jump}',
                (20, 120),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
                cv.LINE_AA,
            )

            vis_bgr = cv.cvtColor(vis, cv.COLOR_RGB2BGR)
            cv.imshow(self.vis_window_name, vis_bgr)
            cv.waitKey(1)

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

        if accepted_bbox_center_src is not None:
            self.last_person_center = accepted_bbox_center_src

    def parse_pose_output(self, tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Expected tensor layout:
        shape = (1, N, 57)

        Assumed per detection format:
        [cx, cy, w, h, conf, cls, kpt0_x, kpt0_y, kpt0_conf, ..., kpt16_x, kpt16_y, kpt16_conf]

        Returns:
        pred: (N, 57)
        scores: (N,)
        """
        dims = tuple(int(x) for x in tensor.shape.dims)
        data = np.frombuffer(tensor.data, dtype=np.float32)

        if np.prod(dims) != data.size:
            raise ValueError(f'Shape {dims} does not match data length {data.size}')

        out = data.reshape(dims)

        if len(dims) != 3 or dims[0] != 1 or dims[2] != 57:
            raise ValueError(f'Unexpected output shape: {dims}, expected (1, N, 57)')

        pred = out[0]  # (N, 57)
        scores = pred[:, 4]

        return pred, scores
    def candidate_to_data(
        self,
        cand: np.ndarray,
    ) -> Tuple[float, float, float, float, float, float, float, np.ndarray]:
        """
        Parse one candidate row of shape (57,)

        Returns:
        box_cx, box_cy, box_w, box_h,
        wrist_x, wrist_y, wrist_conf,
        kpts_model (K, 3)
        """
        c = cand.shape[0]
        kpt_start = 6
        num_kpts = (c - kpt_start) // 3

        if num_kpts <= self.right_wrist_kpt_index:
            raise ValueError(
                f'Output only has {num_kpts} keypoints, but right wrist index is {self.right_wrist_kpt_index}'
            )

        box_cx = float(cand[0])
        box_cy = float(cand[1])
        box_w = float(cand[2])
        box_h = float(cand[3])

        base = kpt_start + self.right_wrist_kpt_index * 3
        wrist_x = float(cand[base + 0])
        wrist_y = float(cand[base + 1])
        wrist_conf = float(cand[base + 2])

        kpts_flat = cand[kpt_start:kpt_start + num_kpts * 3]
        kpts_model = kpts_flat.reshape(num_kpts, 3).copy()

        return box_cx, box_cy, box_w, box_h, wrist_x, wrist_y, wrist_conf, kpts_model
    def model_box_center_to_source_point(
        self,
        cx: float,
        cy: float,
        src_w: int,
        src_h: int,
        model_size: int,
    ) -> Tuple[float, float]:
        scale = min(float(model_size) / float(src_w), float(model_size) / float(src_h))
        new_w = float(src_w) * scale
        new_h = float(src_h) * scale
        pad_x = (float(model_size) - new_w) / 2.0
        pad_y = (float(model_size) - new_h) / 2.0

        cx = cx - pad_x
        cy = cy - pad_y

        cx = np.clip(cx, 0.0, new_w) / scale
        cy = np.clip(cy, 0.0, new_h) / scale

        cx = float(np.clip(cx, 0.0, float(src_w - 1)))
        cy = float(np.clip(cy, 0.0, float(src_h - 1)))

        return cx, cy
    def model_point_to_source_point(
        self,
        x: float,
        y: float,
        src_w: int,
        src_h: int,
        model_size: int,
    ) -> Tuple[float, float]:
        scale = min(float(model_size) / float(src_w), float(model_size) / float(src_h))
        new_w = float(src_w) * scale
        new_h = float(src_h) * scale
        pad_x = (float(model_size) - new_w) / 2.0
        pad_y = (float(model_size) - new_h) / 2.0

        x = x - pad_x
        y = y - pad_y

        x = np.clip(x, 0.0, new_w)
        y = np.clip(y, 0.0, new_h)

        x = x / scale
        y = y / scale

        x = float(np.clip(x, 0.0, float(src_w - 1)))
        y = float(np.clip(y, 0.0, float(src_h - 1)))

        return x, y

    def model_keypoints_to_source_keypoints(
        self,
        kpts_model: np.ndarray,
        src_w: int,
        src_h: int,
        model_size: int,
    ) -> np.ndarray:
        """
        kpts_model: (K, 3) with columns [x, y, conf] in model coords
        returns:   (K, 3) with [x_src, y_src, conf]
        """
        scale = min(float(model_size) / float(src_w), float(model_size) / float(src_h))
        new_w = float(src_w) * scale
        new_h = float(src_h) * scale
        pad_x = (float(model_size) - new_w) / 2.0
        pad_y = (float(model_size) - new_h) / 2.0

        kpts_src = kpts_model.copy().astype(np.float32)
        kpts_src[:, 0] = (kpts_src[:, 0] - pad_x).clip(0.0, new_w) / scale
        kpts_src[:, 1] = (kpts_src[:, 1] - pad_y).clip(0.0, new_h) / scale

        kpts_src[:, 0] = np.clip(kpts_src[:, 0], 0.0, float(src_w - 1))
        kpts_src[:, 1] = np.clip(kpts_src[:, 1], 0.0, float(src_h - 1))

        return kpts_src

    def is_point_inside_roi(
        self,
        x: float,
        src_w: int,
    ) -> bool:
        left = int(self.roi_left_bound)
        right = int(self.roi_right_bound)

        if left < 0:
            left = 0
        if right < 0:
            right = src_w - 1

        left = max(0, min(left, src_w - 1))
        right = max(0, min(right, src_w - 1))

        if left > right:
            left, right = right, left

        return left <= x <= right
        
    def draw_pose_overlay(
        self,
        image: np.ndarray,
        kpts: np.ndarray,
        right_wrist_idx: int,
        kpt_threshold: float,
    ) -> None:
        """
        image: RGB image
        kpts: (K, 3) in source coords [x, y, conf]
        """
        num_kpts = kpts.shape[0]

        # draw skeleton
        for i, j in self.skeleton_edges:
            if i >= num_kpts or j >= num_kpts:
                continue

            xi, yi, ci = kpts[i]
            xj, yj, cj = kpts[j]

            if ci >= kpt_threshold and cj >= kpt_threshold:
                pt1 = (int(round(xi)), int(round(yi)))
                pt2 = (int(round(xj)), int(round(yj)))
                cv.line(image, pt1, pt2, (0, 255, 0), 2)

        # draw keypoints
        for idx in range(num_kpts):
            if idx not in [5,6,8,10,11 ,12]:
                continue
            x, y, c = kpts[idx]
            if c < kpt_threshold:
                continue

            pt = (int(round(x)), int(round(y)))

            if idx == right_wrist_idx:
                cv.circle(image, pt, 7, (255, 0, 0), -1)
                cv.putText(
                    image,
                    'RWrist',
                    (pt[0] + 8, pt[1] - 8),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                    cv.LINE_AA,
                )
            else:
                cv.circle(image, pt, 4, (0, 0, 255), -1)

    def is_point_valid(
        self,
        x: float,
        src_w: int,
    ) -> bool:
        ratio = float(self.center_region_width_ratio)
        ratio = max(0.0, min(1.0, ratio))

        allowed_w = src_w * ratio
        allowed_x1 = 0.5 * (src_w - allowed_w)
        allowed_x2 = 0.5 * (src_w + allowed_w)

        return allowed_x1 <= x <= allowed_x2

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

    def is_center_jump_acceptable(
        self,
        new_center: Tuple[float, float],
        max_jump_px: float,
    ) -> bool:
        if self.last_person_center is None:
            return True

        old_x, old_y = self.last_person_center
        new_x, new_y = new_center

        dist = float(np.hypot(new_x - old_x, new_y - old_y))
        return dist <= max_jump_px

    def destroy_node(self):
        if self.show_visualization:
            try:
                cv.destroyWindow(self.vis_window_name)
            except Exception:
                pass
        super().destroy_node()
    def publish_pose_markers(
        self,
        kpts: np.ndarray,
        stamp,
        frame_id: str,
        kpt_threshold: float,
    ) -> None:
        """
        Publish full body pose as a single Marker containing all landmark points
        in fixed order. Low-confidence points are stored as (-1, -1, 0).
        """

        marker_array = MarkerArray()

        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = frame_id
        marker.ns = 'fullbody_keypoints'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 8.0
        marker.scale.y = 8.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for idx in range(kpts.shape[0]):
            x, y, conf = kpts[idx]

            if conf < kpt_threshold:
                p = self.make_marker_point(-1.0, -1.0, 0.0)
            else:
                p = self.make_marker_point(float(x), float(y), 0.0)

            marker.points.append(p)

        marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)
    def clear_pose_markers(
        self,
        stamp,
        frame_id: str,
    ) -> None:
        marker_array = MarkerArray()

        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = frame_id
        m.action = Marker.DELETEALL

        marker_array.markers.append(m)
        self.marker_pub.publish(marker_array)
    
    def make_marker_point(self, x: float, y: float, z: float = 0.0) -> Point:
        p = Point()
        p.x = x
        p.y = y
        p.z = z
        return p
def main(args=None):
    rclpy.init(args=args)
    node = FullBodyPoseDecoder()
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