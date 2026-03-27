#!/usr/bin/env python3

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Sequence

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from isaac_ros_tensor_list_interfaces.msg import Tensor, TensorList, TensorShape


class LiveGestureTensorPublisher(Node):
    def __init__(self) -> None:
        super().__init__('live_gesture_tensor_publisher')

        self.declare_parameter('button_topic', '/arduino_buttons')
        self.declare_parameter('hand_marker_topic', '/pose_markers')
        self.declare_parameter('body_marker_topic', '/fullbody_pose_markers')
        self.declare_parameter('tensor_topic', '/gesture_input_tensor')

        # Must match ONNX input shapes:
        # body: [batch, 64, 8, 2]
        # hand: [batch, 64, 21, 2]
        self.declare_parameter('sequence_length', 64)
        self.declare_parameter('hand_landmarks', 21)

        # Must match the body keypoints used during training
        self.declare_parameter('body_indices', [5, 6,8,10,11,12])

        self.button_topic = str(self.get_parameter('button_topic').value)
        self.hand_marker_topic = str(self.get_parameter('hand_marker_topic').value)
        self.body_marker_topic = str(self.get_parameter('body_marker_topic').value)
        self.tensor_topic = str(self.get_parameter('tensor_topic').value)

        self.sequence_length = int(self.get_parameter('sequence_length').value)
        self.hand_landmarks = int(self.get_parameter('hand_landmarks').value)
        self.body_indices = list(self.get_parameter('body_indices').value)
        self.body_landmarks=len(self.body_indices)


        self._recording = False
        self._hand_buffer: Deque[np.ndarray] = deque(maxlen=self.sequence_length)
        self._body_buffer: Deque[np.ndarray] = deque(maxlen=self.sequence_length)

        self.button_sub = self.create_subscription(
            String, self.button_topic, self.button_callback, 10
        )
        self.hand_marker_sub = self.create_subscription(
            MarkerArray, self.hand_marker_topic, self.hand_marker_callback, 10
        )
        self.body_marker_sub = self.create_subscription(
            MarkerArray, self.body_marker_topic, self.body_marker_callback, 10
        )

        self.tensor_pub = self.create_publisher(TensorList, self.tensor_topic, 10)

        self.get_logger().info(
            'Live gesture tensor publisher ready. '
            f'button={self.button_topic}, '
            f'hand_markers={self.hand_marker_topic}, '
            f'body_markers={self.body_marker_topic}, '
            f'tensor_topic={self.tensor_topic}, '
            f'seq_len={self.sequence_length}, '
            f'body_landmarks={self.body_landmarks}, '
            f'hand_landmarks={self.hand_landmarks}, '
            f'body_indices={self.body_indices}'
        )

    def _clear_buffers(self) -> None:
        self._hand_buffer.clear()
        self._body_buffer.clear()

    def button_callback(self, msg: String) -> None:
        data = msg.data.strip()

        if data == 'button_1_pressed':
            if self._recording == False:
                self._recording = True
                self._clear_buffers()
                self.get_logger().info('Started inference capture. Buffers cleared.')

        elif data == 'button_1_released':
            if not self._recording:
                self.get_logger().warn('Button released but capture was not active.')
                return

            self.get_logger().info(
                f'Button released. Collected '
                f'{len(self._body_buffer)} body frames and '
                f'{len(self._hand_buffer)} hand frames.'
            )

            self.publish_once()
            self._recording = False
            self._clear_buffers()
            self.get_logger().info('Inference capture ended. Buffers cleared.')

        elif data == 'button_2_pressed':
            self._recording = False
            self._clear_buffers()
            self.get_logger().info('Capture cancelled. Buffers cleared.')

    def _extract_keypoints_only(self, msg: MarkerArray) -> Optional[np.ndarray]:
        """
        Extract keypoints from a MarkerArray.

        Supports two common formats:
        1. One POINTS marker containing all keypoints in marker.points
        2. Multiple single-keypoint markers using marker.pose.position

        Returns:
            np.ndarray of shape [N, 2], or None if nothing usable is found.
        """
        if not msg.markers:
            return None

        # -----------------------------------------
        # Case 1: a POINTS marker stores all keypoints
        # -----------------------------------------
        for marker in msg.markers:
            if marker.type == Marker.POINTS and len(marker.points) > 0:
                frame_points = [[p.x, p.y] for p in marker.points]
                return np.asarray(frame_points, dtype=np.float32)

        # -----------------------------------------
        # Case 2: individual markers store keypoints
        # -----------------------------------------
        keypoint_markers = []

        for marker in msg.markers:
            if len(marker.points) > 0:
                continue

            if marker.type not in (
                Marker.SPHERE,
                Marker.CUBE,
                Marker.CYLINDER,
                Marker.ARROW,
            ):
                continue

            keypoint_markers.append(marker)

        if len(keypoint_markers) == 0:
            return None

        keypoint_markers.sort(key=lambda m: m.id)

        frame_points = []
        for marker in keypoint_markers:
            pos = marker.pose.position
            frame_points.append([pos.x, pos.y])

        return np.asarray(frame_points, dtype=np.float32)

    def _select_body_landmarks(self, frame_array: np.ndarray) -> Optional[np.ndarray]:
        if frame_array.ndim != 2 or frame_array.shape[1] != 2:
            self.get_logger().warn(f'Unexpected body frame shape: {frame_array.shape}')
            return None

        max_idx = max(self.body_indices)
        if frame_array.shape[0] <= max_idx:
            self.get_logger().warn(
                f'Body frame has only {frame_array.shape[0]} landmarks, '
                f'but body_indices requires index {max_idx}.'
            )
            return None
        #self.get_logger().info(frame_array.shape)
        selected = frame_array[self.body_indices, :]
        #self.get_logger().info(selected.shape)

        if selected.shape != (self.body_landmarks, 2):
            self.get_logger().warn(
                f'Selected body frame shape {selected.shape} does not match '
                f'expected ({self.body_landmarks}, 2).'
            )
            return None

        return selected.astype(np.float32)

    def hand_marker_callback(self, msg: MarkerArray) -> None:
        if not self._recording:
            return

        frame_array = self._extract_keypoints_only(msg)
        if frame_array is None:
            return

        if frame_array.shape[0] != self.hand_landmarks:
            self.get_logger().warn(
                f'Skipping hand frame with {frame_array.shape[0]} landmarks; '
                f'expected {self.hand_landmarks}.'
            )
            return

        self._hand_buffer.append(frame_array)

    def body_marker_callback(self, msg: MarkerArray) -> None:
        if not self._recording:
            return

        frame_array = self._extract_keypoints_only(msg)
        
        if frame_array is None:

            return

        selected = self._select_body_landmarks(frame_array)

        if selected is None:
            return

        self._body_buffer.append(selected)

    @staticmethod
    def _compute_contiguous_strides(dims: Sequence[int]) -> List[int]:
        strides = [1] * len(dims)
        running = 1
        for i in range(len(dims) - 1, -1, -1):
            strides[i] = running
            running *= dims[i]
        return strides

    def _make_tensor(self, name: str, arr: np.ndarray) -> Tensor:
        arr = np.ascontiguousarray(arr.astype(np.float32))
        dims = list(arr.shape)

        tensor = Tensor()
        tensor.name = name

        tensor.shape = TensorShape()
        tensor.shape.rank = len(dims)
        tensor.shape.dims = dims

        tensor.data_type = 9  # float32
        tensor.strides = self._compute_contiguous_strides(dims)
        tensor.data = arr.tobytes()

        return tensor

    def _build_window(
        self,
        buffer: Deque[np.ndarray],
        expected_shape: tuple[int, int],
        tensor_name: str,
    ) -> Optional[np.ndarray]:
        """
        Always returns [1, sequence_length, K, 2].

        - If buffer has more than sequence_length frames:
          keep the most recent sequence_length.
        - If buffer has fewer than sequence_length frames:
          pad the remaining tail with zeros.
        - If buffer is empty:
          return an all-zero window.
        """
        k, d = expected_shape
        out = np.zeros((self.sequence_length, k, d), dtype=np.float32)

        if len(buffer) == 0:
            self.get_logger().warn(
                f'No frames collected for {tensor_name}. Publishing all-zero window.'
            )
        else:
            frames = list(buffer)[-self.sequence_length:]
            valid = np.stack(frames, axis=0).astype(np.float32)
            out[:valid.shape[0]] = valid

            if valid.shape[0] < self.sequence_length:
                self.get_logger().info(
                    f'{tensor_name} collected {valid.shape[0]}/{self.sequence_length} frames; '
                    f'zero-padded remaining {self.sequence_length - valid.shape[0]} frames.'
                )

        if out.shape != (self.sequence_length, k, d):
            self.get_logger().warn(
                f'Unexpected {tensor_name} window shape {out.shape}; '
                f'expected {(self.sequence_length, k, d)}.'
            )
            return None

        out = np.expand_dims(out, axis=0)  # [1, T, K, 2]
        return out

    def publish_once(self) -> None:
        body_arr = self._build_window(
            self._body_buffer,
            (self.body_landmarks, 2),
            'body',
        )
        hand_arr = self._build_window(
            self._hand_buffer,
            (self.hand_landmarks, 2),
            'hand',
        )

        if body_arr is None or hand_arr is None:
            self.get_logger().warn('Did not publish inference tensor due to tensor build failure.')
            return

        msg = TensorList()
        msg.tensors = [
            self._make_tensor('body', body_arr),   # [1, 64, 8, 2]
            self._make_tensor('hand', hand_arr),   # [1, 64, 21, 2]
        ]
        #self.get_logger().info(body_arr.shape)
       

        self.tensor_pub.publish(msg)

        self.get_logger().info(
            f'Published one inference tensor: '
            f'body={body_arr.shape}, hand={hand_arr.shape}'
        )

    def destroy_node(self):
        self._clear_buffers()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LiveGestureTensorPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()