#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

from isaac_ros_tensor_list_interfaces.msg import TensorList


class ClassifierOutputDecoder(Node):
    def __init__(self) -> None:
        super().__init__('classifier_output_decoder')

        self.declare_parameter('input_topic', '/classifier_tensor_output')
        self.declare_parameter('output_topic', '/classifier_output')
        self.declare_parameter('logits_tensor_name', 'logits')

        self.input_topic = str(self.get_parameter('input_topic').value)
        self.output_topic = str(self.get_parameter('output_topic').value)
        self.logits_tensor_name = str(self.get_parameter('logits_tensor_name').value)

        self.sub = self.create_subscription(
            TensorList,
            self.input_topic,
            self.tensor_callback,
            10,
        )

        self.pub = self.create_publisher(Int32, self.output_topic, 10)

        self.get_logger().info(
            f'Classifier output decoder ready. '
            f'input_topic={self.input_topic}, '
            f'output_topic={self.output_topic}, '
            f'logits_tensor_name={self.logits_tensor_name}'
        )

    def _find_tensor(self, msg: TensorList):
        for tensor in msg.tensors:
            if tensor.name == self.logits_tensor_name:
                return tensor

        if len(msg.tensors) > 0:
            self.get_logger().warn(
                f'Tensor "{self.logits_tensor_name}" not found. Using first tensor "{msg.tensors[0].name}".'
            )
            return msg.tensors[0]

        return None

    def tensor_callback(self, msg: TensorList) -> None:
        tensor = self._find_tensor(msg)
        if tensor is None:
            self.get_logger().warn('Received empty TensorList.')
            return

        dims = list(tensor.shape.dims)
        if len(dims) == 0:
            self.get_logger().warn('Received logits tensor with empty shape.')
            return

        try:
            arr = np.frombuffer(bytes(tensor.data), dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f'Failed to read tensor data: {e}')
            return

        expected_count = int(np.prod(dims))
        if arr.size != expected_count:
            self.get_logger().warn(
                f'Logits tensor data size mismatch: got {arr.size}, expected {expected_count}. '
                f'dims={dims}'
            )
            return

        try:
            logits = arr.reshape(dims)
        except Exception as e:
            self.get_logger().error(f'Failed to reshape logits to {dims}: {e}')
            return

        if logits.ndim == 1:
            pred = int(np.argmax(logits))
        else:
            pred = int(np.argmax(logits[0]))

        out_msg = Int32()
        out_msg.data = pred
        self.pub.publish(out_msg)

        self.get_logger().info(
            f'Published classifier_output={pred}, logits_shape={tuple(logits.shape)}'
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ClassifierOutputDecoder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()