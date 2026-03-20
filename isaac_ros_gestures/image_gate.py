#!/usr/bin/env python3

from __future__ import annotations

import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String


class ImageGate(Node):
    def __init__(self) -> None:
        super().__init__('image_gate')

        self.declare_parameter('input_topic', '/image_raw')
        self.declare_parameter('output_topic', '/image_raw_gated')
        self.declare_parameter('button_topic', '/arduino_buttons')
        self.declare_parameter('default_enabled', False)

        input_topic = str(self.get_parameter('input_topic').value)
        output_topic = str(self.get_parameter('output_topic').value)
        button_topic = str(self.get_parameter('button_topic').value)
        default_enabled = bool(self.get_parameter('default_enabled').value)

        self._enabled = default_enabled
        self._lock = threading.Lock()

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.image_pub = self.create_publisher(Image, output_topic, image_qos)

        self.image_sub = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            image_qos,
        )

        self.button_sub = self.create_subscription(
            String,
            button_topic,
            self.button_callback,
            10,
        )

        self.get_logger().info(
            f'Image gate started. input={input_topic}, output={output_topic}, '
            f'button_topic={button_topic}, enabled={self._enabled}'
        )

    def button_callback(self, msg: String) -> None:
        data = msg.data.strip()

        new_enabled = None
        if data == 'button_1_pressed':
            new_enabled = True
        elif data == 'button_1_released':
            new_enabled = False
        else:
            return

        with self._lock:
            changed = (new_enabled != self._enabled)
            self._enabled = new_enabled

        if changed:
            state = 'OPEN' if new_enabled else 'CLOSED'
            self.get_logger().info(f'Gate {state} from message: {data}')

    def image_callback(self, msg: Image) -> None:
        with self._lock:
            enabled = self._enabled

        if not enabled:
            return

        self.image_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImageGate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()