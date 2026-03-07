#!/usr/bin/env python3
import csv
import numpy as np
import time
from collections import deque
from enum import Enum

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray

class Gesture(Enum):
    NULL = 0
    LEFT = 1
    RIGHT = 2
    FORWARD = 3
    STAY = 4

class DataCollectorNode(Node):
    def __init__(self):
        super().__init__('data_collector_node')
        
        self.label = Gesture.NULL
        self.queue = deque(maxlen=30)
        
        self.csv_file = open("training_data.csv", "a", newline="")
        self.writer = csv.writer(self.csv_file)
        
        # Subscribe to handpose markers published by handpose_decoder
        self.subscription = self.create_subscription(
            MarkerArray,
            '/pose_markers',
            self.marker_callback,
            10
        )
        
        # Timer to change label every 5 seconds
        self.label_timer = self.create_timer(5.0, self.change_label)
        
        self.get_logger().info("I: starting the frame writer (writing to training_data.csv)...")
        self.get_logger().info(f"I: classifying as {self.label.name} starting in 5 seconds")
        
    def change_label(self):
        # Cycle through all gestures
        self.label = Gesture((self.label.value + 1) % len(Gesture))
        self.queue.clear()
        self.get_logger().info(f"I: now classifying as {self.label.name}")
        
    def marker_callback(self, msg: MarkerArray):
        # The marker array contains a DELETEALL marker + the hand keypoint markers.
        # If there is only 1 marker or fewer, it means no hands were confidently detected.
        if len(msg.markers) <= 1:
            self.get_logger().warn('no confident hand pose read, queue cleared')
            self.queue.clear()
            return
            
        # Find the hand keypoints marker
        hand_marker = None
        for mk in msg.markers:
            if mk.ns == 'hand_keypoints' and mk.action == 0: # 0 == ADD
                hand_marker = mk
                break
                
        if not hand_marker or not hand_marker.points:
            self.get_logger().warn('no confident hand pose read, queue cleared')
            self.queue.clear()
            return

        # Extract X, Y for all 21 MediaPipe joints (ignoring Z to match legacy intentionality)
        joints_xy = []
        for point in hand_marker.points:
            joints_xy.extend([point.x, point.y])
            
        self.queue.append(joints_xy)
        
        # Once we have 30 continuous valid frames, write rolling window to CSV
        if len(self.queue) == 30:
            row = np.array(self.queue).flatten()
            row_s = [f"{x:.2f}" for x in row]
            row_s.append(str(self.label.value))
            self.writer.writerow(row_s)
            self.csv_file.flush()

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
