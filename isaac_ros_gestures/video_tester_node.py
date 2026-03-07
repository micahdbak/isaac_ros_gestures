#!/usr/bin/env python3

"""Video Tester Node - Replaces ThetaUvcSrc with a local video file."""

import cv2 as cv
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class VideoTesterNode(Node):
    """ROS 2 node that reads from a video file and publishes frames."""

    def __init__(self):
        super().__init__('video_tester_node')
        
        # Parameters
        self.declare_parameter('width', 1920)
        self.declare_parameter('height', 960)
        self.declare_parameter('frame_id', 'theta_camera')
        self.declare_parameter('video_path', '/workspaces/isaac_ros-dev/test_video.mp4')
        self.declare_parameter('fps', 30.0)
        
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.frame_id = self.get_parameter('frame_id').value
        self.video_path = self.get_parameter('video_path').value
        fps = self.get_parameter('fps').value
        
        # Publisher
        self.image_pub = self.create_publisher(Image, 'image_raw', 10)
        self.bridge = CvBridge()
        
        # Open video file
        self.cap = cv.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open video file at {self.video_path}")
            return
            
        # Determine actual video FPS if possible
        video_fps = self.cap.get(cv.CAP_PROP_FPS)
        if video_fps > 0:
            fps = video_fps
            
        self.get_logger().info(f"Publishing video {self.video_path} at {fps} FPS")
        
        # Timer for publishing frames
        timer_period = 1.0 / fps
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        
        if not ret:
            # Loop the video if it ends
            self.get_logger().info("Video ended. Restarting...")
            self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().error("Failed to read from video even after rewinding.")
                self.timer.cancel()
                return

        try:
            # Resize frame according to parameters
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv.resize(frame, (self.width, self.height))
                
            # Convert BGR (OpenCV default) to RGB
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
            # Convert to ROS message
            msg = self.bridge.cv2_to_imgmsg(frame_rgb, encoding='rgb8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            
            # Publish
            self.image_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing frame: {e}')

    def destroy_node(self):
        """Clean shutdown."""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VideoTesterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
