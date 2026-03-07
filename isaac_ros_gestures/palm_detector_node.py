#!/usr/bin/env python3

"""Palm Detector Node - Runs MediaPipe Palm Detection to crop hands."""

import os
import cv2 as cv
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Import the provided MPPalmDet class
from isaac_ros_gestures.mp_palmdet import MPPalmDet

class PalmDetectorNode(Node):
    def __init__(self):
        super().__init__('palm_detector_node')
        
        # Parameters
        self.declare_parameter('model_path', '/workspaces/isaac_ros-dev/models/palm_detection_mediapipe_2023feb.onnx')
        self.declare_parameter('score_threshold', 0.6)
        self.declare_parameter('nms_threshold', 0.3)
        self.declare_parameter('padding_ratio', 3.0)
        
        model_path = self.get_parameter('model_path').value
        self.score_threshold = self.get_parameter('score_threshold').value
        self.nms_threshold = self.get_parameter('nms_threshold').value
        self.padding_ratio = self.get_parameter('padding_ratio').value
        
        self._is_processing = False
        
        self.bridge = CvBridge()
        
        # Initialize the detector
        if not os.path.exists(model_path):
            self.get_logger().error(f"Palm detection model not found at {model_path}")
            
        self.detector = MPPalmDet(
            modelPath=model_path,
            nmsThreshold=self.nms_threshold,
            scoreThreshold=self.score_threshold,
            backendId=cv.dnn.DNN_BACKEND_OPENCV,
            targetId=cv.dnn.DNN_TARGET_CPU
        )
        
        # Publisher for cropped image
        self.image_pub = self.create_publisher(Image, 'image_cropped', 1)
        
        # Subscriber to raw image with depth=1 to drop old frames
        self.image_sub = self.create_subscription(
            Image,
            'image_raw',
            self.image_callback,
            1
        )

    def image_callback(self, msg: Image):
        # Drop frame if we are already busy to prevent lag
        if self._is_processing:
            return
            
        self._is_processing = True
        try:
            # Convert ROS Image to OpenCV image (BGR for MPPalmDet)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Run inference
            palms = self.detector.infer(cv_image)
            
            if len(palms) == 0:
                # No palms detected, return out
                return
                
            # Select largest palm by bounding box area (w * h)
            best_palm = max(palms, key=lambda p: (p[2] - p[0]) * (p[3] - p[1]))
            
            # Extract bounding box
            x1, y1, x2, y2 = best_palm[0:4]
            
            # Calculate center and dimensions
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2
            
            # Apply padding and ensure it's square (expected by handpose model)
            side = max(w, h) * self.padding_ratio
            
            new_x1 = int(cx - side / 2)
            new_y1 = int(cy - side / 2)
            new_x2 = int(cx + side / 2)
            new_y2 = int(cy + side / 2)
            
            # Clamp to image boundaries
            img_h, img_w, _ = cv_image.shape
            new_x1 = max(0, new_x1)
            new_y1 = max(0, new_y1)
            new_x2 = min(img_w, new_x2)
            new_y2 = min(img_h, new_y2)
            
            # Ensure even dimensions for downstream DNN encoder video buffer
            if (new_x2 - new_x1) % 2 != 0:
                new_x2 -= 1
            if (new_y2 - new_y1) % 2 != 0:
                new_y2 -= 1
            
            # Ensure we have a valid crop
            if new_x2 <= new_x1 or new_y2 <= new_y1:
                return
                
            # Crop image
            cropped_img = cv_image[new_y1:new_y2, new_x1:new_x2]
            
            # Resize explicitly to 224x224 (target for handpose) to bypass downstream DNN scaling bugs
            cropped_img = cv.resize(cropped_img, (224, 224), interpolation=cv.INTER_AREA)
            
            # Convert back to RGB for downstream DNN Image Encoder
            cropped_img_rgb = cv.cvtColor(cropped_img, cv.COLOR_BGR2RGB)
            
            # Convert back to ROS message
            cropped_msg = self.bridge.cv2_to_imgmsg(cropped_img_rgb, encoding='rgb8')
            cropped_msg.header = msg.header
            
            # Publish
            self.image_pub.publish(cropped_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
        finally:
            self._is_processing = False

def main(args=None):
    rclpy.init(args=args)
    node = PalmDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
