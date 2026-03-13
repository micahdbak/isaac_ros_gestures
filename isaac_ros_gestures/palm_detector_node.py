#!/usr/bin/env python3

"""Palm Detector Node - Runs MediaPipe Palm Detection to crop hands for YOLO26 inference."""

import os
import cv2 as cv
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

# Import the provided MPPalmDet class
from isaac_ros_gestures.mp_palmdet import MPPalmDet

class PalmDetectorNode(Node):
    def __init__(self):
        super().__init__('palm_detector_node')
        
        # Parameters
        self.declare_parameter('model_path', '/workspaces/isaac_ros-dev/models/palm_detection_mediapipe_2023feb.onnx')
        self.declare_parameter('score_threshold', 0.75)
        self.declare_parameter('nms_threshold', 0.3)
        self.declare_parameter('padding_ratio', 4.0)
        self.declare_parameter('max_movement_threshold', 200.0)
        
        model_path = self.get_parameter('model_path').value
        self.score_threshold = self.get_parameter('score_threshold').value
        self.nms_threshold = self.get_parameter('nms_threshold').value
        self.padding_ratio = self.get_parameter('padding_ratio').value
        self.max_movement_threshold = self.get_parameter('max_movement_threshold').value
        
        self.last_palm_center = None
        self.last_crop_coords = None
        
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
        
        # Publisher for cropped image and ROI
        self.image_pub = self.create_publisher(Image, 'image_cropped', 1)
        self.roi_pub = self.create_publisher(CameraInfo, 'palm_roi', 1)
        
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
            
            # Crop image which is 1920x960 to the central 960x960 square before running inference
            crop_offset_x = 0
            if cv_image.shape[1] == 1920 and cv_image.shape[0] == 960:
                crop_offset_x = 480
                cv_image = cv_image[:, crop_offset_x:1440]
            
            # Run inference
            palms = self.detector.infer(cv_image)
            
            valid_palm_found = False
            
            if len(palms) > 0:
                # Select largest palm by bounding box area (w * h)
                best_palm = max(palms, key=lambda p: (p[2] - p[0]) * (p[3] - p[1]))
                
                # Extract bounding box
                x1, y1, x2, y2 = best_palm[0:4]
                
                # Calculate center and dimensions
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w / 2
                cy = y1 + h / 2
                
                # If the new palm is too far from the previous one, drop it
                dist_ok = True
                if self.last_palm_center is not None:
                    dist = ((cx - self.last_palm_center[0]) ** 2 + (cy - self.last_palm_center[1]) ** 2) ** 0.5
                    if dist > self.max_movement_threshold:
                        dist_ok = False
                
                if dist_ok:
                    valid_palm_found = True
                    # Update the last recorded center
                    self.last_palm_center = (cx, cy)
                    
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
                    
                    self.last_crop_coords = (new_x1, new_y1, new_x2, new_y2)
            
            if not valid_palm_found:
                if self.last_crop_coords is None:
                    # No previous palm detected, return out
                    return
                # Reuse the previously sent bounding box
                new_x1, new_y1, new_x2, new_y2 = self.last_crop_coords
            
            # Ensure we have a valid crop
            if new_x2 <= new_x1 or new_y2 <= new_y1:
                return
                
            # Crop image
            cropped_img = cv_image[new_y1:new_y2, new_x1:new_x2]
            
            # Convert back to RGB for downstream DNN Image Encoder
            cropped_img_rgb = cv.cvtColor(cropped_img, cv.COLOR_BGR2RGB)
            
            # Convert back to ROS message
            cropped_msg = self.bridge.cv2_to_imgmsg(cropped_img_rgb, encoding='rgb8')
            cropped_msg.header = msg.header
            
            # Create CameraInfo to hold the ROI and Header
            roi_msg = CameraInfo()
            roi_msg.header = msg.header
            roi_msg.roi.x_offset = int(new_x1) + crop_offset_x
            roi_msg.roi.y_offset = int(new_y1)
            roi_msg.roi.height = int(new_y2 - new_y1)
            roi_msg.roi.width = int(new_x2 - new_x1)
            roi_msg.roi.do_rectify = False
            
            # Publish
            self.image_pub.publish(cropped_msg)
            self.roi_pub.publish(roi_msg)
            
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
