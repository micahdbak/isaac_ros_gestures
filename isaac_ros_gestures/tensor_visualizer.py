#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from isaac_ros_tensor_list_interfaces.msg import TensorList

class TensorVisualizer(Node):
    def __init__(self):
        super().__init__('tensor_visualizer')
        
        # Subs/Pubs
        self.tensor_sub = self.create_subscription(
            TensorList,
            'tensor_view',
            self.tensor_callback,
            10
        )
        self.image_pub = self.create_publisher(
            Image,
            'tensor_view_image',
            10
        )

    def tensor_callback(self, msg: TensorList):
        if not msg.tensors:
            return

        # Find input_tensor
        target_tensor = None
        for t in msg.tensors:
            if t.name == 'input_tensor':
                target_tensor = t
                break
        
        if target_tensor is None:
            # Fallback to first tensor
            target_tensor = msg.tensors[0]

        try:
            # Parse data
            # Shape is NCHW: [1, 3, 224, 224]
            shape = [d for d in target_tensor.shape.dims]
            if len(shape) != 4:
                return 
                
            raw_data = np.frombuffer(bytes(target_tensor.data), dtype=np.float32)
            raw_data = raw_data.reshape(shape) # [N, C, H, W]
            
            # Take first batch
            img_tensor = raw_data[0] # [C, H, W]
            
            # Transpose to HWC for visualization: [3, 224, 224] -> [224, 224, 3]
            img_np = np.transpose(img_tensor, (1, 2, 0))
            
            # Denormalize
            img_np = img_np * 127.5 + 127.5
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            
            # Create ROS Image message
            h, w, c = img_np.shape
            img_msg = Image()
            img_msg.header = msg.header
            img_msg.height = h
            img_msg.width = w
            img_msg.encoding = 'rgb8'
            img_msg.is_bigendian = 0
            img_msg.step = w * c
            img_msg.data = img_np.tobytes()
            
            self.image_pub.publish(img_msg)

        except Exception as e:
            self.get_logger().error(f'Failed to visualize tensor: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = TensorVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
