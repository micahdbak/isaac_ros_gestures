#!/usr/bin/env python3
"""ThetaUvcSrc - ROS 2 node for Ricoh Theta camera via GStreamer."""

import threading
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst


class ThetaUvcSrc(Node):
    """ROS 2 node that captures Theta camera frames and publishes them."""

    def __init__(self):
        super().__init__('theta_uvc_src')
        
        # Parameters
        self.declare_parameter('width', 1920)
        self.declare_parameter('height', 960)
        self.declare_parameter('frame_id', 'theta_camera')
        
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.frame_id = self.get_parameter('frame_id').value
        
        # Publisher
        self.image_pub = self.create_publisher(Image, 'image_raw', 10)
        
        # Initialize GStreamer
        Gst.init(None)
        
        # Build and verify pipeline
        self.pipeline = None
        self.appsink = None
        self._build_pipeline()
        
        # Start capture thread
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        
        # Start pipeline, then thread
        self._start_pipeline()
        self._thread.start()

    def _build_pipeline(self):
        """Build GStreamer pipeline with software decoding."""
        pipeline_str = ' ! '.join([
            'thetauvcsrc mode=2K',
            'queue max-size-buffers=3 leaky=downstream',
            'h264parse',
            'avdec_h264',
            'queue max-size-buffers=2 leaky=downstream', 
            'videoscale',
            'videoconvert',
            f'video/x-raw,format=RGB,width={self.width},height={self.height}',
            'appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true'
        ])
        
        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
        except Exception as e:
            self.get_logger().error(f'Failed to create pipeline: {e}')
            raise
        
        self.appsink = self.pipeline.get_by_name('sink')
        if not self.appsink:
            raise RuntimeError('Failed to get appsink element')

    def _start_pipeline(self):
        """Start the GStreamer pipeline."""
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        
        if ret == Gst.StateChangeReturn.FAILURE:
            self.get_logger().error('Failed to start pipeline')
            raise RuntimeError('Pipeline failed to start')
        
        # Wait for pipeline to reach PLAYING state
        ret, state, pending = self.pipeline.get_state(5 * Gst.SECOND)
        if ret != Gst.StateChangeReturn.SUCCESS:
            self.get_logger().error(f'Pipeline state change failed: {ret}')
            raise RuntimeError('Pipeline state change failed')

    def _capture_loop(self):
        """Background thread that pulls frames from GStreamer."""
        while self._running:
            # Check pipeline health
            bus = self.pipeline.get_bus()
            msg = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if msg:
                if msg.type == Gst.MessageType.ERROR:
                    err, debug = msg.parse_error()
                    self.get_logger().error(f'Pipeline error: {err.message}')
                elif msg.type == Gst.MessageType.EOS:
                    pass
                break
            
            # Pull frame with 100ms timeout
            sample = self.appsink.emit('try-pull-sample', 100 * Gst.MSECOND)
            
            if sample is None:
                continue
            
            # Extract frame data
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            
            if not caps:
                continue
                
            structure = caps.get_structure(0)
            width = structure.get_value('width')
            height = structure.get_value('height')
            
            # Map buffer for reading
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if not success:
                continue
            
            try:
                import time
                import numpy as np
                from cv_bridge import CvBridge
                
                bridge = CvBridge()
                
                # Use numpy for fast buffer access (zero-copy view)
                np_array = np.frombuffer(map_info.data, dtype=np.uint8)
                np_array = np_array.reshape((height, width, 3))
                
                # Use cv_bridge to convert to ROS message (optimized)
                msg = bridge.cv2_to_imgmsg(np_array, encoding='rgb8')
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.frame_id
                
                # Publish
                self.image_pub.publish(msg)
                    
            finally:
                buffer.unmap(map_info)

    def destroy_node(self):
        """Clean shutdown."""
        self._running = False
        
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ThetaUvcSrc()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
