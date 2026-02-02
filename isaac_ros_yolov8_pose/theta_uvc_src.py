#!/usr/bin/env python3
# Copyright 2024 Isaac ROS YOLOv8 Pose
# SPDX-License-Identifier: MIT

"""
ThetaUvcSrc Node

GStreamer-based ROS 2 node that captures frames from a Ricoh Theta camera
using thetauvcsrc and publishes them as sensor_msgs/Image.

Pipeline: thetauvcsrc → h264parse → nvv4l2decoder → videoconvert → appsink
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib


class ThetaUvcSrc(Node):
    """
    ROS 2 Node for capturing frames from Ricoh Theta camera via GStreamer.
    
    Uses thetauvcsrc element with hardware-accelerated H.264 decoding
    and publishes frames as sensor_msgs/Image.
    """

    def __init__(self):
        super().__init__('theta_uvc_src')
        
        # Declare parameters
        self.declare_parameter('width', 1920)
        self.declare_parameter('height', 1080)
        self.declare_parameter('framerate', 30)
        self.declare_parameter('frame_id', 'theta_camera')
        self.declare_parameter('output_topic', 'image_raw')
        
        # Get parameters
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.framerate = self.get_parameter('framerate').value
        self.frame_id = self.get_parameter('frame_id').value
        output_topic = self.get_parameter('output_topic').value
        
        self.get_logger().info(
            f'ThetaUvcSrc initialized: {self.width}x{self.height}@{self.framerate}fps'
        )
        
        # Publisher for image output
        self.image_pub = self.create_publisher(
            Image,
            output_topic,
            10
        )
        
        # Initialize GStreamer
        Gst.init(None)
        
        # Build and start the pipeline
        self.pipeline = None
        self.appsink = None
        self._build_pipeline()
        self._start_pipeline()
        
        # Create timer to poll for frames
        self.poll_timer = self.create_timer(1.0 / self.framerate, self._poll_frame)
        
        self.frame_count = 0

    def _build_pipeline(self):
        """Build the GStreamer pipeline for Theta camera capture."""
        
        # Pipeline based on reference_deepstream.c:
        # thetauvcsrc → h264parse → nvv4l2decoder → videoconvert → appsink
        pipeline_str = (
            'thetauvcsrc name=src ! '
            'h264parse ! '
            'nvv4l2decoder ! '
            'nvvideoconvert ! '
            f'video/x-raw,format=RGB,width={self.width},height={self.height} ! '
            'appsink name=sink emit-signals=true sync=false max-buffers=2 drop=true'
        )
        
        self.get_logger().info(f'GStreamer pipeline: {pipeline_str}')
        
        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
        except GLib.Error as e:
            self.get_logger().error(f'Failed to create GStreamer pipeline: {e}')
            raise RuntimeError(f'GStreamer pipeline creation failed: {e}')
        
        # Get the appsink element
        self.appsink = self.pipeline.get_by_name('sink')
        if not self.appsink:
            raise RuntimeError('Failed to get appsink element')
        
        # Connect to new-sample signal
        self.appsink.connect('new-sample', self._on_new_sample)

    def _start_pipeline(self):
        """Start the GStreamer pipeline."""
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self.get_logger().error('Failed to start GStreamer pipeline')
            raise RuntimeError('GStreamer pipeline failed to start')
        
        self.get_logger().info('GStreamer pipeline started')

    def _on_new_sample(self, appsink):
        """Callback when a new sample is available from appsink."""
        sample = appsink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.ERROR
        
        # Get buffer and caps
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        
        # Extract frame dimensions from caps
        structure = caps.get_structure(0)
        width = structure.get_value('width')
        height = structure.get_value('height')
        
        # Map buffer to read data
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            self.get_logger().warn('Failed to map GStreamer buffer')
            return Gst.FlowReturn.ERROR
        
        try:
            # Create ROS Image message
            msg = Image()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            
            msg.width = width
            msg.height = height
            msg.encoding = 'rgb8'
            msg.is_bigendian = False
            msg.step = width * 3  # RGB = 3 bytes per pixel
            msg.data = bytes(map_info.data)
            
            # Publish
            self.image_pub.publish(msg)
            
            self.frame_count += 1
            if self.frame_count % 100 == 0:
                self.get_logger().debug(f'Published {self.frame_count} frames')
                
        finally:
            buffer.unmap(map_info)
        
        return Gst.FlowReturn.OK

    def _poll_frame(self):
        """Poll for GStreamer bus messages (errors, EOS, etc.)."""
        if self.pipeline is None:
            return
        
        bus = self.pipeline.get_bus()
        while True:
            msg = bus.pop()
            if msg is None:
                break
            
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                self.get_logger().error(f'GStreamer error: {err.message} ({debug})')
            elif msg.type == Gst.MessageType.WARNING:
                warn, debug = msg.parse_warning()
                self.get_logger().warn(f'GStreamer warning: {warn.message} ({debug})')
            elif msg.type == Gst.MessageType.EOS:
                self.get_logger().info('GStreamer end of stream')

    def destroy_node(self):
        """Clean up GStreamer pipeline on shutdown."""
        if self.pipeline:
            self.get_logger().info('Stopping GStreamer pipeline')
            self.pipeline.set_state(Gst.State.NULL)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ThetaUvcSrc()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
