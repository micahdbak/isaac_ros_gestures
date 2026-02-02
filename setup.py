import os
from glob import glob
from setuptools import setup

package_name = 'isaac_ros_yolov8_pose'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='micah_baker@sfu.ca',
    description='YOLOv8 Pose Estimation using TensorRT with Isaac ROS',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolov8_pose_decoder = isaac_ros_yolov8_pose.yolov8_pose_decoder:main',
            'theta_uvc_src = isaac_ros_yolov8_pose.theta_uvc_src:main',
        ],
    },
)
