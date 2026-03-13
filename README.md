# isaac ros gestures

## Build:

```
colcon build --packages-select isaac_ros_gestures
```

## Run:

```
ros2 launch isaac_ros_gestures pipeline.launch.py
```

## System Description

### Key Files & Components
- **`pipeline.launch.py`**: The primary launch configuration. It orchestrates the entire ROS 2 computational graph, injecting TensorRT nodes, managing remappings, and setting up the static TF for RViz visualization.
- **`theta_uvc_src.py`**: A specialized camera node connecting to the Ricoh Theta via GStreamer/UVC, extracting and publishing high-resolution uncompressed images.
- **`video_collector_node.py`**: A video-folder source/collector node that iterates through `.mp4` files, publishes frames to the pipeline, subscribes to `/pose_markers`, and writes fixed-length per-video CSV keypose arrays.
- **`handbox_decoder.py`**: Consumes a full-frame YOLO26 TensorRT output, selects the best hand bounding box, crops that region from `/image_raw`, and publishes a 640x640 RGB crop to `/image_cropped`.
- **`handpose_decoder.py`**: A parser node that translates the raw output tensors of the TensorRT keypoint model into fully scaled, 3D color-coded arrays ready to be drawn in RViz.

### Data Flow Architecture
1. **Source Generation:** `theta_uvc_src` captures 1920x960 frames and publishes them to `/image_raw`.
2. **Full-frame Hand Detection:** A `dnn_image_encoder` resizes `/image_raw` to 640x640 (letterboxed) and feeds a YOLO26 TensorRT node.
3. **Cropping:** `handbox_decoder` consumes the YOLO26 output, crops the best hand region from `/image_raw`, resizes it to 640x640, and publishes `/image_cropped`.
4. **Keypoint Inference:** A second `dnn_image_encoder` + YOLO26 TensorRT node runs on `/image_cropped` and publishes `/tensor_output`.
5. **Decoding & Scaling:** `handpose_decoder` consumes `/tensor_output` and publishes `visualization_msgs/MarkerArray` to `/pose_markers`.

### Central ROS Core Topics
- `/image_raw` (*`sensor_msgs/Image`*): The foundational full-resolution 1920x960 image stream.
- `/image_cropped` (*`sensor_msgs/Image`*): The 640x640 cropped square tracking the hand.
- `/tensor_view` (*`isaac_ros_tensor_list_interfaces/TensorList`*): The prepared tensors entering the execution block.
- `/tensor_output` (*`isaac_ros_tensor_list_interfaces/TensorList`*): The raw probabilistic variables exiting the model.
- `/pose_markers` (*`visualization_msgs/MarkerArray`*): The 3D positional nodes representing the hand skeleton (filtered via RViz using the `hand_keypoints/0` namespace).

## Training Data Collection

This repository includes a standalone Python script, `collector.py`, located in the root directory. It acts as an autonomous ROS 2 node that hooks directly into the running pipeline to record 1-second continuous training frame captures alongside dynamic gesture categorization.

### Running `collector.py`

Ensure that your `pipeline.launch.py` graph is currently running in a separate terminal and outputting hand keypoints.

1. Open a new terminal in the Dev Container.
2. Source your ROS 2 environment:
   ```bash
   source install/setup.bash
   ```
3. Run the standalone collector script from the repository root:
   ```bash
   cd src/isaac_ros_gestures/
   python3 collector.py
   ```

The node will immediately subscribe to `/pose_markers` and dynamically rotate through the 5 gesture classes (`NULL`, `LEFT`, `RIGHT`, `FORWARD`, `STAY`) every 5 seconds, writing tightly packed 30-frame sliding window rows of 21 `(X, Y)` keypoint coordinates to `training_data.csv` locally for your downstream machine learning tasks.
