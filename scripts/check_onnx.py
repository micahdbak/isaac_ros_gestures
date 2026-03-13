import onnxruntime as ort

# Load the model
session = ort.InferenceSession("../yolo26s-pose-hands.onnx")

# Get output details
for output in session.get_outputs():
    print(f"Output Name: {output.name}")
    print(f"Output Shape: {output.shape}")
    print(f"Output Type: {output.type}")
