#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
import argparse

def install_dependencies():
    """Checks and installs Python requirements."""
    missing = []
    
    # Check by attempting actual imports
    try:
        import ultralytics
    except ImportError:
        missing.append("ultralytics")
    
    try:
        import onnx
    except ImportError:
        missing.append("onnx")

    if missing:
        print(f"[INFO] Installing missing dependencies: {', '.join(missing)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + missing)
            print("[INFO] Dependencies installed. Re-run the script.")
            sys.exit(0)
        except subprocess.CalledProcessError:
            print("[ERROR] Failed to install dependencies. Try: pip install ultralytics onnx")
            sys.exit(1)


def find_trtexec():
    """Locates the trtexec binary common in Isaac ROS containers."""
    candidates = [
        "/usr/src/tensorrt/bin/trtexec",  # Standard Jetson/Isaac ROS path
        "/usr/bin/trtexec",
        shutil.which("trtexec")
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None

def main():
    parser = argparse.ArgumentParser(description="Download YOLOv8 Pose and build TensorRT Engine.")
    parser.add_argument("--model", type=str, default="yolov8s-pose", 
                        help="Model size (yolov8n-pose, yolov8s-pose, yolov8m-pose, etc.)")
    parser.add_argument("--fp16", action="store_true", default=True, 
                        help="Enable FP16 precision (Recommended for Jetson)")
    parser.add_argument("--output_dir", type=str, default=os.getcwd(), 
                        help="Directory to save the final .plan file")
    
    args = parser.parse_args()

    # 1. Setup
    install_dependencies()
    from ultralytics import YOLO

    trtexec_path = find_trtexec()
    if not trtexec_path:
        print("[ERROR] 'trtexec' not found. Are you inside the Isaac ROS container?")
        sys.exit(1)
    
    print(f"[INFO] Using trtexec at: {trtexec_path}")

    # 2. Download and Export to ONNX
    print(f"[INFO] Downloading and loading {args.model}...")
    try:
        model = YOLO(f"{args.model}.pt")
    except Exception as e:
        print(f"[ERROR] Failed to download model: {e}")
        sys.exit(1)

    print("[INFO] Exporting to ONNX...")
    # opset=12 or higher is usually safe for TensorRT. Dynamic=False is preferred for standard Isaac ROS.
    onnx_path = model.export(format="onnx", dynamic=False, simplify=True)
    
    if not onnx_path:
        print("[ERROR] ONNX export failed.")
        sys.exit(1)

    # 3. Build TensorRT Engine (.plan)
    engine_name = f"{args.model}.plan"
    engine_path = os.path.join(args.output_dir, engine_name)
    
    cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}"
    ]
    
    if args.fp16:
        cmd.append("--fp16")

    print(f"[IMPORTANT] Ready for compilation; run `sudo {' '.join(cmd)}`.")

if __name__ == "__main__":
    main()
