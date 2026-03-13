import sys
from ultralytics import YOLO

def main():
    pt_path = sys.argv[1]
    model = YOLO(pt_path)
    model.export(format='onnx', opset=12, simplify=True)

if __name__ == '__main__':
    main()
