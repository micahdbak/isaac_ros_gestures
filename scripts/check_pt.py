import torch

# 1. Load the tensor from the .pt file
try:
    data = torch.load('../yolo26s-pose-hands.pt', weights_only=False)
    
    # 2. Check the type of the loaded object
    if isinstance(data.model, torch.Tensor):
        output_size = data.size()
        print(f"Loaded object is a single tensor.")
        print(f"Output tensor size: {output_size}")
        # Convert torch.Size object to a Python list if needed
        output_size_list = list(output_size)
        print(f"Output tensor size as list: {output_size_list}")
        
    elif isinstance(data, dict):
        print("Loaded object is a dictionary (likely a state_dict or multiple tensors).")
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                print(f"  Tensor '{key}' size: {value.size()}")
            else:
                print(f"  Key '{key}' holds non-tensor data (type: {type(value)})")
                
    else:
        print(f"Loaded object is of type: {type(data)}. Cannot determine tensor size directly.")

except Exception as e:
    print(f"An error occurred while loading the file: {e}")

