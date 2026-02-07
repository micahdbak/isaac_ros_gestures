import onnx
from onnx import helper, TensorProto

def convert_to_nchw_input(input_model_path, output_model_path):
    # Load the model
    model = onnx.load(input_model_path)
    graph = model.graph
    
    # Original input info
    original_input = graph.input[0]
    input_name = original_input.name
    
    # Create new input tensor info
    new_input_name = 'input_tensor'
    new_input_shape = [1, 3, 224, 224]
    new_input = helper.make_tensor_value_info(
        new_input_name,
        TensorProto.FLOAT,
        new_input_shape
    )
    
    # Create Transpose node: NCHW -> NHWC
    transpose_node = helper.make_node(
        'Transpose',
        inputs=[new_input_name],
        outputs=[input_name],
        perm=[0, 2, 3, 1],
        name='Transpose_NCHW_to_NHWC'
    )
    
    # Insert the new node at the beginning of the graph
    graph.node.insert(0, transpose_node)
    
    # Replace the input in the graph
    graph.input.remove(original_input)
    graph.input.insert(0, new_input)
    
    # Save the modified model
    onnx.save(model, output_model_path)
    print(f"Model saved to {output_model_path}")

if __name__ == "__main__":
    convert_to_nchw_input(
        '/home/robot/workspaces/isaac_ros-dev/models/handpose.onnx',
        '/home/robot/workspaces/isaac_ros-dev/models/handpose_nchw.onnx'
    )
