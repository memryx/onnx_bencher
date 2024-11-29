import argparse
import onnx
import onnxruntime as ort
import numpy as np
import time

ort.set_default_logger_severity(3)

def get_model_input_shapes(model_path):
    """
    Load an ONNX model and extract input shapes.
    :param model_path: Path to the ONNX model file.
    :return: Dictionary of input names and their shapes.
    """
    model = onnx.load(model_path)
    input_shapes = {}
    for input_tensor in model.graph.input:
        name = input_tensor.name
        shape = [
            dim.dim_value if dim.dim_value > 0 else "dynamic"
            for dim in input_tensor.type.tensor_type.shape.dim
        ]
        input_shapes[name] = shape
    return input_shapes

def generate_random_input(input_shapes):
    """
    Generate random input data based on the input shapes.
    :param input_shapes: Dictionary of input tensor names to shapes.
    :return: Dictionary of input data with random values.
    """
    inputs = {}
    for name, shape in input_shapes.items():
        # Replace dynamic dimensions with 1 for inference
        resolved_shape = [dim if dim != "dynamic" else 1 for dim in shape]
        inputs[name] = np.random.random(resolved_shape).astype(np.float32)
    return inputs

def run_inferences(model_path, num_inferences, no_tensor_cores):
    """
    Run multiple inferences on an ONNX model using ONNX Runtime.
    :param model_path: Path to the ONNX model file.
    :param num_inferences: Number of inferences to run.
    :return: Total inference time in seconds.
    """
    # Load input shapes and create random input data
    input_shapes = get_model_input_shapes(model_path)
    random_inputs = generate_random_input(input_shapes)

    # Initialize ONNX Runtime session
    if no_tensor_cores:
        ort_session = ort.InferenceSession(model_path, providers=[ ("CUDAExecutionProvider", {"use_tf32": 0}) ])
    else:
        ort_session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])

    # Warm-up: Run one inference to initialize
    ort_session.run(None, random_inputs)

    # Run multiple inferences
    start_time = time.time()
    for _ in range(num_inferences):
        ort_session.run(None, random_inputs)
    end_time = time.time()

    total_time = end_time - start_time

    avg_fps = float(num_inferences / total_time)
    avg_lat = float(total_time / num_inferences)*1000.0
    return avg_lat, avg_fps

if __name__ == "__main__":

    # first make sure the CUDAExecutionProvider is even available...
    providers = ort.get_available_providers()

    if "CUDAExecutionProvider" in providers:
        pass
    else:
        print(f"ERROR: CUDAExecutionProvider not available. You only have {providers}")
        exit(1)

    parser = argparse.ArgumentParser(description = "ONNX benchmarker (nVidia CUDA execution provider)")
    parser.add_argument("model_path",
                        type = str,
                        help = "Input file name")
    parser.add_argument("-n", "--num_frames",
                        type = int,
                        default = 1000,
                        help = "Number of frames to run (default 1000)")
    parser.add_argument("--no_tensor_cores",
                        default = False,
                        action = "store_true",
                        help = "By default, CUDA backend will use TensorCores in Ampere and later. Use this to disable instead.")

    args = parser.parse_args()

    model_path = args.model_path

    # Number of inferences to perform
    num_inferences = args.num_frames

    # Run inferences and measure performance
    print(f"Running {num_inferences} inferences on the model: {model_path}")
    lat, fps = run_inferences(model_path, num_inferences, args.no_tensor_cores)

    print(f"Ran {args.num_frames} frames (batch=1)")
    print(f"  Average FPS: {fps:.2f}")
    print(f"  Average System Latency: {lat:.2f} ms")
