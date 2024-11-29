import argparse
import numpy as np
import time
import os
from memryx import NeuralCompiler
from memryx import Benchmark

def compile_if_no_dfp(model_path, effort):
    """
    Checks if a .dfp version of the given file exists.
    If it does not, compile it with NeuralCompiler.
    Saves the DFP to file, and returns a string path
    to the DFP + the ONNX model to compare against.

    The ONNX model may be either the original, or in the
    case of the Compiler cropping the model for MX3, it
    is a path to the core model (no pre/post).

    :param model_path: Source ONNX model path. MUST BE IN CWD!
    :return: dfp path, onnx path
    """

    basename = str(os.path.basename(model_path))

    model_without_ext, _ = os.path.splitext(basename)

    dfp_path = str(model_without_ext) + ".dfp"
    core_model = "model_0_" + model_without_ext + ".onnx"
    #pre_model = "model_0_" + model_without_ext + "_pre.onnx"
    #post_model = "model_0_" + model_without_ext + "_post.onnx"

    if os.path.exists(dfp_path):
        print("DFP already found, skipping compilation")

        # was it a cropped DFP ?
        if os.path.exists(core_model):
            return dfp_path, core_model
        else:
            return dfp_path, basename


    # else we need to compile first
    print(f"Compiling {basename}.....")
    try:
        nc = NeuralCompiler(models=basename, verbose=1, effort=effort, no_sim_dfp=True, dfp_fname=model_without_ext)
        dfp = nc.run()
        dfp.write(dfp_path)
    except:
        try:
            # try again with autocrop
            nc = NeuralCompiler(models=basename, verbose=1, effort=effort, no_sim_dfp=True, autocrop=True, dfp_fname=model_without_ext)
            dfp = nc.run()
            dfp.write(dfp_path)
        except:
            raise RuntimeError("NeuralCompiler failed both with and without autocrop!")
        else:
            return dfp_path, core_model
    else:
        return dfp_path, basename

    # should never get here
    return None, None


def run_mx3(dfp_path, num_frames):

    def calculate_latency(time_points):
        # gets average latency by cutting out outliers (>3 sigma),
        # which may occur from run to run (e.g., P-core/E-core, etc.)
        time_points = np.array(sorted(time_points))

        std = np.std(time_points[:-1])
        mean = np.average(time_points[:-1])

        filtered_time_points = time_points[time_points < mean+3*std]
        filtered_time_points = filtered_time_points[filtered_time_points > mean-3*std]

        return np.average(filtered_time_points)

    latency, fps = 0, 0
    with Benchmark(dfp=dfp_path) as bench:
        latency_time_points = []
        for i in range(20):
            _,latency,_ = bench.run(frames=1,threading=False)
            latency_time_points.append(latency)
        latency = calculate_latency(latency_time_points)

        _,_,fps = bench.run(frames=num_frames)

    return latency, fps


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "ONNX benchmarker (nVidia CUDA execution provider)")
    parser.add_argument("model_path",
                        type = str,
                        help = "Input file name")
    parser.add_argument("-n", "--num_frames",
                        type = int,
                        default = 1000,
                        help = "Number of frames to run (default 1000)")
    parser.add_argument("--effort_hard",
                        default = False,
                        action = "store_true",
                        help = "Use effort=hard for the NeuralCompiler instead of default effort=normal. Longer compile time, but better MX3 performance.")

    args = parser.parse_args()

    effort = "normal"
    if args.effort_hard:
        effort = "hard"

    dfp_path, _ = compile_if_no_dfp(args.model_path, effort)
    print(f"Compilation finished. Now benchmarking {dfp_path}....")

    lat, fps = run_mx3(dfp_path, args.num_frames)


    print(f"Ran {args.num_frames} frames (batch=1)")
    print(f"  Average FPS: {fps:.2f}")
    print(f"  Average System Latency: {lat:.2f} ms")
