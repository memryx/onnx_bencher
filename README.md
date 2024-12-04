# ONNX Bencher
Benchmark ONNX models between MemryX MX3 and ONNXRuntime on CPU, CUDA, TensorRT, and OpenVINO.


## Setup
### Requirements
All runners use Python **3.11**. Please make sure you have it installed, along with its corresponding `venv`, `pip`, and `-dev` packages.

Some dependent pip packages might require `cmake` (such as `onnxsim`). Make sure to install them if prompted.

#### MX3
Follow the [installation steps](https://developer.memryx.com/get_started/index.html) if using a MemryX chip.

### Make Python venvs
Execute the `make_venvs.sh` script and it'll make the 4 different required virtual envs.

```
./make_venvs.sh
```


## Running
In this repo's `run/` folder, execute the `run.sh` script with the following syntax:

```
./run.sh [path to onnx file] [list of backends to bench]
```

The first argument is the ONNX file to benchmark (make sure it's the original FP32 model, **not** a quantized model!).

The remaining args are a space-separated list of backends to run on. The options are:
* `mx3` (MemryX MX3 M.2)
* `cpu` (default onnxruntime CPU EP)
* `cuda` (GPU using CUDA)
* `trt` (GPU using TensorRT)
* `inpu` (Intel NPU using OpenVINO)

**IMPORTANT NOTE**: if you're including `mx3` in the list, always put it **first**.

#### Examples
For example, to benchmark a yolov8s onnx model on the MX3, then the CPU, then an nVidia GPU using TensorRT, do:
```
./run.sh yolov8s.onnx mx3 cpu trt
```

Or for running a resnet50 onnx on everything:
```
./run.sh resnet50.onnx mx3 cpu cuda trt inpu
```

Or for running just on CPU and CUDA:
```
./run.sh resnet50 cpu cuda
```


## Configuration
The `run.sh` script has a couple variables defined at the top you may want to edit. They are:
* `FRAMES`: the number of frames to benchmark across all backends
* `EFFORT_HARD`: whether or not to use `--effort hard` [flag](https://developer.memryx.com/tools/neural_compiler.html#mapping-arguments) for the NeuralCompiler
* `FP16`: whether or not to use [FP16](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) math when running on a GPU. This option is a more fair comparison with the MX3's [BF16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) than the default FP32.


## Models to Run

The MX3 specializes in computer vision models. See the [Model Explorer](https://developer.memryx.com/model_explorer/models.html) for a handy subset of models that have been downloaded from their original sources and tested.

Good sources of ONNX models to try include [the ONNX Model Zoo](https://github.com/onnx/models), [PINTO's model zoo](https://github.com/PINTO0309/PINTO_model_zoo), [OpenMMLab](https://github.com/open-mmlab), and [Ultralytics YOLOs exported to ONNX (up to v9)](https://docs.ultralytics.com/integrations/onnx/).



## Tweaking
There are additional arguments possible with each of the backends' `run_*.py` scripts. Try running them directly to play around with options to onnxruntime.

For future work, one might consider adding variable batch sizes for the non-MX3 runners. Or maybe a better way to cache DFPs you've already compiled.
