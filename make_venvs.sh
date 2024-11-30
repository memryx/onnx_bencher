#!/bin/bash

which python3.11
if [ $? -ne 0 ]; then
  echo "ERROR: missing python3.11"
  exit 1
fi


###############################################
# create the CPU venv
###############################################
cd cpu
python3.11 -m venv cpu_venv311
if [ $? -ne 0 ]; then
  echo "ERROR: creating venv failed. Is python3-venv for 3.11 installed?"
  exit 1
fi
. cpu_venv311/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt

# deactivate that env so we can make new ones in the other folders
deactivate



###############################################
# create the CUDA / TensorRT venv
###############################################
cd ../nvidia
python3.11 -m venv nv_venv311
. nv_venv311/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt

# override LD_LIBRARY_PATH so we use the installed CUDA/TRT libs
echo "" >> nv_venv311/bin/activate
printf 'export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/cublas/lib/:${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/cuda_cupti/lib/:${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib/:${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib/:${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/cudnn/lib/:${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/cufft/lib/:${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/curand/lib/:${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/cusolver/lib/:${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/cusparse/lib/:${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/nccl/lib/:${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/nvjitlink/lib/:${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/nvtx/lib/:${VIRTUAL_ENV}/lib/python3.11/site-packages/tensorrt_libs/"' >> nv_venv311/bin/activate
echo "" >> nv_venv311/bin/activate

deactivate



###############################################
# create the OpenVINO venv (Intel NPU, etc.)
###############################################
cd ../openvino
python3.11 -m venv ov_venv311
. ov_venv311/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt

deactivate



###############################################
# create the MemryX MX3 SDK venv
###############################################
cd ../mx3
python3.11 -m venv mx_venv311
. mx_venv311/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt

deactivate
