#!/bin/bash

FRAMES=1000     # number of frames to run for all
EFFORT_HARD=0   # compile for MX3 with effort=hard
FP16=1          # use FP16 for TensorRT


# give this script the .onnx you want to run
if [[ ! -f ${1} ]]; then
  echo "Error, file: ${1} not found"
  exit 1
fi

# make a rundir
ORIGPWD=$(pwd)
MODELFILE=$(basename "${1}")
MODELNAME=$(basename "${1}" .onnx)

RUNDIR="rundir_${MODELNAME}_$(date +%H_%M_%S)"
mkdir -v $RUNDIR
if [ $? -ne 0 ]; then
  exit 1
fi

ORIG_FILE_ABS=$(readlink -f ${1})

cd $RUNDIR
ln -sv $ORIG_FILE_ABS .


#######
# RUN #
# MX3 #
#######
run_mx3 () {
  echo -e "\nMX3"
  . ../../mx3/mx_venv311/bin/activate
  if [[ $EFFORT_HARD -eq 1 ]]; then
    python3 ../../mx3/run_mx3.py --effort_hard --num_frames $FRAMES $MODELFILE
  else
    python3 ../../mx3/run_mx3.py --num_frames $FRAMES $MODELFILE
  fi
  if [ $? -ne 0 ]; then
    exit 1
  fi
  
  deactivate
}


#######
# RUN #
# CPU #
#######
run_cpu () {
  echo -e "\nCPU"
  . ../../cpu/cpu_venv311/bin/activate
  python3 ../../cpu/run_cpu.py --num_frames $FRAMES $MODELFILE
  if [ $? -ne 0 ]; then
    exit 1
  fi
  
  deactivate
}



#######
# RUN #
# GPU #
#######
run_cuda () {
  echo -e "\nCUDA"
  . ../../nvidia/nv_venv311/bin/activate
  python3 ../../nvidia/run_cuda.py --num_frames $FRAMES $MODELFILE
  if [ $? -ne 0 ]; then
    exit 1
  fi
  
  deactivate
}
run_trt () {
  echo -e "\nTensorRT"
  . ../../nvidia/nv_venv311/bin/activate
  if [[ $FP16 -eq 1 ]]; then
    python3 ../../nvidia/run_trt.py --use_fp16 --num_frames $FRAMES $MODELFILE
  else
    python3 ../../nvidia/run_trt.py --num_frames $FRAMES $MODELFILE
  fi
  if [ $? -ne 0 ]; then
    exit 1
  fi
  
  deactivate
}



#######
# RUN #
# NPU #
#######
run_intel_npu () {
  echo -e "\nIntel NPU"
  . ../../openvino/ov_venv311/bin/activate
  python3 ../../openvino/run_openvino.py --npu --num_frames $FRAMES $MODELFILE
  if [ $? -ne 0 ]; then
    exit 1
  fi
  
  deactivate
}







#################################################################
# Go through the rest of the args and run the according benchmark
#################################################################

for arg in "$@" ; do

  if [[ $arg == $1 ]]; then
    continue
  else

    if [[ $arg == "mx3" ]]; then
      run_mx3
    elif [[ $arg == "cpu" ]]; then
      run_cpu
    elif [[ $arg == "cuda" ]]; then
      run_cuda
    elif [[ $arg == "trt" ]]; then
      run_trt
    elif [[ $arg == "inpu" ]]; then
      run_intel_npu
    else
      echo "Unknown arg ${arg}. Valid options are mx3, cpu, cuda, trt, and inpu"
      exit 1
    fi

  fi

done


cd $ORIGPWD
