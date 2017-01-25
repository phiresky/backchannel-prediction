#!/bin/bash
set -e

# trainNN.train or evaluate.evaluate
module="$1"

config="$2"

echo config=$config
echo PATH=$PATH
echo LD=$LD_LIBRARY_PATH
echo CUDA=$CUDA_VISIBLE_DEVICES
#export CUDA_VISIBLE_DEVICES=1
export THEANO_FLAGS="device=cpu,floatX=float32"
echo python=$(which python3)
echo SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export JOBS=$SLURM_CPUS_PER_TASK
python3 -m "$module" "$config"
