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
export THEANO_FLAGS="device=gpu0"
echo python=$(which python3)
export JOBS=1
python3 -m "$module" "$config"
