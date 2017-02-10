#!/bin/bash
set -e

gpuid="$1"
# trainNN.train or evaluate.evaluate
module="$2"

config="$3"

echo config=$config
echo PATH=$PATH
echo LD=$LD_LIBRARY_PATH
echo CUDA=$CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$gpuid
export THEANO_FLAGS="device=gpu0"
echo python=$(which python3)
export JOBS=1
python3 -m "$module" "$config"
