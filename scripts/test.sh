#!/bin/sh
# LOG
# shellcheck disable=SC2230
# shellcheck disable=SC2086
set -x
# Exit script when a command returns nonzero state
set -e
set -o pipefail

PYTHON=python
dataset=Flickr1024
TEST_CODE=test.py

exp_name=$1
config=$2

exp_dir=Exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result

mkdir -p ${model_dir} ${result_dir}
mkdir -p ${exp_dir}/result/last/${dataset}
mkdir -p ${exp_dir}/result/best/${dataset}
cp scripts/test.sh main/${TEST_CODE} ${exp_dir}

now=$(date +"%Y%m%d_%H%M%S")

export PYTHONPATH=./

$PYTHON -u ${exp_dir}/test.py \
  --config=${config} \
  save_folder ${exp_dir}/result/best \
  model_path ${model_dir}/model_best.pth.tar \
  2>&1 | tee ${exp_dir}/test_best-$now.log
