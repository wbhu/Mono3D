#!/bin/sh
# LOG
# shellcheck disable=SC2230
# shellcheck disable=SC2086
set -x
# Exit script when a command returns nonzero state
set -e
set -o pipefail

export OMP_NUM_THREADS=10
export KMP_INIT_AT_FORK=FALSE

PYTHON=python
dataset=Flickr1024
TRAIN_CODE=train.py
TEST_CODE=test.py

exp_name=$1
config=$2

exp_dir=Exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result

now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${model_dir} ${result_dir}
mkdir -p ${exp_dir}/result/last
mkdir -p ${exp_dir}/result/best
cp scripts/train.sh scripts/test.sh main/${TRAIN_CODE} main/${TEST_CODE} ${config} ${exp_dir}

export PYTHONPATH=./
echo $OMP_NUM_THREADS | tee -a ${exp_dir}/train-$now.log
nvidia-smi | tee -a ${exp_dir}/train-$now.log
which pip | tee -a ${exp_dir}/train-$now.log

$PYTHON -u ${exp_dir}/${TRAIN_CODE} \
  --config=${config} \
  save_path ${exp_dir} \
  2>&1 | tee -a ${exp_dir}/train-$now.log

# TEST
now=$(date +"%Y%m%d_%H%M%S")

$PYTHON -u ${exp_dir}/${TEST_CODE} \
  --config=${config} \
  save_folder ${exp_dir}/result/best \
  model_path ${model_dir}/model_best.pth.tar \
  2>&1 | tee ${exp_dir}/test_best-$now.log
