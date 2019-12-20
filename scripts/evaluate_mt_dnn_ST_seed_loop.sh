#!/bin/bash
if [[ $# -ne 5 ]]; then
  echo "bash evaluate_mt_dnn_ST_seed_loop.sh <gpu> <model_name> <dataset> <timestamp> <train_data_ratio>"
  echo "<train_data_ratio> in percentage (i.e. 10, 20, 30, ...)!"
  exit 1
fi

# the timestamp can be found either in the result files or as the folder name of the checkpoints

data=$3
seeds=("0" "1" "2" "3" "4")
model=$2 #bert_model_large, mt_dnn_large

prefix="mt-dnn-${data}_ST"
BATCH_SIZE=16
gpu=$1
echo "export CUDA_VISIBLE_DEVICES=${gpu}"
export CUDA_VISIBLE_DEVICES=${gpu}
tstr=$4 # e.g. "2019-06-19T1750"

train_datasets=${data}
test_datasets=${data}
stress_tests="NONE" #negation,spelling,paraphrase
DATA_DIR="../data/mt_dnn"

answer_opt=1
optim="adamax"
grad_clipping=0
global_grad_clipping=1
lr="5e-5"
epochs=5
max_seq_len=100 # for the longer stress tests it will automatically choose 512
dump_to_checkpoints=1 # if 0, only dumps results to result folder
train_data_ratio=$5

for seed in "${seeds[@]}" ; do
    if [[ $train_data_ratio -eq 100 ]]; then
        model_dir="../checkpoints/${prefix}_seed${seed}_ep${epochs}_${model}_answer_opt${answer_opt}_${tstr}"
    else
        model_dir="../checkpoints/${prefix}_seed${seed}_ep${epochs}_${model}_answer_opt${answer_opt}_trainratio${train_data_ratio}_${tstr}"
    fi
    BERT_PATH="${model_dir}/model.pt"
    cp "../mt_dnn_models/vocab.txt" $model_dir
    log_file="${model_dir}/log.log"
    python ../predict.py --train_data_ratio ${train_data_ratio} --dump_to_checkpoints ${dump_to_checkpoints}  --stress_tests ${stress_tests} --max_seq_len ${max_seq_len} --seed ${seed} --epochs ${epochs} --data_dir ${DATA_DIR} --init_checkpoint ${BERT_PATH} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} --train_datasets ${test_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --learning_rate ${lr} --multi_gpu_on
done