#!/bin/bash
# coding:utf-8
# Author: Hongji Wang

exp_dir=''
model_path=''
nj=1
gpus="[]"
data_type="segments"  # shard/raw

. tools/parse_options.sh
set -e

data_name_array=("Michele_5-23-22_chunk9_18000")
data_list_path_array=("/shared/workspaces/jprofant/GitHub/revspeech/kaldi_egs/problematic/Michele_5-23-22_chunk9_18000/diarization/Michele_5-23-22_chunk9_18000/kaldi/segments")
data_scp_path_array=("/shared/workspaces/jprofant/GitHub/revspeech/kaldi_egs/problematic/Michele_5-23-22_chunk9_18000/diarization/Michele_5-23-22_chunk9_18000/kaldi/segments") # to count the number of wavs
nj_array=($nj $nj)
batch_size_array=(1) # batch_size of test set must be 1 !!!
num_workers_array=(1)
count=${#data_name_array[@]}

for i in $(seq 0 $(($count - 1))); do
  wavs_num=$(wc -l ${data_scp_path_array[$i]} | awk '{print $1}')
  echo "bash tools/extract_embedding_emb.sh --exp_dir ${exp_dir} \
    --model_path $model_path \
    --data_type ${data_type} \
    --data_list ${data_list_path_array[$i]} \
    --wavs_num ${wavs_num} \
    --store_dir ${data_name_array[$i]} \
    --batch_size ${batch_size_array[$i]} \
    --num_workers ${num_workers_array[$i]} \
    --nj ${nj_array[$i]} \
    --gpus $gpus"
done

wait

echo "Embedding dir is (${exp_dir}/embeddings)."
