#!/bin/bash
# coding:utf-8
# Author: Hongji Wang, Chendong Liang

exp_dir='exp/XVEC'
model_path='avg_model.pt'
data_type='shard/raw'
data_list='shard.list/raw.list'
wavs_num=
store_dir='vox2_dev/vox1'
batch_size=1
num_workers=1
nj=4
gpus="[0,1]"

. tools/parse_options.sh
set -e

embed_dir=${exp_dir}/${store_dir}
log_dir=${embed_dir}/log
[ ! -d ${log_dir} ] && mkdir -p ${log_dir}

# split the data_list file into sub_file, then we can use multi-gpus to extract embeddings
data_num=$(wc -l ${data_list} | awk '{print $1}')
subfile_num=$(($data_num / $nj + 1))
split -l ${subfile_num} -d -a 3 ${data_list} ${log_dir}/split_
num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
gpus=(`echo $gpus | cut -d '[' -f2 | cut -d ']' -f1 | tr ',' ' '`)

for suffix in $(seq 0 $(($nj - 1))); do
  idx=$[$suffix % $num_gpus]
  suffix=$(printf '%03d' $suffix)
  data_list_subfile=${log_dir}/split_${suffix}
  embed_ark=${embed_dir}/xvector_${suffix}.ark
  CUDA_VISIBLE_DEVICES=${gpus[$idx]} /shared/workspaces/jprofant/anaconda3/envs/wespeaker/bin/python wespeaker/bin/extract_cpu.py \
    --config ${exp_dir}/config.yaml \
    --base_path $(dirname ${data_list}) \
    --model_path ${model_path} \
    --data_type ${data_type} \
    --data_list ${data_list_subfile} \
    --embed_ark ${embed_ark} \
    --batch-size ${batch_size} \
    --num-workers ${num_workers} \
    >${log_dir}/split_${suffix}.log 2>&1
done

