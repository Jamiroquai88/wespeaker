#!/bin/bash
# coding:utf-8
# Author: Hongji Wang

. ./path.sh

stage=-1
stop_stage=-1

config=conf/resnet.yaml
exp_dir=exp/ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_rev-data
data_type="shard"  # shard/raw
#gpus="[0]"
gpus="[0,1,2,3]"
num_avg=10
checkpoint=
#exp/ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_rev-data/models/model_15.pt

score_norm_method="asnorm"  # asnorm/snorm
top_n=100
trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"

. tools/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  ./local/prepare_data.sh --stage 1 --stop_stage 4
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert train and test data to ${data_type}..."
  for dset in train; do
    if [ $data_type == "shard" ]; then
      python tools/make_shard_list_segments.py --num_utts_per_shard 1000 \
          --num_threads 8 \
          --prefix shards \
          --shuffle \
          data/$dset/wav.scp data/$dset/utt2spk data/$dset/segments \
          data/$dset/audio data/$dset/shards data/$dset/shard.list
    else
      python tools/make_raw_list.py data/$dset/wav.scp \
          data/$dset/utt2spk data/$dset/raw.list
    fi
  done
  # Convert all musan data to LMDB
  # FIXME python tools/make_lmdb.py data/musan/wav.scp data/musan/lmdb
  # Convert all rirs data to LMDB
  # FIXME python tools/make_lmdb.py data/rirs/wav.scp data/rirs/lmdb
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wespeaker/bin/train.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus $gpus \
      --num_avg ${num_avg} \
      --data_type "${data_type}" \
      --train_data data/train/${data_type}.list.1k \
      --train_label data/train/utt2spk \
      --reverb_data data/rirs/lmdb \
      --noise_data data/musan/lmdb \
      ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Do model average ..."
  avg_model=$exp_dir/models/avg_model.pt
  #python wespeaker/bin/average_model.py \
  #  --dst_model $avg_model \
  #  --src_path $exp_dir/models \
  #  --num ${num_avg}

  echo "Extract embeddings ..."
  local/extract_vox.sh \
    --exp_dir $exp_dir --model_path $avg_model \
    --nj 1 --gpus [0] --data_type $data_type
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Score ..."
  local/score.sh \
    --stage 1 --stop-stage 2 \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Score norm ..."
  local/score_norm.sh \
    --stage 1 --stop-stage 3 \
    --score_norm_method $score_norm_method \
    --cohort_set vox2_dev \
    --top_n $top_n \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Export the best model ..."
  python wespeaker/bin/export_jit.py \
    --config $exp_dir/config.yaml \
    --checkpoint $exp_dir/models/avg_model.pt \
    --output_file $exp_dir/models/final.zip
fi
