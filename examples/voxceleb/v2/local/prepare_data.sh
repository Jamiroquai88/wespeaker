#!/bin/bash
# coding:utf-8
# Author: Hongji Wang

stage=-1
stop_stage=-1

. tools/parse_options.sh || exit 1

download_dir=data/download_data
rawdata_dir=data/raw_data

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Download musan.tar.gz, rirs_noises.zip, vox1_test_wav.zip, vox1_dev_wav.zip, and vox2_aac.zip."
  echo "This may take a long time. Thus we recommand you to download all archives above in your own way first."

  ./local/download_data.sh --download_dir ${download_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Decompress all archives ..."
  echo "This could take some time ..."

  for archive in musan.tar.gz rirs_noises.zip; do
    [ ! -f ${download_dir}/$archive ] && echo "Archive $archive not exists !!!" && exit 1
  done
  [ ! -d ${rawdata_dir} ] && mkdir -p ${rawdata_dir}

  if [ ! -d ${rawdata_dir}/musan ]; then
    tar -xzvf ${download_dir}/musan.tar.gz -C ${rawdata_dir}
  fi

  if [ ! -d ${rawdata_dir}/RIRS_NOISES ]; then
    unzip ${download_dir}/rirs_noises.zip -d ${rawdata_dir}
  fi

  echo "Decompress success !!!"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Prepare wav.scp for each dataset ..."
  export LC_ALL=C # kaldi config

  mkdir -p data/musan data/rirs data/vox1 data/vox2_dev
  # musan
  find $(pwd)/${rawdata_dir}/musan -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >data/musan/wav.scp
  # rirs
  find $(pwd)/${rawdata_dir}/RIRS_NOISES/simulated_rirs -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >data/rirs/wav.scp

  echo "Success !!!"
fi
