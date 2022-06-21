# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import io
import logging
import os
import random
import tarfile
import tempfile
import time
import multiprocessing
import shutil
import subprocess

import torchaudio

AUDIO_FORMAT_SETS = {'flac', 'mp3', 'ogg', 'opus', 'wav', 'wma'}

utt2path = {}


def write_wavs(data_list, tempdir):
    done_utts = []
    for item in data_list:
        segment_key, _, utt, wav, _, _ = item
        if len(wav) == 1:
            # path to the file
            suffix = os.path.splitext(wav[0])[1][1:]
            assert suffix in AUDIO_FORMAT_SETS, f'{suffix} not in {AUDIO_FORMAT_SETS}'
            utt2path[utt] = wav[0]
        else:
            assert wav[-1] == '|'
            wav = wav[:-1]
            if utt not in done_utts:
                out_dir = os.path.join(tempdir, os.path.dirname(utt))
                os.makedirs(out_dir, exist_ok=True)
                wav_out = os.path.join(tempdir, f'{utt}.wav')
                if not os.path.isfile(wav_out):
                    with open(wav_out, 'wb') as f:
                        subprocess.check_call(' '.join(wav), stdout=f, shell=True)
                utt2path[utt] = wav_out
                done_utts.append(utt)


def write_tar_file(data_list, tar_file, tempdir, utt2path, index=0, total=1):
    logging.info('Processing {} {}/{}'.format(tar_file, index, total))
    read_time = 0.0
    write_time = 0.0

    with tarfile.open(tar_file, "w") as tar:
        for item in data_list:
            try:
                segment_key, spk, utt, wav, start, end = item

                waveform, sample_rate = torchaudio.load(utt2path[utt])
                waveform = waveform[:, int(start * sample_rate):int(end * sample_rate)]

                # add segment wavefile
                with tempfile.NamedTemporaryFile(dir=tempdir, suffix='.wav') as fwav:
                    torchaudio.save(fwav.name, waveform, sample_rate)
                    fwav.flush()
                    fwav.seek(0)
                    data = fwav.read()

                    # add speaker label to the tar file
                    spk_file = f'{segment_key}.spk'
                    spk = spk.encode('utf8')
                    spk_data = io.BytesIO(spk)
                    spk_info = tarfile.TarInfo(spk_file)
                    spk_info.size = len(spk)
                    tar.addfile(spk_info, spk_data)

                    # add wave file to tar
                    wav_file = f'{segment_key}.wav'
                    wav_data = io.BytesIO(data)
                    wav_info = tarfile.TarInfo(wav_file)
                    wav_info.size = len(data)
                    tar.addfile(wav_info, wav_data)
            except:
                logging.info(f'Encountered problem with {segment_key} reading from {utt2path[utt]}')
                continue


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_utts_per_shard',
                        type=int,
                        default=1000,
                        help='num utts per shard')
    parser.add_argument('--num_threads',
                        type=int,
                        default=1,
                        help='num threads for make shards')
    parser.add_argument('--prefix',
                        default='shards',
                        help='prefix of shards tar file')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--shuffle',
                        action='store_true',
                        help='whether to shuffle data')
    parser.add_argument('wav_file', help='wav file')
    parser.add_argument('utt2spk_file', help='utt2spk file')
    parser.add_argument('segments_file', help='segments file')
    parser.add_argument('audio_dir', help='temporary path for storing audios')
    parser.add_argument('shards_dir', help='output shards dir')
    parser.add_argument('shards_list', help='output shards list file')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    random.seed(args.seed)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    wav_table = {}
    with open(args.wav_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            segment_key = arr[0]
            # this might also be a command, not necesarrily path to wave file
            wav_table[segment_key] = arr[1:]

    utt2spk = {}
    with open(args.utt2spk_file, 'r', encoding='utf8') as fin:
        for line in fin:
            wav_utt, spk = line.strip().split()
            utt2spk[wav_utt] = spk

    data = []
    with open(args.segments_file, 'r', encoding='utf8') as fin:
        for line in fin:
            segment_key, wav_utt, start, end = line.strip().split()
            assert wav_utt in wav_table
            assert segment_key in utt2spk
            data.append((segment_key, utt2spk[segment_key], wav_utt, wav_table[wav_utt], float(start), float(end)))

    logging.info(f'Obtained {len(data)} segments')
    if args.shuffle:
        random.shuffle(data)

    num = args.num_utts_per_shard
    chunks = [data[i:i + num] for i in range(0, len(data), num)]
    os.makedirs(args.shards_dir, exist_ok=True)

    # Using thread pool to speedup
    # pool_wavs = multiprocessing.Pool(processes=args.num_threads)
    # os.makedirs(args.audio_dir, exist_ok=True)
    # for _ in range(args.num_threads):
    #     pool_wavs.apply_async(write_wavs, (data, args.audio_dir))
    # pool_wavs.close()
    # pool_wavs.join()

    # pretty stupid, but it is faster to call write_wavs again in the main thread, so utt2path will get updated
    # instead of shared memory across threads/processes
    write_wavs(data, args.audio_dir)

    shards_list = []
    num_chunks = len(chunks)
    pool_tars = multiprocessing.Pool(processes=args.num_threads)
    for i, chunk in enumerate(chunks):
        tar_file = os.path.join(args.shards_dir,
                                '{}_{:09d}.tar'.format(args.prefix, i))
        shards_list.append(tar_file)
        pool_tars.apply_async(write_tar_file, (chunk, tar_file, args.audio_dir, utt2path, i, num_chunks))

    pool_tars.close()
    pool_tars.join()

    with open(args.shards_list, 'w', encoding='utf8') as fout:
        for name in shards_list:
            fout.write(name + '\n')


if __name__ == '__main__':
    main()
