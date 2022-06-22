# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#               2022 Hongji Wang (jijijiang77@gmail.com)
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

import boto3
import io
import kaldiio
import json
import logging
import os
import random
import re
import tarfile
from subprocess import PIPE, Popen
from urllib.parse import urlparse

import numpy as np
from scipy import signal
from scipy.io import wavfile
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])

# The following are required to avoid S3 messages polluting our logs
# and slowing down the training because of the high number of messages
logging.getLogger('boto3').setLevel(logging.CRITICAL)
logging.getLogger('botocore').setLevel(logging.CRITICAL)
logging.getLogger('nose').setLevel(logging.CRITICAL)
logging.getLogger('s3transfer').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)

# FIXME : JPR : how can we get this credential thing work a bit more transparently?
# one needs to add the following in their ~/.aws/config
#
# [profile rev-inst]
# credential_source=Ec2InstanceMetadata
#
# this uses the fact that we have EC2-level permissions to access our s3 speech buckets
try:
    session = boto3.Session(profile_name='rev-inst')
    s3res = session.client('s3', region_name="us-west-2")
except:
    logging.warning('Failed to initialize s3 session. If you are using s3:// shards it might not work.')


def url_opener(data):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
    """
    for sample in data:
        assert 'src' in sample
        # TODO(Binbin Zhang): support HTTP
        url = sample['src']
        try:
            pr = urlparse(url)
            # local file
            if pr.scheme == '' or pr.scheme == 'file':
                stream = open(url, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
            elif pr.scheme == 's3':
                buf = io.BytesIO()
                s3res.download_fileobj(pr.netloc, pr.path[1:], buf)
                buf.seek(0)
                stream = io.BufferedReader(buf)
            else:
                cmd = f'curl -s -L {url}'
                process = Popen(cmd, shell=True, stdout=PIPE)
                sample.update(process=process)
                stream = process.stdout
            sample.update(stream=stream)
            yield sample
        except Exception as ex:
            logging.warning('Failed to open {}'.format(url))


def tar_file_and_group(data):
    """ Expand a stream of open tar files into a stream of tar file contents.
        And groups the file with same prefix

        Args:
            data: Iterable[{src, stream}]

        Returns:
            Iterable[{key, wav, spk, sample_rate}]
    """
    for sample in data:
        assert 'stream' in sample
        stream = tarfile.open(fileobj=sample['stream'], mode="r|*")
        prev_prefix = None
        example = {}
        valid = True
        for tarinfo in stream:
            name = tarinfo.name
            pos = name.rfind('.')
            assert pos > 0
            prefix, postfix = name[:pos], name[pos + 1:]
            if prev_prefix is not None and prefix != prev_prefix:
                example['key'] = prev_prefix
                if valid:
                    yield example
                example = {}
                valid = True
            with stream.extractfile(tarinfo) as file_obj:
                try:
                    if postfix in ['spk']:
                        example[postfix] = file_obj.read().decode(
                            'utf8').strip()
                    elif postfix in AUDIO_FORMAT_SETS:
                        waveform, sample_rate = torchaudio.load(file_obj)
                        example['wav'] = waveform
                        example['sample_rate'] = sample_rate
                    else:
                        example[postfix] = file_obj.read()
                except Exception as ex:
                    valid = False
                    logging.warning('error to parse {}'.format(name))
            prev_prefix = prefix
        if prev_prefix is not None:
            example['key'] = prev_prefix
            yield example
        stream.close()
        if 'process' in sample:
            sample['process'].communicate()
        sample['stream'].close()


def parse_raw(data):
    """ Parse key/wav/spk from json line

        Args:
            data: Iterable[str], str is a json line has key/wav/spk

        Returns:
            Iterable[{key, wav, spk, sample_rate}]
    """
    for sample in data:
        assert 'src' in sample
        json_line = sample['src']
        obj = json.loads(json_line)
        assert 'key' in obj
        assert 'wav' in obj
        assert 'spk' in obj
        key = obj['key']
        wav_file = obj['wav']
        spk = obj['spk']
        try:
            waveform, sample_rate = torchaudio.load(wav_file)
            example = dict(key=key,
                           spk=spk,
                           wav=waveform,
                           sample_rate=sample_rate)
            yield example
        except Exception as ex:
            logging.warning('Failed to read {}'.format(wav_file))


def parse_segments(data):
    bdir, data = data
    with open(os.path.join(bdir, 'wav.scp')) as f:
        lines = f.readlines()
        assert len(lines) == 1
        wav = lines[0].split()[1]
        assert os.path.isfile(wav), f'Missing wave file {wav}.'

    waveform, sample_rate = None, None
    for sample_idx, sample in enumerate(data):
        assert 'src' in sample
        segment, utt, start_sec, end_sec = sample['src'].split()
        utt = utt[:-2]
        start_sec, end_sec = float(start_sec), float(end_sec)

        if waveform is None:
            waveform, sample_rate = torchaudio.load(wav)

        # print(start_sec, end_sec)
        waveform_segment = waveform[:, int(start_sec * sample_rate):int(end_sec * sample_rate)]
        slen = waveform_segment.shape[1]
        seg_len, seg_jump = int(1.2 * sample_rate), int(0.24 * sample_rate)
        start = 0
        # print(f'{bdir} {waveform_segment.shape} {start_sec} {end_sec} {slen} {seg_len} {seg_jump}\n')
        for start in range(0, slen - seg_len, seg_jump):
            key = f'{utt}-A_{sample_idx:04}-' \
                  f'{int(start / sample_rate * 100):08}-{int((start + seg_len) / sample_rate * 100):08}'
            assert start <= start + seg_len <= waveform_segment.shape[1], \
                f'Start: {start}, end: {start + seg_len}, shape: {waveform_segment.shape[1]}'
            example = dict(key=key,
                           spk=(start / sample_rate + start_sec, (start + seg_len) / sample_rate + start_sec),
                           wav=waveform_segment[:, start:start + seg_len],
                           sample_rate=sample_rate)
            # print(f'{start} {slen} {seg_len} {seg_jump} {key} {waveform_segment[:, start:start + seg_len].shape}\n')
            yield example

        # print(slen, start, seg_jump)
        if slen - start - seg_jump > 0.12 * sample_rate:
            key = f'{utt}-A_{sample_idx:04}-' \
                  f'{int((start + seg_jump) / sample_rate * 100):08}-{int(slen / sample_rate * 100):08}'
            assert start + seg_jump <= slen <= waveform_segment.shape[1],\
                f'Start: {start + seg_jump}, end: {slen}, shape: {waveform_segment.shape[1]}'
            example = dict(key=key,
                           spk=((start + seg_jump) / sample_rate + start_sec, slen / sample_rate + start_sec),
                           wav=waveform_segment[:, start + seg_jump:slen],
                           sample_rate=sample_rate)
            # print(f'{start} {slen} {seg_len} {seg_jump} {key}\n')
            yield example


def parse_feat(data):
    """ Parse key/feat/spk from json line

        Args:
            data: Iterable[str], str is a json line has key/feat/spk

        Returns:
            Iterable[{key, feat, spk}]
    """
    for sample in data:
        assert 'src' in sample
        json_line = sample['src']
        obj = json.loads(json_line)
        assert 'key' in obj
        assert 'feat' in obj
        assert 'spk' in obj
        key = obj['key']
        feat_ark = obj['feat']
        spk = obj['spk']
        try:
            feat = torch.from_numpy(kaldiio.load_mat(feat_ark))
            example = dict(key=key,
                           spk=spk,
                           feat=feat)
            yield example
        except Exception as ex:
            logging.warning('Failed to load {}'.format(feat_ark))


def shuffle(data, shuffle_size=2500):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, wav, spk, sample_rate}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, wav, spk, sample_rate}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def spk_to_id(data, spk2id):
    """ Parse spk id

        Args:
            data: Iterable[{key, wav, spk, sample_rate}]
            spk2id: Dict[str, int]

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'spk' in sample
        if sample['spk'] in spk2id:
            label = spk2id[sample['spk']]
        else:
            label = sample['spk']
        sample['label'] = label
        yield sample


def speed_perturb(data, num_spks):
    """ Apply speed perturb to the data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    speeds = [1.0, 0.9, 1.1]
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        speed_idx = random.randint(0, 2)
        if speed_idx > 0:
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate,
                [['speed', str(speeds[speed_idx])], ['rate',
                                                     str(sample_rate)]])
            sample['wav'] = wav
            sample['label'] = sample['label'] + num_spks * speed_idx

        yield sample


def get_random_chunk(data, chunk_len):
    """ Get random chunk

        Args:
            data: torch.Tensor (random len)
            chunk_len: chunk length

        Returns:
            torch.Tensor (exactly chunk_len)
    """
    data_len = len(data)
    data_shape = data.shape
    # random chunk
    if data_len >= chunk_len:
        chunk_start = random.randint(0, data_len - chunk_len)
        data = data[chunk_start:chunk_start + chunk_len]
    else:
        # padding
        repeat_factor = chunk_len // data_len + 1
        repeat_shape = repeat_factor if len(data_shape) == 1 else (repeat_factor, 1)
        data = data.repeat(repeat_shape)
        data = data[:chunk_len]

    return data


def random_chunk(data, data_type='shard/raw/feat', num_frms=200):
    """ Random chunk the data into `num_frms` frames

        Args:
            data: Iterable[{key, wav/feat, label, sample_rate}]
            num_frms: num of frames for each training sample

        Returns:
            Iterable[{key, wav/feat, label, sample_rate}]
    """
    # Note(Binbin Zhang): We assume the sample rate is 16000,
    #                     frame shift 10ms, frame length 25ms
    if data_type == 'feat':
        chunk_len = num_frms
    else:
        chunk_len = (num_frms - 1) * 160 + 400

    for sample in data:
        assert 'key' in sample

        if data_type == 'feat':
            assert 'feat' in sample
            feat = sample['feat']
            feat = get_random_chunk(feat, chunk_len)
            sample['feat'] = feat
        else:
            assert 'wav' in sample
            wav = sample['wav'][0]
            try:
                wav = get_random_chunk(wav, chunk_len)
            except:
                logging.warning(f'{sample["key"]} probably empty. '
                                f'If you see this error often there might be a problem with your data.')
                continue
            sample['wav'] = wav.unsqueeze(0)
        yield sample


def add_reverb_noise(data, reverb_source, noise_source, aug_prob):
    """ Add reverb & noise aug

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            reverb_source: reverb LMDB data source
            noise_source: noise LMDB data source

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'wav' in sample
        assert 'key' in sample
        if aug_prob > random.random():
            augtype = random.randint(1, 2)
            if augtype == 1:
                # add reverb
                audio = sample['wav'].numpy()[0]
                audio_len = audio.shape[0]

                _, rir_data = reverb_source.random_one()
                _, rir_audio = wavfile.read(io.BytesIO(rir_data))
                rir_audio = rir_audio.astype(np.float32)
                rir_audio = rir_audio / np.sqrt(np.sum(rir_audio**2))
                out_audio = signal.convolve(audio, rir_audio,
                                            mode='full')[:audio_len]
            elif augtype == 2:
                # add noise
                audio = sample['wav'].numpy()[0]
                audio_len = audio.shape[0]
                audio_db = 10 * np.log10(np.mean(audio**2) + 1e-4)

                key, noise_data = noise_source.random_one()
                if key.startswith('noise'):
                    snr_range = [0, 15]
                elif key.startswith('speech'):
                    snr_range = [10, 30]
                elif key.startswith('music'):
                    snr_range = [5, 15]
                else:
                    snr_range = [0, 15]
                _, noise_audio = wavfile.read(io.BytesIO(noise_data))
                noise_audio = noise_audio.astype(np.float32) / (1 << 15)
                noise_audio = get_random_chunk(noise_audio, audio_len)
                noise_snr = random.uniform(snr_range[0], snr_range[1])
                noise_db = 10 * np.log10(np.mean(noise_audio**2) + 1e-4)
                noise_audio = np.sqrt(10**(
                    (audio_db - noise_db - noise_snr) / 10)) * noise_audio
                out_audio = audio + noise_audio

            # normalize into [-1, 1]
            out_audio = out_audio / (np.max(np.abs(out_audio)) + 1e-4)
            sample['wav'] = torch.from_numpy(out_audio).unsqueeze(0)

        yield sample


def compute_fbank(data,
                  num_mel_bins=80,
                  frame_length=25,
                  frame_shift=10,
                  dither=1.0):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        waveform = waveform * (1 << 15)
        # Only keep key, feat, label
        mat = kaldi.fbank(waveform,
                          num_mel_bins=num_mel_bins,
                          frame_length=frame_length,
                          frame_shift=frame_shift,
                          dither=dither,
                          sample_frequency=sample_rate,
                          window_type='hamming',
                          use_energy=False)
        yield dict(key=sample['key'], label=sample['label'], feat=mat)


def apply_cmvn(data, norm_mean=True, norm_var=False):
    """ Apply CMVN

        Args:
            data: Iterable[{key, feat, label}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'key' in sample
        assert 'feat' in sample
        assert 'label' in sample
        mat = sample['feat']
        if norm_mean:
            mat = mat - torch.mean(mat, dim=0)
        if norm_var:
            mat = mat / torch.sqrt(torch.var(mat, dim=0) + 1e-8)
        yield dict(key=sample['key'], label=sample['label'], feat=mat)


def spec_aug(data, num_t_mask=1, num_f_mask=1, max_t=10, max_f=8, prob=0.6):
    """ Do spec augmentation
        Inplace operation

        Args:
            data: Iterable[{key, feat, label}]
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
            prob: prob of spec_aug

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        if random.random() < prob:
            assert 'feat' in sample
            x = sample['feat']
            assert isinstance(x, torch.Tensor)
            # y = x.clone().detach()
            y = x.detach()  # inplace operation
            max_frames = y.size(0)
            max_freq = y.size(1)
            # time mask
            for i in range(num_t_mask):
                start = random.randint(0, max_frames - 1)
                length = random.randint(1, max_t)
                end = min(max_frames, start + length)
                y[start:end, :] = 0
            # freq mask
            for i in range(num_f_mask):
                start = random.randint(0, max_freq - 1)
                length = random.randint(1, max_f)
                end = min(max_freq, start + length)
                y[:, start:end] = 0
            sample['feat'] = y
        yield sample
