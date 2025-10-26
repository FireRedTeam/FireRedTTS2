import os
import sys
import tarfile
import argparse
import glob
import re
import json
import concurrent
from concurrent.futures import ProcessPoolExecutor
import torch
import torchaudio
import librosa
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

from pathlib import Path
from torch.nn.utils.rnn import pad_sequence

from io import BytesIO
from fireredtts2.codec import RedCodecInfer

from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")


torch.set_num_threads(2)


def read_jsonl(path):
    path = os.path.expanduser(path)
    with open(path, "r") as f:
        json_str = f.read()
    data_list = []
    for line in json_str.splitlines():
        data = json.loads(line)
        data_list.append(data)
    return data_list


def split_list_into_chunks(lst, chunk_size):
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def _load_one_audio(audio_path: str):
    audio, audio_sr = torchaudio.load(audio_path)
    audio16k = torchaudio.functional.resample(audio, audio_sr, 16000)
    return audio16k


def extract_tokens(jsonl, pretrained_dir):
    """_summary_

    Args:
        jsonl (_type_): _description_
        pretrained_dir (_type_): _description_
    """
    codec_config_path = os.path.join(pretrained_dir, "config_codec.json")
    codec_ckpt_path = os.path.join(pretrained_dir, "codec.pt")
    assert os.path.exists(codec_config_path)
    assert os.path.exists(codec_ckpt_path)

    # ==== Load Torch Audio Tokenizer ====
    device = torch.device("cuda")
    torch_codec = RedCodecInfer.from_pretrained(codec_config_path, codec_ckpt_path)
    torch_codec.eval()
    torch_codec = torch_codec.to(device)
    print("[INFO] Codec Loaded...")

    input_jsonl_file = jsonl
    output_jsonl_file = jsonl[:-6] + "_token" + ".jsonl"
    data_list = read_jsonl(path=input_jsonl_file)

    f_out = open(output_jsonl_file, "w", encoding="utf-8")

    for data_dict in tqdm(data_list):
        for segment in data_dict["segments"]:
            audio_path = segment["audio_path"]
            duration = segment["duration"]
            audio16k = _load_one_audio(audio_path=audio_path)
            audio16k_length = torch.tensor([audio16k.shape[1]], dtype=torch.long)

            token, token_length = torch_codec.encode(
                audio16k.to(device), audio16k_length.to(device), batch_size=1
            )

            desire_len = int(duration * 12.5)
            if abs(token.shape[-1] - desire_len) > 2:
                print("---wrong token length,skip!!：", audio_path)
                continue

            one_token = token.squeeze().cpu().numpy().tolist()
            segment["audio_tokens"] = one_token

        json.dump(data_dict, f_out, ensure_ascii=False)
        f_out.write("\n")

    f_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default="")
    parser.add_argument("--pretrained_dir", default="")

    args = parser.parse_args()
    assert os.path.exists(args.jsonl)
    assert os.path.exists(args.pretrained_dir)

    extract_tokens(jsonl=args.jsonl, pretrained_dir=args.pretrained_dir)
