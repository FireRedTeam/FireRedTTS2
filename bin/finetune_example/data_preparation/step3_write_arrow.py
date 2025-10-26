import os
import argparse
import concurrent
from concurrent.futures import ProcessPoolExecutor
import json
import random
import librosa
import torch
import torchaudio
import pickle
from tqdm import tqdm


from datasets import Dataset as HF_Dataset
from datasets import Array2D, Sequence, Features, Value
from datasets.arrow_writer import ArrowWriter
from datasets import Audio
import numpy as np

from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing

from fireredtts2.llm.utils import load_custom_tokenizer


# config
MAX_AUDIO_DURATION = 35.0
MAX_TEXT_TOKEN_LEN = 180


def read_jsonl(path):
    path = os.path.expanduser(path)
    with open(path, "r") as f:
        json_str = f.read()
    data_list = []
    for line in json_str.splitlines():
        data = json.loads(line)
        data_list.append(data)
    return data_list


def get_speaker_dict(segments):
    """_summary_

    Args:
        segment (_type_): _description_
    """
    speaker_dict = {}
    num = 1
    for seg in segments:
        speaker = seg["speaker"]
        if speaker not in speaker_dict:
            speaker_dict[speaker] = "[S" + str(num) + "]"
            num += 1
    return speaker_dict


def pack(jsonl, pretrained_dir, dataset_dir, prefix):
    """pack .arrow

    Args:
        jsonl (_type_): _description_
        pretrained_dir (_type_): _description_
        dataset_dir (_type_): _description_
        prefix (_type_): _description_
    """
    # load tokenizers
    pretrained_qwen_path = os.path.join(pretrained_dir, "Qwen2.5-1.5B")
    assert os.path.exists(pretrained_qwen_path)
    text_tokenizer = load_custom_tokenizer(qwen2_tokenizer_path=pretrained_qwen_path)
    full_out_arrow_name = os.path.join(dataset_dir, prefix + ".arrow")

    # arrow definition
    column_features = Features(
        {
            "audio": Sequence(feature=Value("int64")),
            "text": Sequence(feature=Value("int64")),
            "len": Value("int64"),
            "audio_segment_len": Sequence(feature=Value("int64")),
            "text_segment_len": Sequence(feature=Value("int64")),
        }
    )
    arrow_writer = ArrowWriter(path=full_out_arrow_name, features=column_features)
    token_data_list = read_jsonl(path=jsonl)
    for i in tqdm(range(len(token_data_list))):
        token_data = token_data_list[i]
        is_useful = True
        segments = token_data["segments"]

        # Find all speakers
        speaker_dict = get_speaker_dict(segments=segments)
        if len(speaker_dict) > 5:
            print("Skipping: too many speakers....")
            is_useful = False

        audio_token_list = []
        text_token_list = []
        audio_segment_len = []
        text_segment_len = []
        total_len = 0

        for seg in segments:
            speaker = seg["speaker"]
            text = seg["text"]
            duration = seg["duration"]

            # audio tokens
            audio_tokens = seg["audio_tokens"]
            audio_tokens = np.array(audio_tokens)
            audio_tokens_flatten = audio_tokens.reshape([-1])

            # tokenize text
            text = speaker + "<|text_start|>" + text + "<|text_end|>"
            text_tokens = text_tokenizer.encode(text)

            if duration > MAX_AUDIO_DURATION:
                print("Skipping: audio exceeds 35s....")
                continue

            if len(text_tokens) > MAX_TEXT_TOKEN_LEN:
                print("Skipping: text too long...", text, len(text_tokens))
                continue

            audio_text_ratio = audio_tokens.shape[1] / len(text_tokens)

            if audio_text_ratio < 1.5 and len(text_tokens) > 30:
                print("Text-to-audio too dense — skipping....", text, len(text_tokens))
                continue
            if audio_text_ratio > 7.5:
                print("Skipping: insufficient text density....", text, len(text_tokens))
                continue

            audio_token_list.append(audio_tokens_flatten)
            text_token_list.append(text_tokens)
            audio_segment_len.append(len(audio_tokens_flatten))
            text_segment_len.append(len(text_tokens))

            # total lengths: don't forget extra EOS token len
            total_len += audio_tokens.shape[-1] + len(text_tokens) + 1

        if len(audio_token_list) > 0:
            audio_tokens_flatten = np.concatenate(audio_token_list, axis=0)
            text_tokens = np.concatenate(text_token_list, axis=0)
            arrow_writer.write(
                {
                    "audio": audio_tokens_flatten,
                    "text": text_tokens,
                    "len": total_len,
                    "audio_segment_len": audio_segment_len,
                    "text_segment_len": text_segment_len,
                }
            )
    arrow_writer.finalize()
    arrow_writer.close()


def recovery_debug(
    audio_tokens_flatten, text_tokens, audio_segment_len, text_segment_len
):
    """_summary_

    Args:
        audio_tokens_flatten (_type_): _description_
        text_tokens (_type_): _description_
        audio_segment_len (_type_): _description_
        text_segment_len (_type_): _description_
    """
    audio_segment_index = []
    text_segment_index = []

    start_audio_segment_index = 0
    start_text_segment_index = 0

    for i in range(len(audio_segment_len)):
        end_audio_segment_index = start_audio_segment_index + audio_segment_len[i]
        end_text_segment_index = start_text_segment_index + text_segment_len[i]
        audio_segment_index.append([start_audio_segment_index, end_audio_segment_index])
        text_segment_index.append([start_text_segment_index, end_text_segment_index])

        start_audio_segment_index = end_audio_segment_index
        start_text_segment_index = end_text_segment_index

    print("---audio_segment_index:\n", audio_segment_index)
    print("---text_segment_index:\n", text_segment_index)

    # 复原tokens
    for i in range(len(audio_segment_index)):
        audio_segment_tokens = audio_tokens_flatten[
            audio_segment_index[i][0] : audio_segment_index[i][1]
        ].reshape([16, -1])

        text_segment_tokens = text_tokens[
            text_segment_index[i][0] : text_segment_index[i][1]
        ]

        print(
            "---audio_segment_tokens:\n",
            audio_segment_tokens,
            audio_segment_tokens.shape,
        )

        print(
            "---text_segment_tokens:\n",
            text_segment_tokens,
            text_segment_tokens.shape,
        )

    print("\n\n\n")


def read_test(dataset_dir):
    """_summary_

    Args:
        dataset_dir (_type_): _description_
    """
    # read test
    ds = HF_Dataset.load_from_disk(dataset_path=dataset_dir)
    print("basic_info:", ds)
    print("---num_rows:", ds.num_rows)
    # print("ds[0]:\n", ds[0])

    all_len = ds[:]["len"]
    print("---all_len:\n", all_len)

    idx = 1

    for idx in range(ds.num_rows):

        audio_tokens_flatten = torch.tensor(ds[idx]["audio"], dtype=torch.long)
        text_tokens = torch.tensor(ds[idx]["text"], dtype=torch.long)
        total_len = ds[idx]["len"]
        audio_segment_len = ds[idx]["audio_segment_len"]
        text_segment_len = ds[idx]["text_segment_len"]

        recovery_debug(
            audio_tokens_flatten=audio_tokens_flatten,
            text_tokens=text_tokens,
            audio_segment_len=audio_segment_len,
            text_segment_len=text_segment_len,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--pretrained_dir", default="")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)

    args = parser.parse_args()

    print("################# Config Information #############")
    print("---jsonl:", args.jsonl)
    print("---pretrained_dir:", args.pretrained_dir)
    print("---dataset_dir:", args.dataset_dir)
    print("---prefix:", args.prefix)
    print("################################################\n")

    assert os.path.exists(args.jsonl)
    assert os.path.exists(args.pretrained_dir)

    os.makedirs(args.dataset_dir, exist_ok=True)

    pack(
        jsonl=args.jsonl,
        pretrained_dir=args.pretrained_dir,
        dataset_dir=args.dataset_dir,
        prefix=args.prefix,
    )

    # read_test(dataset_dir=args.dataset_dir)
