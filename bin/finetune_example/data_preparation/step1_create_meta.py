import os
import sys
import tarfile
import argparse
import glob
import re
import json
import librosa

from tqdm import tqdm


def get_uttrid2path(wav_dir):
    """_summary_

    Args:
        wav_dir (_type_): _description_
    """
    uttrid2path = {}
    wav_files = os.listdir(wav_dir)
    for wav_file in wav_files:
        uttr_id = wav_file.split(sep=".")[0]
        full_path = os.path.join(wav_dir, wav_file)
        uttrid2path[uttr_id] = full_path

    return uttrid2path


def create_meta(data_dir, out_jsonl):
    """_summary_

    Args:
        data_dir (_type_): LJspeech root path
        out_jsonl (_type_): output meta
    """
    wav_dir = os.path.join(data_dir, "wavs")
    meta_path = os.path.join(data_dir, "metadata.csv")
    assert os.path.exists(wav_dir)
    assert os.path.exists(meta_path)

    uttrid2path = get_uttrid2path(wav_dir=wav_dir)

    # print("---uttrid2path:\n", uttrid2path)

    f_in = open(meta_path)
    lines = f_in.readlines()
    f_in.close()

    f_out = open(file=out_jsonl, mode="w")

    for line in tqdm(lines):
        out_dict = {}
        uttr_id, text, tn_text = line.strip().split(sep="|")
        # debug
        # print("---uttr_id:", uttr_id)
        # print("---text:", text)
        # print("---tn_text:", tn_text)

        # segment
        segments = []
        if uttr_id in uttrid2path:
            segment_dict = {}

            full_path = uttrid2path[uttr_id]
            dur = librosa.get_duration(filename=full_path)

            segment_dict["duration"] = dur
            segment_dict["audio_path"] = full_path
            segment_dict["speaker"] = "[S_DIALOG_1]"
            segment_dict["text"] = tn_text
            segments.append(segment_dict)

            out_dict["segments"] = segments

            json.dump(out_dict, f_out, ensure_ascii=False)
            f_out.write("\n")

    f_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="")
    parser.add_argument("--out_jsonl", default="")

    args = parser.parse_args()
    assert os.path.exists(args.data_dir)

    create_meta(data_dir=args.data_dir, out_jsonl=args.out_jsonl)
