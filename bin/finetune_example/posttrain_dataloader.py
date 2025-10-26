import os
from pathlib import Path
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence


import random

# dataset
from datasets import Dataset as HF_Dataset
from datasets import Array2D, Sequence, Features, Value
from datasets.arrow_writer import ArrowWriter

AUDIO_NUM_CODEBOOKS = 16


class TokenizedDataset(Dataset):
    """
    .arrow-backed dataset for tokenized audio and text samples.
    Assumes audio is saved as flat vlen int32 arrays (flattened [n_codebooks, seq_len]).
    """

    def __init__(self, dataset_dir: str, device):
        """_summary_

        Args:
            dataset_dir (str): _description_
            device (_type_): for debug
        """
        self.dataset_dir = dataset_dir
        self.hf_dataset = HF_Dataset.load_from_disk(dataset_path=dataset_dir)
        print("---dataset_info:\n", self.hf_dataset)
        print("---num_rows:", self.hf_dataset.num_rows)

        self.device = device

    def __len__(self):
        return self.hf_dataset.num_rows

    def get_seq_len(self):
        seq_lengths = self.hf_dataset[:]["len"]
        return seq_lengths

    def __getitem__(self, idx: int):
        # print("---device:", self.device, "---idx:", idx)

        flat_audio = self.hf_dataset[idx]["audio"]
        text = self.hf_dataset[idx]["text"]
        audio_segment_len = self.hf_dataset[idx]["audio_segment_len"]
        text_segment_len = self.hf_dataset[idx]["text_segment_len"]

        # to tensor
        flat_audio = torch.tensor(flat_audio, dtype=torch.long)
        text = torch.tensor(text, dtype=torch.long)

        return {
            "audio": flat_audio,
            "text": text,
            "audio_segment_len": audio_segment_len,
            "text_segment_len": text_segment_len,
        }


def get_index(audio_segment_len, text_segment_len):
    """_summary_

    Args:
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

        # 记录index
        audio_segment_index.append([start_audio_segment_index, end_audio_segment_index])
        text_segment_index.append([start_text_segment_index, end_text_segment_index])

        start_audio_segment_index = end_audio_segment_index
        start_text_segment_index = end_text_segment_index

    return audio_segment_index, text_segment_index


def interleave(audio_tokens_flatten, text_tokens, audio_segment_len, text_segment_len):
    """_summary_

    Args:
        audio_tokens_flatten (_type_): _description_
        text_tokens (_type_): _description_
        audio_segment_len (_type_): _description_
        text_segment_len (_type_): _description_
    """
    AUDIO_MAX_LEN = 2300
    TOTAL_MAX_LEN = 3100

    # step1.
    audio_segment_index, text_segment_index = get_index(
        audio_segment_len, text_segment_len
    )

    # step2.
    MAX_SEGMENTS = 16
    if len(audio_segment_index) > MAX_SEGMENTS:
        start_index = random.randint(0, len(audio_segment_index) - MAX_SEGMENTS)
        end_index = start_index + MAX_SEGMENTS
        audio_segment_index = audio_segment_index[start_index:end_index]
        text_segment_index = text_segment_index[start_index:end_index]

    total_audio_len = (
        audio_segment_index[-1][-1] - audio_segment_index[0][0]
    ) // AUDIO_NUM_CODEBOOKS

    # step3.
    if total_audio_len > AUDIO_MAX_LEN:
        start_index = 0
        end_index = len(audio_segment_index) - 1
        while True:
            random_num = random.random()
            if random_num <= 0.5:
                start_index += 1
            else:
                end_index -= 1

            new_total_audio_len = (
                audio_segment_index[end_index][-1] - audio_segment_index[start_index][0]
            ) // AUDIO_NUM_CODEBOOKS

            if start_index == end_index:
                print("start_index == end_index", start_index, "==", end_index)
                break

            if new_total_audio_len <= AUDIO_MAX_LEN:
                audio_segment_index = audio_segment_index[start_index:end_index]
                text_segment_index = text_segment_index[start_index:end_index]
                break

    # step4.
    interleave_tokens = []
    interleave_masks = []
    len_recorder = 0
    for i in range(len(audio_segment_index)):
        text_segment_tokens = text_tokens[
            text_segment_index[i][0] : text_segment_index[i][1]
        ]
        audio_segment_tokens = audio_tokens_flatten[
            audio_segment_index[i][0] : audio_segment_index[i][1]
        ].reshape([AUDIO_NUM_CODEBOOKS, -1])

        len_recorder += text_segment_tokens.shape[0] + audio_segment_tokens.shape[1]
        if len_recorder > TOTAL_MAX_LEN:
            break

        # Add EOS frame to audio
        eos_frame = torch.zeros(audio_segment_tokens.size(0), 1, dtype=torch.long)
        audio_segment_tokens = torch.cat([audio_segment_tokens, eos_frame], dim=1)

        # extra dimension is for text tokens
        audio_frame = torch.zeros(
            audio_segment_tokens.size(1), AUDIO_NUM_CODEBOOKS + 1
        ).long()
        audio_frame[:, :-1] = audio_segment_tokens.transpose(0, 1)
        audio_frame_mask = torch.zeros(
            audio_segment_tokens.size(1), AUDIO_NUM_CODEBOOKS + 1
        ).bool()
        audio_frame_mask[:, :-1] = True

        # Format text frame with same shape
        text_frame = torch.zeros(
            len(text_segment_tokens), AUDIO_NUM_CODEBOOKS + 1
        ).long()
        text_frame[:, -1] = text_segment_tokens
        text_frame_mask = torch.zeros(
            len(text_segment_tokens), AUDIO_NUM_CODEBOOKS + 1
        ).bool()
        text_frame_mask[:, -1] = True

        interleave_tokens.append(text_frame)
        interleave_tokens.append(audio_frame)

        interleave_masks.append(text_frame_mask)
        interleave_masks.append(audio_frame_mask)

    # concat interleave
    interleave_tokens_concat = torch.cat(tensors=interleave_tokens, dim=0)
    interleave_masks_concat = torch.cat(tensors=interleave_masks, dim=0)

    return interleave_tokens_concat, interleave_masks_concat


def collate_fn(batch: List[dict]):
    """
    Collate function for tokenized audio and text.
    Merges variable-length audio/text into a single padded tensor.
    """
    tokens, tokens_mask = [], []

    for item in batch:
        audio_tokens = item["audio"]  # [audio_seq_len*16]
        text_tokens = item["text"]  # [text_seq_len]
        audio_segment_len = item["audio_segment_len"]
        text_segment_len = item["text_segment_len"]

        # interleave format
        interleave_tokens_concat, interleave_masks_concat = interleave(
            audio_tokens_flatten=audio_tokens,
            text_tokens=text_tokens,
            audio_segment_len=audio_segment_len,
            text_segment_len=text_segment_len,
        )

        # Concatenate and collect
        tokens.append(interleave_tokens_concat)
        tokens_mask.append(interleave_masks_concat)

    tokens = pad_sequence(tokens, batch_first=True)
    tokens_mask = pad_sequence(tokens_mask, batch_first=True, padding_value=False)

    return tokens, tokens_mask


class BucketSampler(Sampler):
    """
    Groups samples of similar lengths into bins to minimize padding.
    """

    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        shuffle: bool = True,
        is_infinite: bool = True,
        random_seed: int = 42,
    ):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.is_infinite = is_infinite
        self.random_seed = random_seed
        self.local_step = 0
        self.bins = self._create_bins(lengths, batch_size)

    def _create_bins(self, lengths: List[int], batch_size: int) -> List[List[int]]:
        indices_with_lengths = sorted(enumerate(lengths), key=lambda x: x[1])
        bins, current_bin = [], []

        for idx, _ in indices_with_lengths:
            current_bin.append(idx)
            if len(current_bin) >= batch_size:
                bins.append(current_bin)
                current_bin = []

        if current_bin:
            bins.append(current_bin)

        return bins

    def _shuffle_bins(self, epoch: int):
        rng = np.random.RandomState(epoch + self.random_seed)
        rng.shuffle(self.bins)
        for bin_ in self.bins:
            rng.shuffle(bin_)

    def __iter__(self):
        epoch = 0
        while True:
            if self.shuffle:
                self._shuffle_bins(epoch)
            for bin_indices in self.bins:
                yield bin_indices
                self.local_step += 1
            if not self.is_infinite:
                break
            epoch += 1

    def __len__(self):
        return len(self.bins)


def create_dataloaders(
    train_datasets: str,
    validation_datasets: str,
    batch_size: int,
    device,  # for debug
    infinite_train: bool = False,
    num_workers: int = 0,
):

    trainset = TokenizedDataset(dataset_dir=train_datasets, device=device)
    valset = TokenizedDataset(dataset_dir=validation_datasets, device=device)
    trainloader = DataLoader(
        trainset,
        # batch_sampler=trainsampler,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=12,
        pin_memory=True,
        shuffle=True,
    )

    valloader = DataLoader(
        valset,
        # batch_sampler=valsampler,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )

    return trainloader, valloader
