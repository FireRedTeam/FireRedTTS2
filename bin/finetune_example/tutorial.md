## Part1.Data Preparation

We use LJSpeech as an example dataset for fine-tuning. You can download it from <https://keithito.com/LJ-Speech-Dataset/>.

### Step1.Create standard metadata

Considering that different open-source datasets are provided in varying formats or methods, in this step, we first aggregate the key information from the datasets to **facilitate the subsequent use of a unified feature extraction script.** We provide an example code based on LJSpeech(`bin/finetune_example/data_preparation/step1_create_meta.py`). If you have speech datasets in other languages or of other types, you can also easily write similar code yourself.

Run the following command:

```shell
python step1_create_meta.py --data_dir xxx/LJSpeech-1.1 --out_jsonl ./ljspeech.jsonl
```

After running the above command, you will obtain a `ljspeech.jsonl` file containing the essential information for each audio clip, such as duration, path, and text. Below are a few lines from this `.jsonl` file as a visual reference.

```json
{"segments": [{"duration": 6.450657596371882, "audio_path": "xxx/LJSpeech-1.1/wavs/LJ004-0094.wav", "speaker": "[S_DIALOG_1]", "text": "This act set forth that \"whereas the malignant fever commonly called the jail distemper"}]}
{"segments": [{"duration": 4.651111111111111, "audio_path": "xxx/LJSpeech-1.1/wavs/LJ004-0095.wav", "speaker": "[S_DIALOG_1]", "text": "is found to be owing to want of cleanliness and fresh air in the several jails,"}]}
```

We see that the **segments** list contains only one dictionary entry because LJSpeech consists of individual sentences rather than dialogue. If you have dialogue data, you will need to write your own code to add each dialogue turn to this list. In that case, the meta field might take the following form:

```json
{"segments": [{"duration": 6.45, "audio_path": "xxx/s1.wav", "speaker": "[S_DIALOG_1]", "text": "text_s1"},{"duration": 4.2, "audio_path": "xxx/s2.wav", "speaker": "[S_DIALOG_3]", "text": "text_s2"}]}
```

### Step2.Feature Extraction

To accelerate the training process, audio tokens need to be pre-extracted before the training phase rather than being dynamically generated during training. In the previous step, the dataset metadata has been written to a `.jsonl` file, from which we retrieve the audio information and perform quantization.

Run the following command:

```shell
python step2_extract_token.py --jsonl ./ljspeech.jsonl --pretrained_dir xxx/pretrained_models
```

After running the above command, you will obtain a `ljspeech_token.jsonl` file. The difference between this file and `ljspeech.jsonl` is that it includes an additional `audio_token` field, which is used to store the extracted audio tokens.

### Step3.Write to an `.arrow` file

In large model training, we utilize `.arrow` files with the Datasets library to prevent I/O bottlenecks. Therefore, the final step involves storing processed data into `.arrow` format for seamless training.

Execute this command to generate an `out_datasets` folder containing `ljspeech.arrow`:

```shell
python step3_write_arrow.py --jsonl ./ljspeech_token.jsonl --pretrained_dir xxx/pretrained_models --dataset_dir ./out_datasets --prefix ljspeech
```

To load `.arrow` files via the Datasets library, add these files under out_datasets: `dataset_info.json` (records data schema and types), `state.json` (tracks loaded `.arrow` files)

- **dataset_info.json**

```json
{
  "citation": "",
  "description": "",
  "features": {
    "audio": {
      "feature": {
        "dtype": "int64",
        "_type": "Value"
      },
      "_type": "Sequence"
    },
    "text": {
      "feature": {
        "dtype": "int64",
        "_type": "Value"
      },
      "_type": "Sequence"
    },
    "len": {
      "dtype": "int64",
      "_type": "Value"
    },
    "audio_segment_len": {
      "feature": {
        "dtype": "int64",
        "_type": "Value"
      },
      "_type": "Sequence"
    },
    "text_segment_len": {
      "feature": {
        "dtype": "int64",
        "_type": "Value"
      },
      "_type": "Sequence"
    }
  },
  "homepage": "",
  "license": ""
}
```

- **state.json**

```json
{
    "_data_files": [
        {
            "filename": "ljspeech.arrow"
        }
    ],
    "_fingerprint": "b8675e6c0eabe906",
    "_format_columns": null,
    "_format_kwargs": {},
    "_format_type": null,
    "_output_all_columns": false,
    "_split": null
}
```

## Part2. Finetuning

All fine-tuning related code and configuration files reside under **`bin/finetune_example`**. Simply modify the specified settings in the Config file to initiate training seamlessly.

### Step 1: Modify the Configuration File

First, edit the **`config_finetune_1.5b_0.2b.json`** file（`bin/finetune_example/config_finetune_1.5b_0.2b.json`）:

- Set `logs_folder` to your desired output directory (this will also store the final trained model).
- Point both `train_dataset_dir` and `valid_dataset_dir` to the dataset folder generated earlier (use the same path for both since validation is skipped during training).
- Configure parameters like `batch_size` according to your GPU resources.

```json
{
    "train": {
        "batch_size": 12,
        "lr": 0.000003,
        "n_epochs": 10,
        "warmup_steps": 12000,
        "weight_decay": 0.002,
        "lr_decay": "linear",
        "accumulate_num": 2,
        "max_grad_norm": 1.3,
        "keep_ckpts": 10,
        "log_every": 10,
        "val_every": 1000,
        "save_every": 1000,
        "gen_every": 1000,
        "num_workers": 4,
        "logs_folder": "xxx/finetune_logs"
    },
    "models": {
        "sample_rate": 16000,
        "backbone_flavor": "qwen-1.5b",
        "decoder_flavor": "qwen-200m",
        "text_vocab_size": 151936,
        "audio_vocab_size": 2051,
        "audio_num_codebooks": 16,
        "decoder_loss_weight": 0.6
    },
    "dataset": {
        "train_dataset_dir": "xxx/out_datasets",
        "valid_dataset_dir": "xxx/out_datasets"
    }
}
```

### Step2.Training

Execute the following command to initiate training:

```shell
accelerate launch --mixed_precision "fp16" posttrain.py --config_path ./config_finetune_1.5b_0.2b.json --checkpoint_path xxx/pretrained_models/llm_posttrain.pt
```

The training display appears as follows

```shell
--device: cuda ---epoch: 1 ---real_step: 2126 ---step: 517 ---total_step: 1063 ---total_loss: 7.59 ---text_loss: 6.1 --backbone_loss: 2.65 --decoder_loss: 4.51 ---learning_rate: 2.65e-07 ---grad_norm: 457.54
--device: cuda ---epoch: 1 ---real_step: 2128 ---step: 518 ---total_step: 1064 ---total_loss: 7.5 ---text_loss: 6.08 --backbone_loss: 2.56 --decoder_loss: 4.49 ---learning_rate: 2.6525e-07 ---grad_norm: 340.37
--device: cuda ---epoch: 1 ---real_step: 2130 ---step: 519 ---total_step: 1065 ---total_loss: 7.48 ---text_loss: 6.03 --backbone_loss: 2.54 --decoder_loss: 4.49 ---learning_rate: 2.655e-07 ---grad_norm: 393.64
--device: cuda ---epoch: 1 ---real_step: 2132 ---step: 520 ---total_step: 1066 ---total_loss: 7.53 ---text_loss: 6.06 --backbone_loss: 2.56 --decoder_loss: 4.52 ---learning_rate: 2.6575e-07 ---grad_norm: 229.78
--device: cuda ---epoch: 1 ---real_step: 2134 ---step: 521 ---total_step: 1067 ---total_loss: 7.49 ---text_loss: 6.01 --backbone_loss: 2.53 --decoder_loss: 4.5 ---learning_rate: 2.66e-07 ---grad_norm: 213.71
--device: cuda ---epoch: 1 ---real_step: 2136 ---step: 522 ---total_step: 1068 ---total_loss: 7.46 ---text_loss: 6.16 --backbone_loss: 2.57 --decoder_loss: 4.45 ---learning_rate: 2.6625e-07 ---grad_norm: 323.64
--device: cuda ---epoch: 1 ---real_step: 2138 ---step: 523 ---total_step: 1069 ---total_loss: 7.54 ---text_loss: 6.13 --backbone_loss: 2.59 --decoder_loss: 4.5 ---learning_rate: 2.665e-07 ---grad_norm: 356.46
```
