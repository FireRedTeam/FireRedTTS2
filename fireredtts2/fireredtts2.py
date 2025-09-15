import os
import time
import json
import torch
import torchaudio

from typing import List, Tuple
from fireredtts2.codec import RedCodecInfer
from fireredtts2.llm import load_llm_model, load_custom_tokenizer
from fireredtts2.llm.utils import Segment
from fireredtts2.utils.spliter import clean_text, split_text, process_text_list
from tqdm import tqdm


class FireRedTTS2:
    def __init__(self, pretrained_dir, gen_type, device):

        assert os.path.exists(pretrained_dir)
        assert gen_type in ["monologue", "dialogue"]
        llm_config_path = os.path.join(pretrained_dir, "config_llm.json")
        if gen_type == "monologue":
            llm_ckpt_path = os.path.join(pretrained_dir, "llm_pretrain.pt")
        else:
            llm_ckpt_path = os.path.join(pretrained_dir, "llm_posttrain.pt")
        codec_config_path = os.path.join(pretrained_dir, "config_codec.json")
        codec_ckpt_path = os.path.join(pretrained_dir, "codec.pt")
        pretrained_qwen_path = os.path.join(pretrained_dir, "Qwen2.5-1.5B")

        # check
        assert os.path.exists(llm_config_path)
        assert os.path.exists(llm_ckpt_path)
        assert os.path.exists(codec_config_path)
        assert os.path.exists(codec_ckpt_path)
        assert os.path.exists(pretrained_qwen_path)

        # ==== Load Torch LLM ====
        llm_config = json.load(open(llm_config_path))
        self._model = load_llm_model(
            configs=llm_config, checkpoint_path=llm_ckpt_path, device=device
        )
        self._model.eval()
        self._model.setup_caches(1)
        print("[INFO] LLM Loaded...")

        # ==== Load Qwen2.5 Text Tokenizer ====
        self._text_tokenizer = load_custom_tokenizer(pretrained_qwen_path)
        print("[INFO] Text Tokenizer Loaded...")

        # ==== Load Torch Audio Tokenizer ====
        torch_codec = RedCodecInfer.from_pretrained(codec_config_path, codec_ckpt_path)
        torch_codec.eval()
        self._audio_tokenizer = torch_codec.to(device)
        print("[INFO] Codec Loaded...")

        self.sample_rate = 16000
        self.device = device
        self.max_seq_len = 3100

    def load_prompt_audio(self, audio_path) -> torch.Tensor:
        audio, audio_sr = torchaudio.load(audio_path)
        # Convert to mono if multi-channel
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        audio16k = torchaudio.functional.resample(audio, audio_sr, 16000)
        return audio16k

    def prepare_prompt(self, text, speaker, audio_path) -> Segment:
        audio_tensor = self.load_prompt_audio(audio_path)
        return Segment(text=text, speaker=speaker, audio=audio_tensor)

    def _tokenize_text_segment(
        self, text: str, speaker: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        text = speaker + "<|text_start|>" + text + "<|text_end|>"
        text_tokens = self._text_tokenizer.encode(text)
        text_frame = torch.zeros(len(text_tokens), 17).long()
        text_frame_mask = torch.zeros(len(text_tokens), 17).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio_length = torch.tensor([audio.shape[1]], dtype=torch.long)
        audio_tokens, token_length = self._audio_tokenizer.encode(
            audio.to(self.device),
            audio_length.to(self.device),
            batch_size=48,
        )

        audio_tokens = audio_tokens.squeeze(0)
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 17).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 17).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (seq_len,17), (seq_len, 17)
        """
        text_tokens, text_masks = self._tokenize_text_segment(
            segment.text, segment.speaker
        )
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat(
            [text_masks, audio_masks], dim=0
        )

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: str,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 20,
    ) -> torch.Tensor:
        self._model.reset_caches()

        max_generation_len = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(
            text, speaker
        )
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = (
            torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)
        )

        max_seq_len = 3100
        max_context_len = max_seq_len - max_generation_len
        if curr_tokens.size(1) >= max_context_len:
            raise ValueError(
                f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}"
            )

        for _ in range(max_generation_len):
            sample = self._model.generate_frame(
                curr_tokens, curr_tokens_mask, curr_pos, temperature, topk
            )
            # eos
            if torch.all(sample == 0):
                break

            samples.append(sample)

            curr_tokens = torch.cat(
                [sample, torch.zeros(1, 1).long().to(self.device)], dim=1
            ).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [
                    torch.ones_like(sample).bool(),
                    torch.zeros(1, 1).bool().to(self.device),
                ],
                dim=1,
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        audio = (
            self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0))
            .squeeze(0)
            .squeeze(0)
        )

        return audio

    # @torch.inference_mode()
    # def generate_stream(
    #     self,
    #     text: str,
    #     speaker: str,
    #     context: List[Segment],
    #     max_audio_length_ms: float = 90_000,
    #     temperature: float = 0.9,
    #     topk: int = 50,
    # ):
    #     self._model.reset_caches()

    #     max_generation_len = int(max_audio_length_ms / 80)
    #     tokens, tokens_mask = [], []
    #     for segment in context:
    #         segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
    #         tokens.append(segment_tokens)
    #         tokens_mask.append(segment_tokens_mask)

    #     gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(
    #         text, speaker
    #     )
    #     tokens.append(gen_segment_tokens)
    #     tokens_mask.append(gen_segment_tokens_mask)

    #     prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
    #     prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

    #     samples = []
    #     curr_tokens = prompt_tokens.unsqueeze(0)
    #     curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
    #     curr_pos = (
    #         torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)
    #     )

    #     max_seq_len = 3100
    #     max_context_len = max_seq_len - max_generation_len
    #     if curr_tokens.size(1) >= max_context_len:
    #         raise ValueError(
    #             f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}"
    #         )

    #     # codec cache
    #     codec_cache = {}
    #     prev_sample = None

    #     for _ in range(max_generation_len):
    #         sample = self._model.generate_frame(
    #             curr_tokens, curr_tokens_mask, curr_pos, temperature, topk
    #         )
    #         # eos
    #         if torch.all(sample == 0):
    #             break

    #         # decode one token
    #         if prev_sample is None:
    #             prev_sample = sample  # sample: (b, nq)
    #         else:
    #             audio_chunk, codec_cache = self._audio_tokenizer.decode_one_token(
    #                 prev_sample.unsqueeze(-1),
    #                 codec_cache,
    #                 last_token=False,
    #             )
    #             yield audio_chunk.squeeze(0)
    #             prev_sample = sample
    #         samples.append(sample)  # sample: (b, nq)

    #         curr_tokens = torch.cat(
    #             [sample, torch.zeros(1, 1).long().to(self.device)], dim=1
    #         ).unsqueeze(1)
    #         curr_tokens_mask = torch.cat(
    #             [
    #                 torch.ones_like(sample).bool(),
    #                 torch.zeros(1, 1).bool().to(self.device),
    #             ],
    #             dim=1,
    #         ).unsqueeze(1)
    #         curr_pos = curr_pos[:, -1:] + 1

    #     audio_chunk, codec_cache = self._audio_tokenizer.decode_one_token(
    #         prev_sample.unsqueeze(-1),
    #         codec_cache,
    #         last_token=True,
    #     )
    #     yield audio_chunk.squeeze(0)

    @torch.inference_mode()
    def generate_dialogue(
        self,
        text_list,
        prompt_wav_list=None,
        prompt_text_list=None,
        temperature=0.9,
        topk=20,
    ):
        all_generated_segments = []
        all_storage_segments = []
        prompt_segments = []
        text_list = process_text_list(text_list=text_list)
        if prompt_wav_list is not None:
            assert len(prompt_wav_list) == len(prompt_text_list)
            # Prepare prompts
            for i in range(len(prompt_wav_list)):
                prompt_wav = prompt_wav_list[i]
                prompt_text = prompt_text_list[i]
                speaker = prompt_text[:4]
                assert speaker in ["[S1]", "[S2]", "[S3]", "[S4]"]
                prompt_segments.append(
                    self.prepare_prompt(
                        text=prompt_text, speaker=speaker, audio_path=prompt_wav
                    )
                )

        for text in tqdm(text_list):
            speaker = text[:4]
            text = text[4:]
            # print("---speaker:", speaker)
            # print("---text:", text)
            assert speaker in ["[S1]", "[S2]", "[S3]", "[S4]"]

            audio_tensor = self.generate(
                text=text,
                speaker=speaker,
                context=prompt_segments + all_generated_segments,
                max_audio_length_ms=30_000,
                temperature=temperature,
                topk=topk,
            )

            # 做上下文管理的时候需要将audio 转到16k
            audio_16k = torchaudio.functional.resample(
                audio_tensor.unsqueeze(0), 24000, 16000
            )
            all_generated_segments.append(
                Segment(text=text, speaker=speaker, audio=audio_16k)
            )

            all_storage_segments.append(
                Segment(text=text, speaker=speaker, audio=audio_tensor.unsqueeze(0))
            )

        # Concatenate all generations
        all_audio = torch.cat([seg.audio for seg in all_storage_segments], dim=1)
        all_audio = all_audio.cpu()
        return all_audio

    @torch.inference_mode()
    def generate_monologue(
        self, text, prompt_wav=None, prompt_text=None, temperature=0.75, topk=20
    ):
        self._model.reset_caches()
        max_generation_len = 400  # ~32s

        # step1. construct context
        if prompt_wav is not None:
            assert os.path.exists(prompt_wav)
            assert prompt_text is not None
            prompt_text = clean_text(text=prompt_text.strip())
            text = clean_text(text=text.strip())

            # print("---prompt_text:", prompt_text)
            # print("---text:", text)

            input_text = prompt_text + text

            prompt_a = self.prepare_prompt(
                text=input_text, speaker="[S1]", audio_path=prompt_wav
            )
            # print("---prompt_a:", prompt_a)

            context = [prompt_a]

            tokens, tokens_mask = [], []
            for segment in context:
                segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
                tokens.append(segment_tokens)
                tokens_mask.append(segment_tokens_mask)

            prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
            prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)
            prompt_tokens = prompt_tokens[:-3, :]
            prompt_tokens_mask = prompt_tokens_mask[:-3, :]

            samples = []
            curr_tokens = prompt_tokens.unsqueeze(0)
            curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
            curr_pos = (
                torch.arange(0, prompt_tokens.size(0))
                .unsqueeze(0)
                .long()
                .to(self.device)
            )

            max_seq_len = 3100
            max_context_len = max_seq_len - max_generation_len
            if curr_tokens.size(1) >= max_context_len:
                raise ValueError(
                    f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}"
                )

            num_token = 0
            start_time = time.time()
            for _ in range(max_generation_len):
                sample = self._model.generate_frame(
                    curr_tokens, curr_tokens_mask, curr_pos, temperature, topk
                )
                # eos
                if torch.all(sample == 0):
                    break

                samples.append(sample)

                curr_tokens = torch.cat(
                    [sample, torch.zeros(1, 1).long().to(self.device)], dim=1
                ).unsqueeze(1)
                curr_tokens_mask = torch.cat(
                    [
                        torch.ones_like(sample).bool(),
                        torch.zeros(1, 1).bool().to(self.device),
                    ],
                    dim=1,
                ).unsqueeze(1)
                curr_pos = curr_pos[:, -1:] + 1
                num_token += 1
                if num_token == 2:
                    end_time = time.time()
                    duration = end_time - start_time
                    print("---first pack duration:", duration)

            gen_tokens = torch.stack(samples).permute(1, 2, 0)
            # print("---gen_tokens:\n", gen_tokens, gen_tokens.shape)
            gen_tokens = gen_tokens[:, :, 1:]  # cut leading silence
            audio = self._audio_tokenizer.decode(gen_tokens).squeeze(0).squeeze(0)

            return audio

        else:
            # random speaker
            text = clean_text(text=text.strip())
            audio_tensor = self.generate(
                text=text,
                speaker="[S1]",
                context=[],
                max_audio_length_ms=30_000,
                temperature=temperature,
                topk=topk,
            )
            return audio_tensor
