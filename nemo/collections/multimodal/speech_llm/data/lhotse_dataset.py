import logging
import random

import torch.utils.data
from lhotse.cut import Cut, CutSet, MixedCut
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import _read_features
from lhotse.dataset.collation import collate_vectors as collate_vectors_lhotse

from nemo.collections.common.data.lhotse.text_adapters import (
    AudioTurn,
    NeMoMultimodalConversation,
    NeMoSFTExample,
    SourceTargetTextExample,
)

from nemo.collections.multimodal.speech_llm.parts.utils.data_utils import (
    PromptFormatterTextProcessing,
    build_loss_mask,
    ceil_to_nearest,
)
from nemo.utils import logging

import logging
import random

import torch.utils.data
from lhotse import CutSet
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import _read_features
from lhotse.dataset.collation import collate_vectors as collate_vectors_lhotse

from nemo.collections.multimodal.speech_llm.parts.utils.data_utils import (
    TextProcessing,
    build_loss_mask,
    ceil_to_nearest,
)

def collate_vectors(items, max_length: int, padding_value):
    vectors = collate_vectors_lhotse(items, padding_value=padding_value)
    if max_length > vectors.size(1):
        vectors = torch.cat(
            [vectors, padding_value * torch.ones(vectors.size(0), max_length - vectors.size(1), dtype=vectors.dtype)],
            dim=1,
        )
    if items[0].shape[0] < 1:
        vectors = vectors.long()
    return vectors


# TODO: the changes in this file needed to be moved out as a derived class
class LhotseAudioQuestionAnswerDataset(torch.utils.data.Dataset):
    """
    This dataset is based on Lhotse ASR dataset from ``audio_to_text_lhotse.py``
    and ``TarredAudioQuestionAnswerDataset`` from ``audio_text_qa_dataset.py``.

    Unlike native NeMo datasets, Lhotse dataset defines only the mapping from
    a CutSet (meta-data) to a mini-batch with PyTorch tensors.
    Specifically, it performs tokenization, I/O, augmentation, and feature extraction (if any).
    Managing data, sampling, de-duplication across workers/nodes etc. is all handled
    by Lhotse samplers instead.

    Args:
        text_processor: TextProcessing object
        default_context: Default question to use if no question is provided
        tokens_to_generate: Number of tokens to generate during inference
        pad_to_max_length: Whether to pad the input to the max sequence length. If False, will pad to the max length of the current batch.
        max_seq_length: Maximum sequence length for each dataset examples. Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        context_key: Key to use for the context in your JSONL file
        default_context_key: Key to use for the default context in lhotse yaml
    """

    def __init__(
        self,
        text_processor: TextProcessing,
        default_context: str,
        tokens_to_generate: int,
        pad_to_max_length: bool,
        max_seq_length: int,
        context_key: str = "context",
        default_context_key: str = "default_context",
        vocab_sizes: list[int] = [-1],
        decoder_reduction_factor: int = 1,
        speech_pad_id: int = 1001,
        speech_unk_id: int = 1002,
        speech_bos_id: int = 1003,
        speech_eos_id: int = 1004,
        filter_by_source_target_text_ratio: bool = False,
        source_target_text_ratio_limit: float = 1.0,
        sample_rate: int = 22050,
        t5_style: bool = False,
    ):
        super().__init__()
        self.text_processor = text_processor
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.tokens_to_generate = tokens_to_generate
        self.pad_to_max_length = pad_to_max_length
        self.max_seq_length = max_seq_length

        self.default_context = default_context
        self.context_key = context_key
        self.default_context_key = default_context_key

        if len(vocab_sizes) == 1 and vocab_sizes[0] <= 0:
            vocab_sizes = [self.text_processor.tokenizer.vocab_size]
        self.vocab_sizes = list(vocab_sizes)
        self.n_speech_codebooks = len(self.vocab_sizes) - 1
        self.decoder_reduction_factor = decoder_reduction_factor
        self.speech_pad_id = speech_pad_id
        self.speech_unk_id = speech_unk_id
        self.speech_bos_id = speech_bos_id
        self.speech_eos_id = speech_eos_id
        self.filter_by_source_target_text_ratio = filter_by_source_target_text_ratio
        self.source_target_text_ratio_limit = source_target_text_ratio_limit
        self.sample_rate = sample_rate

        # To be consistent with SALM text processor
        self.text_processor.add_sep = False
        self.text_processor.max_seq_length = (
            4096  # Set this to a large number for since the speech sequence can be long
        )
        self.t5_style = t5_style

def __getitem__(self, cuts: CutSet) -> dict[str, torch.Tensor | list[str] | dict]:
    ans = {}

    # Step 1: Filter and load audio cuts
    audio_cuts = cuts.filter(lambda c: isinstance(c, Cut))
    if audio_cuts:
        # In this case, we expect that each `Cut` in the CutSet contains both the user request audio and the voice assistant response.
        # We'll extract both audio streams (user request and assistant response).
        user_request_audio = []
        assistant_response_audio = []
        audio_lens = []

        # Iterate over each cut and load the user request and assistant response audio
        for cut in audio_cuts:
            if hasattr(cut, 'user_request_audio') and hasattr(cut, 'assistant_response_audio'):
                # Load both audio streams (assuming the cut has attributes for both audio streams)
                user_request_audio.append(cut.user_request_audio.load_audio())
                assistant_response_audio.append(cut.assistant_response_audio.load_audio())
                audio_lens.append(cut.duration)

        # Ensure the lengths of the user request and assistant response are equal (if not, resample or pad).
        max_length = max(audio_lens)
        user_request_audio = collate_vectors(user_request_audio, max_length=max_length, padding_value=0.0)
        assistant_response_audio = collate_vectors(assistant_response_audio, max_length=max_length, padding_value=0.0)
        audio_lens = torch.FloatTensor(audio_lens)

        # Step 2: Add the audio data to the return batch
        audio_ratio = [1.0] * len(audio_cuts)

        metadata = [{'audio_filepath': cut.id + '.wav'} for cut in audio_cuts]

        # Collate text data as usual
        collated_text_data = collate_text_data(
            cuts=audio_cuts,
            default_context=self.default_context,
            text_processor=self.text_processor,
            tokens_to_generate=self.tokens_to_generate,
            pad_to_max_length=self.pad_to_max_length,
            max_seq_length=self.max_seq_length,
        )

        # Step 3: Build the return batch with two separate audio streams and text data
        ans.update({
            "sample_ids": list(audio_cuts.ids),
            "user_request_audio": user_request_audio,  # First stream: User request
            "assistant_response_audio": assistant_response_audio,  # Second stream: Assistant response
            "audio_signal_length": audio_lens,
            "audio_ratio": torch.FloatTensor(audio_ratio),
            "metadata": metadata,
            **collated_text_data,
        })

    return ans
        return return_batch


def collate_text_data(
    cuts,
    default_context: str,
    text_processor: TextProcessing,
    tokens_to_generate: int,
    pad_to_max_length: bool,
    max_seq_length: int,
) -> dict:
    """Perform text collation equivalent to nemo/collections/multimodal/data/audio_text_qa_dataset.py:121"""
    batch_size = len(cuts)
    pad_id = text_processor.pad_id
    examples = [
        {
            k: torch.as_tensor(v)
            for k, v in text_processor._process_example(
                context=cut.context,
                output=cut.supervisions[0].text if cut.supervisions[0].text is not None else "",
            ).items()
        }
        for cut in cuts
    ]
    fields = as_dict(examples)

    def get_max_len(input_list):
        return max([len(x) for x in input_list])

    max_length = tokens_to_generate + max(
        get_max_len(fields["input_ids"]), get_max_len(fields["context_ids"]), get_max_len(fields["answer_ids"])
    )
    # increase max length to nearest multiple of 4 or 8
    if pad_to_max_length:
        max_length = max_seq_length
    else:
        max_length = min(max_seq_length, ceil_to_nearest(max_length, 8))

    all_tokens = collate_vectors(fields["input_ids"], max_length=max_length, padding_value=pad_id)
    full_lengths = torch.LongTensor([len(item) for item in fields["input_ids"]])

    assert max_length <= max_seq_length, f"{max_length=} <= {max_seq_length=}"

    return {
        "tokens": all_tokens[:, :-1],
        "tokens_length": full_lengths - 1,
        "labels": all_tokens[:, 1:],
        "loss_mask": collate_vectors(
            [torch.as_tensor(build_loss_mask(item)) for item in examples], max_length=max_length, padding_value=0
        )[:, 1:],
        "position_ids": torch.arange(max_length, dtype=torch.long).repeat(batch_size, 1),
        "contexts": collate_vectors(fields["context_ids"], max_length=max_length, padding_value=pad_id),
        "context_lengths": torch.LongTensor([len(seq) for seq in fields["context_ids"]]),
        "answers": collate_vectors(fields["answer_ids"], max_length=max_length, padding_value=pad_id),
        "max_length": torch.LongTensor([max_length] * batch_size),
        "context_ids": fields["context_ids"],
    }


def as_dict(arg: list[dict]) -> dict[str, list]:
    return {k: [item[k] for item in arg] for k in arg[0].keys()}
