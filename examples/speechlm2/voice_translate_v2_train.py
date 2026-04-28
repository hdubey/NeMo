"""
VoiceTranslateV2 training entrypoint.

Mirrors duplex_eartts_train.py but instantiates VoiceTranslateV2 (full E2E model:
FastConformer + Riva-4B LLM + RVQEARTTSModel) instead of DuplexEARTTS.

Data pipeline: reuses DuplexEARTTSDataset (same Lhotse SHAR shards as VT-v1).
Batch format: DuplexEARTTSDataset provides the audio prompt + text tokens + target
audio codes. VoiceTranslateV2.forward() maps these to FastConformer input (source
speech) + LLM input (text) + TTS input (audio codes + LLM hidden states).

Usage (SLURM — see train_vt_v2_sft.sh):
    torchrun --nproc_per_node=$NGPUS voice_translate_v2_train.py \
        --config-path /path/to/configs \
        --config-name voice_translate_v2_sft \
        trainer.max_steps=200 trainer.limit_train_batches=100

    # smoke test:  trainer.max_steps=200 limit_train_batches=100 limit_val_batches=0
    # full run:    trainer.max_steps=200000 limit_val_batches=2
"""

import datetime
import os
import sys

import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from nemo.collections.speechlm2 import DataModule
from nemo.collections.speechlm2.data import DuplexEARTTSDataset
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

from nemo.collections.speechlm2.models.voice_translate_v2 import VoiceTranslateV2

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


@hydra_runner(config_path="conf", config_name="voice_translate_v2_sft")
def train(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=int(cfg.trainer.strategy.get("timeout", 3600))),
    )
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True

    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    with trainer.init_module():
        model = VoiceTranslateV2(OmegaConf.to_container(cfg, resolve=True))

    dataset = DuplexEARTTSDataset(
        tokenizer=model.tokenizer,
        frame_length=cfg.data.frame_length,
        source_sample_rate=cfg.data.source_sample_rate,
        target_sample_rate=cfg.data.target_sample_rate,
        input_roles=cfg.data.input_roles,
        output_roles=cfg.data.output_roles,
        add_text_bos_and_eos_in_each_turn=cfg.data.get("add_text_bos_and_eos_in_each_turn", True),
        add_audio_prompt=cfg.data.get("add_audio_prompt", True),
        audio_prompt_duration=cfg.data.get("audio_prompt_duration", 3),
        num_delay_speech_tokens=cfg.model.tts_config.get("num_delay_speech_tokens", 2),
    )
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
