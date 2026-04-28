"""
VoiceTranslateV2 — unified 3-stage training entrypoint.

Runs Stage 1 (TTS warmup), Stage 2 (E2E SFT), and Stage 3 (GRPO) in a
single Trainer.fit() call.  Stage boundaries are controlled by a callback
that checks global_step.

Usage (SLURM — see train_vt_v2_stages.sh):
    torchrun --nproc_per_node=$NGPUS voice_translate_v2_train_stages.py \\
        --config-path /path/to/conf \\
        --config-name voice_translate_v2_stages \\
        training_stages.stage1_steps=50000 \\
        training_stages.stage2_steps=150000 \\
        training_stages.stage3_steps=2000

Smoke test (100 steps each stage):
    training_stages.stage1_steps=100 training_stages.stage2_steps=100 training_stages.stage3_steps=10
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

# VoiceTranslateV2Stages and StageTransitionCallback live in voice_translate_v2
from nemo.collections.speechlm2.models.voice_translate_v2 import (
    StageTransitionCallback,
    VoiceTranslateV2Stages,
)

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


@hydra_runner(config_path="conf", config_name="voice_translate_v2_stages")
def train(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=int(cfg.trainer.strategy.get("timeout", 3600))),
    )
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True

    stages = cfg.get("training_stages", {})
    s1 = int(stages.get("stage1_steps", 50_000))
    s2 = int(stages.get("stage2_steps", 150_000))
    s3 = int(stages.get("stage3_steps", 2_000))
    total_steps = s1 + s2 + s3

    # Patch trainer.max_steps to cover all stages
    OmegaConf.update(cfg, "trainer.max_steps", total_steps, merge=True)

    stage_cb = StageTransitionCallback(stage1_steps=s1, stage2_steps=s2)
    extra_callbacks = [stage_cb]

    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer), callbacks=extra_callbacks)
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    with trainer.init_module():
        model = VoiceTranslateV2Stages(OmegaConf.to_container(cfg, resolve=True))

        if model.cfg.get("pretrained_model", None):
            from nemo.collections.speechlm2.parts.pretrained import load_checkpoint
            model.restore_from_pretrained_checkpoint(model.cfg.pretrained_model)

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
