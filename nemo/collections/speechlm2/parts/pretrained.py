# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import gc
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

import torch
from omegaconf import OmegaConf, open_dict
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

from nemo.collections.asr.models import ASRModel
from nemo.collections.speechlm2.modules import AudioPerceptionModule
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.tts.models import AudioCodecModel
from nemo.utils import logging

def load_pretrained_nemo(cls, model_path_or_name: str):
    """
    Load pretrained NeMo 1.0 model (inheriting from ModelPT). Works with ASR, TTS, codec models.

    Setting ``pretrained_weights=False`` returns a model that has identical architecture with the checkpoint,
    but is randomly initialized.
    """
    if Path(model_path_or_name).exists() and model_path_or_name.endswith(".nemo"):
        return cls.restore_from(model_path_or_name)
    else:
        return cls.from_pretrained(model_path_or_name)


def load_pretrained_hf(
    model_path_or_name: str,
    pretrained_weights: bool = True,
    dtype=torch.float32,
    trust_remote_code: bool = False,
):
    """
    Load pretrained HuggingFace AutoModelForCausalLM.

    Setting ``pretrained_weights=False`` returns a model with identical architecture but random weights.

    Args:
        model_path_or_name: Path or name of the model to load.
        pretrained_weights: Whether to load pretrained weights (True) or random init (False).
        dtype: Data type for the model.
        trust_remote_code: Whether to trust remote code (needed for some models like Nemotron).
    """
    if pretrained_weights:
        return AutoModelForCausalLM.from_pretrained(
            model_path_or_name, torch_dtype=dtype, trust_remote_code=trust_remote_code
        )
    else:
        config = AutoConfig.from_pretrained(model_path_or_name, trust_remote_code=trust_remote_code)
        return AutoModelForCausalLM.from_config(config, torch_dtype=dtype, trust_remote_code=trust_remote_code)


@contextmanager
def move_embedding(model):
    """Temporarily restores the embedding layer into HF LLM. Supports LoRA models."""
    if isinstance(model.llm, PeftModel):
        model.llm.base_model.model.model.embed_tokens = model.embed_tokens
    else:
        model.llm.model.embed_tokens = model.embed_tokens
    yield
    if isinstance(model.llm, PeftModel):
        del model.llm.base_model.model.model.embed_tokens
    else:
        del model.llm.model.embed_tokens


def setup_audio_codec(model: torch.nn.Module):
    """
    Sets up an ``AudioCodecModel``, initializing it from pretrained weights.
    The result is assigned to ``model.audio_codec`` attribute.

    Includes a workaround for PTL auto-downcasting the codec model to bf16 with bf16-true precision.
    """
    if hasattr(model, "audio_codec") and next(model.audio_codec.parameters()).dtype == torch.float:
        return  # skip if already set up and has the right dtype
    with fp32_precision():
        model.audio_codec = load_pretrained_nemo(AudioCodecModel, model.cfg.pretrained_audio_codec).eval()
    for p in model.audio_codec.parameters():
        p.requires_grad = False
    del model.audio_codec.discriminator  # free up some memory


def setup_speech_encoder(model: torch.nn.Module, pretrained_weights: bool = True):
    """
    Sets up an ``AudioPerceptionModule``, initializing its ``encoder`` and ``preprocessor``
    with a pretrained NeMo ``ASRModel``.
    The result is assigned to ``model.perception`` attribute and is trainable.

    If user config specifies encoder parameters, they override the pretrained model's config.
    """
    if pretrained_weights:
        # Save user-specified encoder config before overwriting with pretrained model's config.
        user_encoder_config = {}
        if 'encoder' in model.cfg.perception:
            user_encoder_config = OmegaConf.to_container(model.cfg.perception.encoder, resolve=True)

        asr = load_pretrained_nemo(ASRModel, model.cfg.pretrained_asr).eval()
        with open_dict(model.cfg):
            model.cfg.perception.preprocessor = asr.cfg.preprocessor
            model.cfg.perception.encoder = asr.cfg.encoder
            model.cfg.perception.output_dim = model.llm.config.hidden_size
            # Override with user-specified encoder parameters (e.g. causal context size).
            for key, value in user_encoder_config.items():
                if value is not None:
                    model.cfg.perception.encoder[key] = value
        model.perception = AudioPerceptionModule(model.cfg.perception).train()
        model.perception.load_state_dict(asr.state_dict(), strict=False)
    else:
        with open_dict(model.cfg):
            model.cfg.perception.output_dim = model.llm.config.hidden_size
        model.perception = AudioPerceptionModule(model.cfg.perception).train()

def load_checkpoint(checkpoint_path: str) -> Dict:
    """
    Load a model checkpoint from disk.

    Supports PyTorch (``.ckpt``, ``.pt``), NeMo (``.nemo`` tar archive),
    and SafeTensors (``.safetensors``) formats. All tensors are loaded onto CPU.

    Args:
        checkpoint_path: Path to the checkpoint file.

    Returns:
        State dictionary mapping parameter names to tensors.
    """
    if ".safetensors" in checkpoint_path:
        return load_file(checkpoint_path, device="cpu")
    elif checkpoint_path.endswith(".nemo"):
        # NeMo archives are tar files containing model_weights.ckpt
        import io
        import tarfile
        with tarfile.open(checkpoint_path, "r:*") as tar:
            weights_member = next(m for m in tar.getmembers() if m.name.endswith("model_weights.ckpt"))
            buf = tar.extractfile(weights_member).read()
        ckpt = torch.load(io.BytesIO(buf), map_location="cpu", weights_only=False)
        # NeMo model_weights.ckpt is a flat state dict (no "state_dict" wrapper)
        return ckpt.get("state_dict", ckpt)
    else:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]


def _load_checkpoint_state(checkpoint_path: str) -> Dict:
    """Load checkpoint state dict from a file or HF-style directory (model.safetensors)."""
    if os.path.isdir(checkpoint_path):
        return load_file(os.path.join(checkpoint_path, "model.safetensors"))
    else:
        return torch.load(checkpoint_path, weights_only=False, map_location='cpu')['state_dict']


def init_perception_from_checkpoint(model: torch.nn.Module, checkpoint_path: str):
    """Load perception module weights from another S2S/STT checkpoint."""
    if checkpoint_path is None:
        return
    logging.info(f"Loading perception from checkpoint: {checkpoint_path}")
    checkpoint_state = _load_checkpoint_state(checkpoint_path)
    checkpoint_state = {k.replace("perception.", ""): v for k, v in checkpoint_state.items() if "perception." in k}
    checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, model.perception.state_dict())
    model.perception.load_state_dict(checkpoint_state, strict=True)


def init_model_from_checkpoint(model: torch.nn.Module, checkpoint_path: str):
    """Load full model weights from a checkpoint."""
    if checkpoint_path is None:
        return
    logging.info(f"Loading model from checkpoint: {checkpoint_path}")
    checkpoint_state = _load_checkpoint_state(checkpoint_path)
    checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, model.state_dict())
    model.load_state_dict(checkpoint_state, strict=True)


def load_pretrained_model(model: torch.nn.Module, checkpoint_path: str):
    """
    Load pretrained S2S model weights from a checkpoint path.

    Supports incremental loading from a SafeTensors directory (avoids OOM for large models)
    when ``model.cfg.incremental_loading`` is True.
    """
    if checkpoint_path is None:
        return
    logging.info(f"Loading pretrained S2S model from {checkpoint_path}")

    if os.path.isdir(checkpoint_path) and model.cfg.get("incremental_loading", False):
        from safetensors import safe_open

        model_state_dict = model.state_dict()
        loaded_keys, missing_keys = [], []
        with safe_open(os.path.join(checkpoint_path, "model.safetensors"), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in model_state_dict:
                    model_state_dict[key].copy_(f.get_tensor(key))
                    loaded_keys.append(key)
                else:
                    missing_keys.append(key)
                if len(loaded_keys) % 100 == 0:
                    gc.collect()
        logging.info(f"Loaded {len(loaded_keys)} tensors from pretrained model")
        if missing_keys:
            logging.warning(f"Keys in checkpoint not in model: {len(missing_keys)}")
        del model_state_dict
        gc.collect()
    else:
        init_model_from_checkpoint(model, checkpoint_path)


def maybe_load_pretrained_models(model: torch.nn.Module):
    """
    Optionally load pretrained weights based on config.

    Checks for:
      - ``pretrained_perception_from_s2s``: perception module from another S2S checkpoint.
      - ``pretrained_s2s_model``: full model weights (supports incremental loading).
    """
    if model.cfg.get("pretrained_perception_from_s2s", None):
        init_perception_from_checkpoint(model, model.cfg.pretrained_perception_from_s2s)
    if model.cfg.get("pretrained_s2s_model", None):
        load_pretrained_model(model, model.cfg.pretrained_s2s_model)


def set_model_dict_for_partial_init(pretrained_dict, model_dict):
    # 1. filter out different size layers
    for k, v in list(pretrained_dict.items()):
        if k in model_dict and hasattr(model_dict[k], "numel") and v.numel() != model_dict[k].numel():
            del pretrained_dict[k]
            logging.info(" | > Layer with shape mismatach in the model definition: {}".format(k)) 
    # 2. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 3. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    logging.info(" | > {} / {} layers are restored.".format(len(pretrained_dict), len(model_dict)))
    return model_dict
