# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""
VoiceTranslate-v1  —  End-to-end speech-to-speech translation model.

Pipeline
--------
  source speech (FR / ES / DE, 16 kHz)
      └─► AudioPerceptionModule (FastConformer, att_context=[70,0])
              └─► Riva-Translate-4B-Instruct (MistralModel base, 34 L, h=3072)
                      └─► RVQEARTTSModel  (TTS backbone, gemma3_text 28 L, h=1152)
                              └─► RVQVAEModel  (codec, 31 quantizers, 22 050 Hz)
                                      └─► English speech

Components
----------
  self.perception   : AudioPerceptionModule  (FastConformer from pretrained_asr)
  self.embed_tokens : nn.Embedding           (Riva-4B token embeddings, frozen)
  self.llm          : MistralModel           (Riva-4B transformer without embed_tokens)
  self.tts_model    : RVQEARTTSModel         (EarTTS TTS backbone, trainable)
  self.audio_codec  : RVQVAEModel            (EarTTS codec, frozen)

Branch : voicetranslate-v1/eartts_inter
Repo   : github.com/hdubey/private-nv-NeMo-s2s (private)
Created: 2026-04-16
"""

import torch
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf

from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.modules.ear_tts_model import RVQEARTTSModel
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers
from nemo.collections.speechlm2.parts.pretrained import (
    load_pretrained_hf,
    load_pretrained_nemo,
    setup_speech_encoder,
)
from nemo.utils import logging

try:
    from nemo.collections.speechlm2.modules.ear_tts_vae_codec import RVQVAEModel
except ImportError:
    from nemo.collections.speechlm2.models.duplex_ear_tts import _build_rvqvae_model as RVQVAEModel


class VoiceTranslateV1(LightningModule):
    """
    End-to-end VoiceTranslate-v1 model.

    Wires together:
      Perception (FastConformer) + LLM (Riva-4B) + TTS (RVQEARTTSModel) + Codec (RVQVAEModel)

    Config keys (model.*):
      pretrained_lm_name       : HF path/name for Riva-Translate-4B-Instruct
      pretrained_asr           : .nemo path for multilingual FastConformer
      pretrained_codec_model   : .nemo/.ckpt path for RVQVAEModel  (null → random weights)
      pretrained_tts_model     : .nemo/.ckpt path for RVQEARTTSModel (null → random weights)
      pretrained_weights       : bool, whether to load LLM pretrained weights (default False for random)
      codec_config             : RVQVAEModel architecture config
      tts_config               : RVQEARTTSModel architecture config (incl. context_hidden_size)
      freeze_params            : list of regexp patterns for frozen params
      perception               : AudioPerceptionModule config (encoder, modality_adapter, etc.)
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()

        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        OmegaConf.set_struct(cfg, False)

        # Store the full config (trainer + model + exp_manager) but expose model sub-config.
        self._full_cfg = cfg
        self.cfg = cfg.model if hasattr(cfg, "model") else cfg

        # ------------------------------------------------------------------
        # 1. Tokenizer  (Riva-4B tokenizer, vocab=131072)
        # ------------------------------------------------------------------
        logging.info("[VoiceTranslateV1] Loading tokenizer from %s", self.cfg.pretrained_lm_name)
        self.tokenizer = AutoTokenizer(self.cfg.pretrained_lm_name, use_fast=True)

        # ------------------------------------------------------------------
        # 2. LLM  (Riva-Translate-4B-Instruct)
        #    self.llm          = MistralModel (34 L, h=3072)   [trainable or frozen]
        #    self.embed_tokens = nn.Embedding (131072, 3072)   [frozen]
        # ------------------------------------------------------------------
        logging.info("[VoiceTranslateV1] Loading LLM: %s (pretrained_weights=%s)",
                     self.cfg.pretrained_lm_name,
                     self.cfg.get("pretrained_weights", False))
        llm = load_pretrained_hf(
            self.cfg.pretrained_lm_name,
            pretrained_weights=self.cfg.get("pretrained_weights", False),
        )
        self.llm = llm.model           # MistralModel (no embed_tokens, no lm_head)
        self.embed_tokens = self.llm.embed_tokens
        del self.llm.embed_tokens      # avoid double-counting; stored in self.embed_tokens

        # ------------------------------------------------------------------
        # 3. Perception  (multilingual FastConformer, att_context=[70,0])
        #    Loaded from pretrained_asr .nemo checkpoint; weights are trainable.
        # ------------------------------------------------------------------
        logging.info("[VoiceTranslateV1] Setting up perception (FastConformer) ...")
        setup_speech_encoder(
            self,
            pretrained_weights=self.cfg.get("pretrained_asr") is not None,
        )

        # ------------------------------------------------------------------
        # 4. Codec  (RVQVAEModel — always frozen)
        # ------------------------------------------------------------------
        logging.info("[VoiceTranslateV1] Setting up RVQVAEModel (codec) ...")
        from nemo.collections.speechlm2.models.duplex_ear_tts import DuplexEARTTS
        self.audio_codec = DuplexEARTTS._build_rvqvae(self.cfg)   # random or from ckpt
        for p in self.audio_codec.parameters():
            p.requires_grad = False

        # ------------------------------------------------------------------
        # 5. TTS model  (RVQEARTTSModel — trainable)
        #    context_hidden_size = LLM hidden size (3072 for Riva-4B)
        # ------------------------------------------------------------------
        logging.info("[VoiceTranslateV1] Setting up RVQEARTTSModel (TTS backbone) ...")
        tts_cfg = OmegaConf.to_container(self.cfg.tts_config, resolve=True)
        # Wire LLM hidden size as TTS context
        tts_cfg["context_hidden_size"] = self.llm.config.hidden_size
        self.tts_model = RVQEARTTSModel(tts_cfg, tokenizer=self.tokenizer)
        if self.cfg.get("pretrained_tts_model"):
            self._load_tts_checkpoint(self.cfg.pretrained_tts_model)

        # ------------------------------------------------------------------
        # 6. Freeze parameters per freeze_params patterns
        # ------------------------------------------------------------------
        self._apply_freeze_params()
        self._log_param_counts()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_tts_checkpoint(self, path: str):
        import torch
        state = torch.load(path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("tts_model.", ""): v for k, v in state.items() if k.startswith("tts_model.")}
        missing, unexpected = self.tts_model.load_state_dict(state, strict=False)
        logging.info("[VoiceTranslateV1] Loaded TTS checkpoint: %d missing, %d unexpected keys", len(missing), len(unexpected))

    def _apply_freeze_params(self):
        import re
        patterns = [re.compile(p) for p in self.cfg.get("freeze_params", [])]
        prevent = [re.compile(p) for p in self.cfg.get("prevent_freeze_params", [])]
        for name, param in self.named_parameters():
            if any(p.match(name) for p in patterns) and not any(p.match(name) for p in prevent):
                param.requires_grad = False

    def _log_param_counts(self):
        def _count(m):
            total = sum(p.numel() for p in m.parameters())
            train = sum(p.numel() for p in m.parameters() if p.requires_grad)
            return total, train

        rows = [
            ("perception",   self.perception),
            ("embed_tokens", self.embed_tokens),
            ("llm",          self.llm),
            ("tts_model",    self.tts_model),
            ("audio_codec",  self.audio_codec),
        ]
        logging.info("[VoiceTranslateV1] %-20s %14s  %14s", "Component", "Total params", "Trainable")
        for name, mod in rows:
            t, tr = _count(mod)
            logging.info("[VoiceTranslateV1] %-20s %14,d  %14,d", name, t, tr)
        t_all, tr_all = _count(self)
        logging.info("[VoiceTranslateV1] %-20s %14,d  %14,d  ← FULL MODEL", "TOTAL", t_all, tr_all)

    # ------------------------------------------------------------------
    # Forward  (training step — simplified skeleton)
    # ------------------------------------------------------------------

    def forward(self, batch):
        """
        Minimal forward for shape validation.  Full loss computation TBD.

        batch keys expected:
          source_audio      : [B, T_src]   float32, 16 kHz
          source_audio_lens : [B]          int
          text_inputs       : [B, T_txt]   int (Riva-4B token ids)
          target_audio      : [B, T_tgt]   float32, 22 050 Hz
          target_audio_lens : [B]          int
        """
        source_audio = batch["source_audio"]
        source_audio_lens = batch["source_audio_lens"]
        text_inputs = batch["text_inputs"]

        # 1. Perception: source speech → encoder hidden states
        source_encoded, source_encoded_lens, _ = self.perception(
            input_signal=source_audio,
            input_signal_length=source_audio_lens,
        )

        # 2. LLM: text token embeddings
        text_embeds = self.embed_tokens(text_inputs)  # [B, T_txt, 3072]

        # 3. LLM forward: concat perception output + text embeddings
        #    (simplified — actual implementation needs attention masking, etc.)
        llm_out = self.llm(
            inputs_embeds=text_embeds,
        )
        llm_hidden = llm_out.last_hidden_state  # [B, T_txt, 3072]

        # 4. Codec: encode target speech → audio codes
        target_audio = batch["target_audio"]
        target_audio_lens = batch["target_audio_lens"]
        target_codes, _ = self.audio_codec.encode(target_audio, target_audio_lens)  # [B, N_q, T_codes]

        # 5. TTS model: predict audio codes conditioned on LLM hidden states
        # (loss computation delegated to RVQEARTTSModel internals)
        return llm_hidden, target_codes

    # ------------------------------------------------------------------
    # Lightning boilerplate
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Full training_step TBD — forward skeleton only for now.")

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("Full validation_step TBD.")

    def configure_optimizers(self):
        return configure_optimizers(self)
