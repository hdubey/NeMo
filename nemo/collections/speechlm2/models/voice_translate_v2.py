"""
VoiceTranslateV2 — True End-to-End Speech-to-Speech Translation
===============================================================

Inherits from DuplexEARTTS and adds full LLM forward pass in the training loop.

Key difference from DuplexEARTTS (VoiceTranslate-v1):
------------------------------------------------------
  VT-v1 (DuplexEARTTS, context_hidden_size=null):
    - No LLM in forward pass at all
    - TTS conditioned only by CharAwareSubword text embeddings
    - Context = None (no LLM hidden states)

  VT-v2 (this class, context_hidden_size=3072):
    - Full Riva-4B LLM runs forward pass in training_step
    - TTS conditioned by LLM hidden states via embed_context (Linear 3072→1152)
    - Gradients flow: audio_loss → TTS → LLM (when LLM unfrozen in Stage 2)

Pipeline during training
------------------------
  batch["target_text_tokens"] [B, T_codes]   ← MFA-aligned, from DuplexEARTTSDataset
      → embed_tokens  [B, T, 3072]
      → language_model (MistralModel)  [B, T, 3072]  ← full LLM forward (not just embeddings)
      → context_hidden_state  [B, T-1, 3072]
      → tts_model.embed_context (Linear 3072→1152)  ← inside RVQEARTTSModel
      → TTS backbone (Gemma3-1B) + MoG head
      → audio loss (lm_loss + c_loss + k_loss)

Training plan (3 stages)
-------------------------
  Stage 1 — TTS warmup (~50K steps):
      freeze: language_model.*, embed_tokens.*, audio_codec.*
      train:  tts_model.* (learn to condition on LLM hidden states from random/frozen LLM)
      config: voice_translate_v2_sft.yaml with freeze_llm=true

  Stage 2 — End-to-end SFT (~150K steps):
      freeze: audio_codec.*  only
      train:  language_model.* + tts_model.*
      config: same yaml, remove language_model freeze pattern

  Stage 3 — GRPO (~2K steps, after Stage 2 checkpoint):
      Process rewards: BLEU at every 8 source words (α=0.4, Hibiki-Zero style)
      config: grpo_voicetranslate_v2.yaml (TBD)

Precision (unchanged from original components)
----------------------------------------------
  LLM (Riva-4B):         fp32  (load_pretrained_hf default)
  TTS (RVQEARTTSModel):  bf16  (PTL bf16-true trainer)
  Codec (RVQVAEModel):   fp32  (fp32_precision() context in setup_audio_codec)
  Perception:            bf16  (PTL bf16-true, optional Stage 2+)
"""

import copy
import re
from typing import Optional

import torch
import torch.nn.functional as F
from lightning.pytorch import Callback

from nemo.collections.speechlm2.models.duplex_ear_tts import DuplexEARTTS
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.utils import logging


class VoiceTranslateV2(DuplexEARTTS):
    """
    End-to-end VoiceTranslate model.

    DuplexEARTTS handles all data loading, codec setup, TTS training, and
    audio_prompt voice-cloning.  VoiceTranslateV2 adds full LLM forward pass.

    Set ``model.tts_config.context_hidden_size: 3072`` in config to activate
    the LLM→TTS hidden-state bridge (required for this class).

    Config keys added on top of DuplexEARTTS:
        model.freeze_llm  : bool (default true) — freeze LLM in Stage 1
    """

    def __init__(self, cfg: dict) -> None:
        # DuplexEARTTS.__init__ calls self._load_language_model() (Python MRO →
        # our override below), extracts embed_tokens, then does `del self.language_model`.
        # We intercept via _cached_llm so the full model is available here without
        # a second Lustre read (which would waste 15–30 min for Riva-4B fp32).
        self._cached_llm = None
        super().__init__(cfg)   # → calls our _load_language_model, stores in _cached_llm

        if self.cfg.tts_config.context_hidden_size is None:
            raise ValueError(
                "VoiceTranslateV2 requires tts_config.context_hidden_size to be set "
                "(e.g. 3072 for Riva-4B).  Set context_hidden_size=null only for "
                "standalone EarTTS (DuplexEARTTS)."
            )

        # Re-use the cached LLM — no second disk read.
        llm_full = self._cached_llm
        self._cached_llm = None
        self.language_model = llm_full.model   # MistralModel backbone
        self.lm_head        = llm_full.lm_head # Linear → vocab logits
        del llm_full

        self._apply_freeze_params_v2()
        self._log_param_counts_v2()

    def _load_language_model(self, cfg):
        """
        Override DuplexEARTTS._load_language_model to:
          1. Respect cfg.pretrained_weights (False → random init, no 16 GB disk read).
          2. Cache the loaded model in self._cached_llm so VoiceTranslateV2.__init__
             can reuse it after DuplexEARTTS.__init__ deletes self.language_model.
        """
        from nemo.collections.speechlm2.parts.pretrained import load_pretrained_hf
        pretrained = cfg.get("pretrained_weights", True)
        logging.info("[VT-V2] Loading LLM from %s (pretrained_weights=%s)",
                     cfg.pretrained_lm_name, pretrained)
        llm = load_pretrained_hf(
            cfg.pretrained_lm_name,
            pretrained_weights=pretrained,
            trust_remote_code=True,
        ).eval()
        self._cached_llm = llm   # intercept before DuplexEARTTS deletes self.language_model
        return llm

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_freeze_params_v2(self):
        """Apply freeze_params patterns to language_model and lm_head too."""
        import re
        patterns = [re.compile(p) for p in self.cfg.get("freeze_params", [])]
        prevent = [re.compile(p) for p in self.cfg.get("prevent_freeze_params", [])]
        for name, param in self.named_parameters():
            frozen = any(p.match(name) for p in patterns)
            prevented = any(p.match(name) for p in prevent)
            if frozen and not prevented:
                param.requires_grad = False

    def _log_param_counts_v2(self):
        rows = [
            ("embed_tokens",    self.embed_tokens),
            ("language_model",  self.language_model),
            ("lm_head",         self.lm_head),
            ("tts_model",       self.tts_model),
            ("audio_codec",     self.audio_codec),
        ]
        logging.info("[VT-V2] %-20s %14s  %14s", "Component", "Total", "Trainable")
        for name, mod in rows:
            t  = sum(p.numel() for p in mod.parameters())
            tr = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            logging.info("[VT-V2] %-20s %14d  %14d", name, t, tr)
        t_all  = sum(p.numel() for p in self.parameters())
        tr_all = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info("[VT-V2] %-20s %14d  %14d  (FULL MODEL)", "TOTAL", t_all, tr_all)

    # ------------------------------------------------------------------
    # Full LLM context computation  (replaces embed_tokens-only path)
    # ------------------------------------------------------------------

    def _compute_llm_context(self, target_text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Run the full Riva-4B forward pass and return hidden states.

        In DuplexEARTTS the context is: embed_tokens(tokens).detach()  — just embeddings.
        Here we run the full LLM: embed → MistralModel → hidden_states.
        When language_model is frozen, gradients don't flow through it (Stage 1).
        When unfrozen (Stage 2), gradients flow end-to-end.

        Args:
            target_text_tokens: [B, T] MFA-aligned text token IDs (Riva-4B vocab)

        Returns:
            context: [B, T, 3072] LLM hidden states
        """
        text_emb = self.embed_tokens(target_text_tokens)        # [B, T, 3072]
        llm_out  = self.language_model(inputs_embeds=text_emb)  # MistralModel
        return llm_out.last_hidden_state                         # [B, T, 3072]

    # ------------------------------------------------------------------
    # Training step  (overrides DuplexEARTTS.training_step)
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        import contextlib
        # Freeze TTS in eval mode if frozen (same as DuplexEARTTS)
        from nemo.collections.speechlm2.parts.optim_setup import is_frozen
        for m in (self.tts_model,):
            if is_frozen(m):
                m.eval()

        # Step 1: shared input preparation (codec encode, audio prompt, masks).
        # prepare_inputs processes target_text_tokens (dropout/masking) and computes
        # subword_ids = F.pad(processed_tokens[:, 1:], [0, 1]).  Both have shape [B, T_total].
        inputs = self.prepare_inputs(batch)

        # Step 2: Replace embed_tokens context with full LLM hidden states.
        # CRITICAL: use inputs["target_text_tokens"] (the PROCESSED version, same sequence
        # prepare_inputs used to build subword_ids), NOT the raw batch tokens.
        # DO NOT slice [:,:-1,:] — DuplexEARTTS passes the full T_total context.
        # context[i] conditions on the token at position i; subword_ids[i] = token[i+1] (shifted).
        processed_tokens = inputs["target_text_tokens"]         # [B, T_total]
        llm_frozen = not any(p.requires_grad for p in self.language_model.parameters())
        ctx = torch.no_grad() if llm_frozen else contextlib.nullcontext()
        with ctx:
            context = self._compute_llm_context(processed_tokens)  # [B, T_total, 3072]
        inputs["context_hidden_state"] = context

        # Step 3: Optional text CE loss (inner-monologue, Hibiki-Zero style)
        text_loss = torch.tensor(0.0, device=self.device)
        text_loss_weight = float(self.cfg.get("text_loss_weight", 0.0))
        if text_loss_weight > 0:
            # lm_head on hidden states [B, T-1, 3072] → predict tokens [B, T-1]
            logits  = self.lm_head(context[:, :-1, :])          # [B, T-1, vocab]
            targets = target_text_tokens[:, 1:]                  # [B, T-1]
            text_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.text_pad_id,
            )

        # Step 4: TTS forward (identical to DuplexEARTTS.training_step)
        tts_output = self.tts_model(
            code=inputs["code"],
            audio_mask=inputs["audio_mask"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            context_hidden_state=inputs["context_hidden_state"],
            subword_ids=inputs["subword_ids"],
            subword_mask=inputs["subword_mask"],
            non_prompt_mask=inputs["non_prompt_mask"],
            dataset_type=batch.get("dataset_type", None),
            tiled_prompt_audio_codes=inputs["tiled_prompt_audio_codes"],
            tiled_prompt_subword_ids=inputs["tiled_prompt_subword_ids"],
            tiled_prompt_subword_mask=inputs["tiled_prompt_subword_mask"],
        )

        # Step 5: Combined loss
        audio_loss_weight = float(getattr(self.cfg, "audio_loss_weight", 1.0))
        audio_loss = tts_output.lm_loss + tts_output.c_loss + tts_output.k_loss
        total_loss = audio_loss_weight * audio_loss + text_loss_weight * text_loss

        num_frames = inputs["output_lens"].sum()
        B, T = inputs["code"].shape[:2]
        log_dict = {
            "loss":          total_loss,
            "audio_loss":    audio_loss,
            "text_loss":     text_loss,
            "lm_loss":       tts_output.lm_loss,
            "c_loss":        tts_output.c_loss,
            "k_loss":        tts_output.k_loss,
            "learning_rate": torch.as_tensor(
                self.trainer.optimizers[0].param_groups[0]['lr']
                if self._trainer is not None else 0.0
            ),
            "batch_size":     B,
            "sequence_length": T,
            "num_frames":     num_frames.to(torch.float32),
            "padding_ratio":  num_frames / (B * T),
        }
        self.log_dict(log_dict, on_step=True)
        step = self.trainer.global_step if self._trainer is not None else batch_idx
        logging.info(
            "[VT-V2 STEP %d | rank %s] BACKWARD_LOSS=%.6f  "
            "audio=%.6f (lm=%.6f c=%.6f k=%.6f)  text=%.6f  lr=%.2e",
            step,
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            float(total_loss),
            float(audio_loss),
            float(tts_output.lm_loss),
            float(tts_output.c_loss),
            float(tts_output.k_loss),
            float(text_loss),
            float(log_dict["learning_rate"]),
        )
        return log_dict


# ──────────────────────────────────────────────────────────────────────────────
# GRPO helpers (inlined from grpo_voicetranslate_v1.py to stay self-contained)
# ──────────────────────────────────────────────────────────────────────────────

class _ASRBLEUReward:
    """ASR-BLEU reward: codec tokens → audio → ASR → BLEU vs EN reference."""

    def __init__(self, asr_model_name: str, device: str = "cuda", utmos_weight: float = 0.0):
        import nemo.collections.asr as nemo_asr
        import sacrebleu as sb
        self._sb = sb
        self.utmos_weight = utmos_weight
        logging.info("[GRPO] Loading ASR reward model: %s", asr_model_name)
        self.asr = nemo_asr.models.EncDecRNNTModel.from_pretrained(asr_model_name)
        self.asr.to(device).eval()
        for p in self.asr.parameters():
            p.requires_grad_(False)
        self.device = device
        self.utmos = None
        if utmos_weight > 0.0:
            try:
                import utmos
                self.utmos = utmos.UTMOSScore(device=device)
            except ImportError:
                logging.warning("[GRPO] utmos not installed; skipping UTMOS reward.")

    @torch.no_grad()
    def __call__(self, audio_list: list, ref_texts: list, sample_rate: int = 22050) -> list:
        audio_np = [a.float().cpu().numpy() for a in audio_list]
        hyps = self.asr.transcribe(audio_np, batch_size=len(audio_np))
        if isinstance(hyps, (list, tuple)) and isinstance(hyps[0], (list, tuple)):
            hyps = hyps[0]
        rewards = []
        for i, (hyp, ref) in enumerate(zip(hyps, ref_texts)):
            bleu = self._sb.corpus_bleu(
                [str(hyp)], [[str(ref)]], tokenize="13a", smooth_method="exp"
            ).score / 100.0
            if self.utmos is not None and self.utmos_weight > 0.0:
                mos = (self.utmos.score(audio_np[i], sample_rate) - 1.0) / 4.0
                bleu = (1 - self.utmos_weight) * bleu + self.utmos_weight * mos
            rewards.append(bleu)
        return rewards


def _grpo_policy_loss(
    nll_policy: torch.Tensor,
    nll_ref: torch.Tensor,
    rewards: torch.Tensor,
    kl_coeff: float = 0.04,
    eps_clip: float = 0.2,
    eps_norm: float = 1e-8,
) -> tuple:
    """GRPO loss: advantage-weighted NLL + KL regularisation."""
    rewards = rewards.detach()
    nll_ref = nll_ref.detach()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        gathered = [torch.zeros_like(rewards) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, rewards)
        global_r = torch.cat(gathered)
        mean_r, std_r = global_r.mean(), global_r.std().clamp_min(eps_norm)
    else:
        mean_r, std_r = rewards.mean(), rewards.std().clamp_min(eps_norm)

    advantage = (rewards - mean_r) / std_r
    log_ratio  = -(nll_policy - nll_ref.detach())
    ratio      = log_ratio.exp()
    pg_loss    = torch.max(-advantage * ratio,
                           -advantage * ratio.clamp(1 - eps_clip, 1 + eps_clip)).mean()
    kl_loss    = (nll_ref.detach() - nll_policy).mean()
    total      = pg_loss + kl_coeff * kl_loss

    stats = {
        "grpo/pg_loss": pg_loss.item(), "grpo/kl_loss": kl_loss.item(),
        "grpo/total_loss": total.item(), "grpo/mean_reward": mean_r.item(),
        "grpo/std_reward": std_r.item(), "grpo/mean_ratio": ratio.mean().item(),
    }
    return total, stats


# ──────────────────────────────────────────────────────────────────────────────
# Stage-transition callback
# ──────────────────────────────────────────────────────────────────────────────

class StageTransitionCallback(Callback):
    """
    Fires model stage transitions at configured global-step boundaries.

    Attach to the PTL Trainer when using VoiceTranslateV2Stages:
        callback = StageTransitionCallback(
            stage1_steps=cfg.training_stages.stage1_steps,
            stage2_steps=cfg.training_stages.stage2_steps,
        )
        trainer = Trainer(..., callbacks=[callback])
    """

    def __init__(self, stage1_steps: int, stage2_steps: int):
        super().__init__()
        self.s1_end = stage1_steps
        self.s2_end = stage1_steps + stage2_steps

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        step = trainer.global_step
        if step == self.s1_end and pl_module.current_stage == 1:
            logging.info("[StageTransition] Step %d → Stage 2 (E2E SFT, LLM unfrozen)", step)
            pl_module.transition_to_stage(2)
        elif step == self.s2_end and pl_module.current_stage == 2:
            logging.info("[StageTransition] Step %d → Stage 3 (GRPO)", step)
            pl_module.transition_to_stage(3)


# ──────────────────────────────────────────────────────────────────────────────
# VoiceTranslateV2Stages — all 3 stages in one PTL fit() call
# ──────────────────────────────────────────────────────────────────────────────

class VoiceTranslateV2Stages(VoiceTranslateV2):
    """
    VoiceTranslateV2 with all 3 training stages in a single Trainer.fit() call.

    Stage 1  (0 … stage1_steps):                TTS warmup  — LLM frozen
    Stage 2  (stage1_steps … s1+s2):            E2E SFT     — LLM unfrozen, text-loss=100
    Stage 3  (s1+s2 … s1+s2+s3):               GRPO        — TTS policy, ASR-BLEU reward

    Attach StageTransitionCallback so transitions fire at the right steps.

    Config keys (under training_stages):
        stage1_steps:              int   (default 50 000)
        stage2_steps:              int   (default 150 000)
        stage3_steps:              int   (default 2 000)
        stage2_text_loss_weight:   float (default 100.0)

    Config keys (under model.grpo):
        num_generations, kl_coeff, eps_clip, reward_clip, gen_max_steps, utmos_weight
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.current_stage: int = 1

        # GRPO internals (lazy-init at Stage 3)
        self._ref_tts: Optional[torch.nn.Module] = None
        self._reward_fn: Optional[_ASRBLEUReward] = None

        gcfg = self.cfg.get("grpo", {})
        self.G              = int(gcfg.get("num_generations", 8))
        self.kl_coeff       = float(gcfg.get("kl_coeff", 0.04))
        self.eps_clip       = float(gcfg.get("eps_clip", 0.2))
        self.gen_max_steps  = int(gcfg.get("gen_max_steps", 500))
        self.reward_clip    = gcfg.get("reward_clip", None)
        self.utmos_weight   = float(gcfg.get("utmos_weight", 0.0))
        self._scoring_asr   = self.cfg.get(
            "scoring_asr", "stt_en_fastconformer_transducer_large"
        )

    # ── optimizer: ALL params registered so unfreeze works without rebuild ──

    def configure_optimizers(self):
        """
        Register ALL parameters (frozen + unfrozen) with the optimizer.
        Frozen params have grad=None → no update.  Unfreezing them at stage
        transitions starts updates immediately without rebuilding the optimizer.
        """
        import hydra

        no_wd_keys = {"bias", "norm", "layernorm"}
        wd_params, no_wd_params = [], []
        for name, param in self.named_parameters():
            if any(k in name.lower() for k in no_wd_keys):
                no_wd_params.append(param)
            else:
                wd_params.append(param)

        opt_cfg = {k: v for k, v in self.cfg["optimizer"].items() if k != "_target_"}
        wd_val  = opt_cfg.pop("weight_decay", 0.0)
        opt_cls = hydra.utils.get_class(self.cfg["optimizer"]["_target_"])
        optimizer = opt_cls(
            [{"params": wd_params, "weight_decay": wd_val},
             {"params": no_wd_params, "weight_decay": 0.0}],
            **opt_cfg,
        )

        sched_cfg = dict(self.cfg.get("lr_scheduler", {}))
        if not sched_cfg:
            return {"optimizer": optimizer}
        sched_cls = hydra.utils.get_class(sched_cfg.pop("_target_"))
        scheduler = sched_cls(optimizer, **sched_cfg)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    # ── stage transitions ───────────────────────────────────────────────────

    def transition_to_stage(self, stage: int) -> None:
        stages_cfg = self.cfg.get("training_stages", {})
        if stage == 2:
            # Unfreeze LLM + embedding table for E2E SFT
            for name, param in self.named_parameters():
                if re.match(r"^(language_model|lm_head|embed_tokens)\..+$", name):
                    param.requires_grad_(True)
            self.cfg["text_loss_weight"] = float(
                stages_cfg.get("stage2_text_loss_weight", 100.0)
            )
        elif stage == 3:
            # Re-freeze LLM; TTS is the GRPO policy
            for name, param in self.named_parameters():
                if re.match(r"^(language_model|lm_head|embed_tokens|audio_codec)\..+$", name):
                    param.requires_grad_(False)
            self.cfg["text_loss_weight"] = 0.0
            self._init_grpo_components()
        self.current_stage = stage
        self._log_param_counts_v2()

    def _init_grpo_components(self):
        if self._ref_tts is None:
            logging.info("[VT-V2-Stages] Building frozen reference TTS for GRPO...")
            self._ref_tts = copy.deepcopy(self.tts_model)
            for p in self._ref_tts.parameters():
                p.requires_grad_(False)
            self._ref_tts.eval()
        if self._reward_fn is None:
            self._reward_fn = _ASRBLEUReward(
                self._scoring_asr,
                device=str(self.device),
                utmos_weight=self.utmos_weight,
            )

    # ── training dispatch ──────────────────────────────────────────────────

    def training_step(self, batch: dict, batch_idx: int):
        if self.current_stage in (1, 2):
            return super().training_step(batch, batch_idx)
        return self._grpo_step(batch, batch_idx)

    # ── GRPO generation helpers ────────────────────────────────────────────

    @torch.no_grad()
    def _generate_codes(self, inputs: dict) -> torch.Tensor:
        gen_cfg = self._get_generation_config(guidance_enabled=True)
        out = self.tts_model(
            code=inputs["code"],
            audio_mask=inputs["audio_mask"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            context_hidden_state=inputs["context_hidden_state"],
            subword_ids=inputs["subword_ids"],
            subword_mask=inputs["subword_mask"],
            non_prompt_mask=inputs["non_prompt_mask"],
            dataset_type=None,
            tiled_prompt_audio_codes=None,
            tiled_prompt_subword_ids=None,
            tiled_prompt_subword_mask=None,
            generation_config=gen_cfg,
        )
        return out.codes.squeeze(0)  # [T, NQ]

    @torch.no_grad()
    def _codes_to_audio(self, codes: torch.Tensor) -> torch.Tensor:
        lens = torch.tensor([codes.shape[0]], device=self.device)
        audio, _ = self.audio_codec.decode(codes.unsqueeze(0).long(), lens)
        return audio.squeeze(0).squeeze(0)  # [T_audio]

    def _compute_nll(self, inputs: dict, codes: torch.Tensor,
                     tts_module: torch.nn.Module) -> torch.Tensor:
        T = min(codes.shape[0], inputs["code"].shape[1])
        codes_tf = inputs["code"].clone()
        codes_tf[:, :T] = codes[:T].unsqueeze(0)
        out = tts_module(
            code=codes_tf,
            audio_mask=inputs["audio_mask"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            context_hidden_state=inputs["context_hidden_state"],
            subword_ids=inputs["subword_ids"],
            subword_mask=inputs["subword_mask"],
            non_prompt_mask=inputs["non_prompt_mask"],
            dataset_type=None,
            tiled_prompt_audio_codes=None,
            tiled_prompt_subword_ids=None,
            tiled_prompt_subword_mask=None,
            training=True,
        )
        return out.c_loss + out.k_loss

    # ── GRPO training step ─────────────────────────────────────────────────

    def _grpo_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        self._init_grpo_components()
        self.tts_model.train()
        self.audio_codec.eval()
        self._ref_tts.eval()

        inputs = self.prepare_inputs(batch)

        # LLM context: use processed tokens (same as training_step); frozen in Stage 3
        with torch.no_grad():
            context = self._compute_llm_context(inputs["target_text_tokens"])  # [B, T_total, 3072]
        inputs["context_hidden_state"] = context  # [B, T_total, 3072] — no slice

        B = inputs["code"].shape[0]
        ref_texts = batch.get("target_texts", batch.get("ref_en_texts", None))
        if ref_texts is None:
            logging.warning("[GRPO] No reference texts in batch — reward will be zero.")
            ref_texts = [""] * B

        all_losses, all_stats = [], []
        for b in range(B):
            single = {k: (v[b:b+1] if isinstance(v, torch.Tensor) else v)
                      for k, v in inputs.items()}

            # Phase 1: generate G samples (no_grad)
            gen_codes = [self._generate_codes(single) for _ in range(self.G)]

            # Phase 2: ASR-BLEU rewards
            audios   = [self._codes_to_audio(c) for c in gen_codes]
            rewards  = self._reward_fn(
                audios, [ref_texts[b]] * self.G,
                sample_rate=self.target_sample_rate,
            )

            # Phase 3: NLL under policy (grad) and reference (no_grad)
            nll_p = torch.stack([self._compute_nll(single, c, self.tts_model) for c in gen_codes])
            with torch.no_grad():
                nll_r = torch.stack([self._compute_nll(single, c, self._ref_tts) for c in gen_codes])

            r_t = torch.tensor(rewards, device=self.device, dtype=torch.float32)
            if self.reward_clip is not None:
                r_t = r_t.clamp(-self.reward_clip, self.reward_clip)

            loss, stats = _grpo_policy_loss(nll_p, nll_r, r_t, self.kl_coeff, self.eps_clip)
            all_losses.append(loss)
            all_stats.append(stats)

        total_loss = torch.stack(all_losses).mean()
        log_dict = {"loss": total_loss}
        for key in all_stats[0]:
            log_dict[key] = sum(s[key] for s in all_stats) / len(all_stats)
        log_dict["learning_rate"] = (
            self.trainer.optimizers[0].param_groups[0]["lr"]
            if self._trainer is not None else 0.0
        )
        self.log_dict(log_dict, on_step=True, on_epoch=False)
        return total_loss
