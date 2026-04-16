"""
GRPO (Group Relative Policy Optimization) training for VoiceTranslate-v1.

Stage 2 after SFT: removes need for aligned data via ASR-BLEU reward.

Algorithm (Hibiki-Zero style):
    For each input batch:
        1. Generate G speech samples using EarTTS (no_grad, MoG sampling)
        2. Decode each sample to audio, compute ASR-BLEU reward
        3. Group-normalize rewards → advantages
        4. Teacher-force each generated sample through policy → get MoG NLL
        5. GRPO loss = sum_g( advantage_g * nll_g ) + kl_coeff * KL(policy || ref)
        6. Backprop through NLL only (advantages are detached)

Key: gradients flow through step 4 (NLL recomputation), not through sampling.
     This is REINFORCE with group-relative baseline.

References:
    Hibiki-Zero   (Kyutai, 2025)    https://arxiv.org/abs/2502.03382
    GRPO          (DeepSeek, 2024)  https://arxiv.org/abs/2402.03300
    Voicebox      (Meta, 2023)      https://arxiv.org/abs/2306.15687

Run:
    python scripts/grpo_voicetranslate_v1.py \\
        --config-path conf --config-name grpo_voicetranslate_v1

Submit:
    sbatch s2s_exp/eartts/train_grpo_vt_v1_test200.sh
"""

import copy
import datetime
import os
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn.functional as F
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf

from nemo.collections.speechlm2 import DataModule
from nemo.collections.speechlm2.data import DuplexEARTTSDataset
from nemo.collections.speechlm2.models.duplex_ear_tts import DuplexEARTTS
from nemo.collections.speechlm2.parts.pretrained import load_checkpoint, set_model_dict_for_partial_init
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

# Note: cuda.set_device is called inside train() to avoid crashing on CPU-only import


# ─────────────────────────────────────────────────────────────────────────────
# Reward
# ─────────────────────────────────────────────────────────────────────────────

class ASRBLEUReward:
    """
    ASR-BLEU reward: decode codec tokens → audio → ASR transcript → BLEU vs ref EN text.

    References:
        Hibiki-Zero §3.2: reward = BLEU(ASR(ŷ), y_ref)
        VALL-E 2:         add UTMOS for speech quality (optional)
    """

    def __init__(self, asr_model_name: str, device: str = "cuda",
                 utmos_weight: float = 0.0):
        import nemo.collections.asr as nemo_asr
        import sacrebleu as sb
        self._sb = sb
        self.utmos_weight = utmos_weight

        logging.info(f"[GRPO] Loading ASR reward model: {asr_model_name}")
        self.asr = nemo_asr.models.EncDecRNNTModel.from_pretrained(asr_model_name)
        self.asr.to(device).eval()
        for p in self.asr.parameters():
            p.requires_grad_(False)

        # Optional: UTMOS for speech quality reward (VALL-E 2 advice)
        self.utmos = None
        if utmos_weight > 0.0:
            try:
                import utmos
                self.utmos = utmos.UTMOSScore(device=device)
                logging.info("[GRPO] UTMOS reward loaded.")
            except ImportError:
                logging.warning("[GRPO] utmos not installed, skipping UTMOS reward.")

        self.device = device
        logging.info("[GRPO] ASR reward model ready.")

    @torch.no_grad()
    def __call__(
        self,
        audio_list: list,           # list of 1-D float32 waveform tensors
        ref_texts: list[str],       # EN reference text for each sample
        sample_rate: int = 22050,
    ) -> list[float]:
        """Return rewards in [0, 1] for each (audio, ref_text) pair."""
        audio_np = [a.float().cpu().numpy() for a in audio_list]
        hyps = self.asr.transcribe(audio_np, batch_size=len(audio_np))
        # NeMo RNNT returns (hypotheses, ...) — handle both formats
        if isinstance(hyps, (list, tuple)) and isinstance(hyps[0], (list, tuple)):
            hyps = hyps[0]

        rewards = []
        for hyp, ref in zip(hyps, ref_texts):
            # BLEU score, normalized to [0, 1]
            bleu = self._sb.corpus_bleu(
                [str(hyp)], [[str(ref)]],
                tokenize="13a",
                smooth_method="exp",
            ).score / 100.0  # → [0, 1]

            if self.utmos is not None and self.utmos_weight > 0.0:
                # MOS score typically [1, 5], normalize to [0, 1]
                mos = (self.utmos.score(audio_np[len(rewards)], sample_rate) - 1.0) / 4.0
                bleu = (1 - self.utmos_weight) * bleu + self.utmos_weight * mos

            rewards.append(bleu)
        return rewards


# ─────────────────────────────────────────────────────────────────────────────
# GRPO loss
# ─────────────────────────────────────────────────────────────────────────────

def grpo_policy_loss(
    nll_policy: torch.Tensor,       # [G] — MoG NLL under policy (higher = less likely)
    nll_ref:    torch.Tensor,       # [G] — MoG NLL under frozen reference (no grad)
    rewards:    torch.Tensor,       # [G] — scalar rewards per generation (no grad)
    kl_coeff:   float = 0.04,
    eps_clip:   float = 0.2,        # PPO-style ratio clipping (DeepSeekMath advice)
    eps_norm:   float = 1e-8,
) -> tuple[torch.Tensor, dict]:
    """
    GRPO loss for EarTTS policy.

    policy_gradient = -advantage * log_p(codes)
                    = +advantage * NLL(codes)   (NLL = -log_p)

    Advantage is group-relative (Hibiki-Zero §3.2, DeepSeekMath §3.1):
        A_g = (r_g - mean(r)) / (std(r) + eps)

    PPO-style ratio clipping prevents large policy updates (DeepSeekMath advice):
        ratio = exp(log_p_policy - log_p_ref) = exp(nll_ref - nll_policy)
        clipped = clip(ratio, 1-ε, 1+ε)

    KL regularization keeps policy close to reference:
        KL ≈ nll_policy - nll_ref  (per-sample average KL)
    """
    G = rewards.shape[0]
    rewards = rewards.detach()
    nll_ref  = nll_ref.detach()

    # Group-relative advantage: normalize over G samples
    # In DDP, each rank has its own G rewards. For correct group-relative normalization,
    # gather rewards from ALL ranks so mean/std are global, not per-rank local.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        gathered = [torch.zeros_like(rewards) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, rewards)
        global_rewards = torch.cat(gathered)   # [world_size * G]
        mean_r = global_rewards.mean()
        std_r  = global_rewards.std().clamp_min(eps_norm)
    else:
        mean_r = rewards.mean()
        std_r  = rewards.std().clamp_min(eps_norm)
    advantage = (rewards - mean_r) / std_r    # [G], detached — normalized globally

    # Log importance ratio: log π(codes) / π_ref(codes) = -(nll - nll_ref)
    log_ratio = -(nll_policy - nll_ref.detach())   # [G]; positive = policy assigns higher prob

    # PPO clipping (DeepSeekMath, eq. 5)
    ratio   = log_ratio.exp()
    pg_unclipped = -advantage * ratio
    pg_clipped   = -advantage * ratio.clamp(1 - eps_clip, 1 + eps_clip)
    pg_loss      = torch.max(pg_unclipped, pg_clipped).mean()

    # KL penalty: keeps policy from straying too far from reference
    # KL(policy || ref) = E_policy[log π_θ/π_ref] = E_policy[nll_ref - nll_θ]  (always ≥ 0)
    # Minimizing total_loss with positive KL keeps policy close to reference.
    # NOTE: (nll_policy - nll_ref) = -KL(policy||ref) — wrong sign, would push policy AWAY.
    kl_loss = (nll_ref.detach() - nll_policy).mean()   # = KL(policy || ref) per token

    total_loss = pg_loss + kl_coeff * kl_loss

    stats = {
        "grpo/pg_loss":     pg_loss.item(),
        "grpo/kl_loss":     kl_loss.item(),
        "grpo/total_loss":  total_loss.item(),
        "grpo/mean_reward": mean_r.item(),
        "grpo/std_reward":  std_r.item(),
        "grpo/mean_ratio":  ratio.mean().item(),
        "grpo/advantage_max": advantage.max().item(),
        "grpo/advantage_min": advantage.min().item(),
    }
    return total_loss, stats


# ─────────────────────────────────────────────────────────────────────────────
# GRPO model
# ─────────────────────────────────────────────────────────────────────────────

class VoiceTranslateGRPO(DuplexEARTTS):
    """
    DuplexEARTTS + GRPO training (Hibiki-Zero style).

    Extends DuplexEARTTS with a two-phase training_step:
        Phase 1 (no_grad): generate G speech samples, compute ASR-BLEU rewards
        Phase 2 (grad):    teacher-force generated codes through policy → MoG NLL
        GRPO update:       advantage-weighted NLL + KL vs frozen reference

    Yaml keys (under model.grpo):
        num_generations:  G per input (default 8)
        kl_coeff:         KL penalty weight (default 0.04)
        eps_clip:         PPO ratio clip (default 0.2)
        utmos_weight:     weight of UTMOS quality reward (default 0.0, BLEU only)
        gen_max_steps:    max generation steps (default 500)
        reward_clip:      max reward magnitude for stability (default None)
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # cfg may be the full config dict ({"model": {...}, "trainer": {...}, ...}) or just
        # the model sub-dict — handle both. grpo settings live under model.grpo.
        _model_cfg = cfg.get("model", cfg) if isinstance(cfg, dict) else cfg
        gcfg = _model_cfg.get("grpo", {})
        self.G             = gcfg.get("num_generations", 8)
        self.kl_coeff      = gcfg.get("kl_coeff", 0.04)
        self.eps_clip      = gcfg.get("eps_clip", 0.2)
        self.utmos_weight  = gcfg.get("utmos_weight", 0.0)
        self.gen_max_steps = gcfg.get("gen_max_steps", 500)
        self.reward_clip   = gcfg.get("reward_clip", None)

        self._scoring_asr  = _model_cfg.get("scoring_asr", "stt_en_fastconformer_transducer_large")
        self._ref_tts: Optional[torch.nn.Module] = None
        self._reward_fn: Optional[ASRBLEUReward] = None

    # ── lazy init ─────────────────────────────────────────────────────────────

    def _ensure_ref_and_reward(self):
        if self._ref_tts is None:
            logging.info("[GRPO] Building frozen reference TTS (copy of current policy)...")
            self._ref_tts = copy.deepcopy(self.tts_model)
            for p in self._ref_tts.parameters():
                p.requires_grad_(False)
            self._ref_tts.eval()
            logging.info("[GRPO] Reference TTS ready.")

        if self._reward_fn is None:
            logging.info(f"[GRPO] Loading ASR reward ({self._scoring_asr})...")
            self._reward_fn = ASRBLEUReward(
                self._scoring_asr,
                device=str(self.device),
                utmos_weight=self.utmos_weight,
            )

    # ── generation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _generate_codes(self, inputs: dict) -> torch.Tensor:
        """
        Generate one speech sample from current policy.
        Returns codes [T, NQ].
        """
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
        return out.codes.squeeze(0)   # [T, NQ]

    @torch.no_grad()
    def _codes_to_audio(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode [T, NQ] codec codes to 1-D float32 waveform."""
        codes_in = codes.unsqueeze(0).long()                    # [1, T, NQ]
        lens     = torch.tensor([codes.shape[0]], device=self.device)
        audio, _ = self.audio_codec.decode(codes_in, lens)      # [1, 1, T_audio]
        return audio.squeeze(0).squeeze(0)                       # [T_audio]

    # ── NLL recomputation (with grad) ─────────────────────────────────────────

    def _compute_nll(self, inputs: dict, generated_codes: torch.Tensor,
                     tts_module: torch.nn.Module) -> torch.Tensor:
        """
        Teacher-force `generated_codes` through `tts_module` and return the
        mean per-frame NLL (c_loss + k_loss) as a scalar.

        This is differentiable w.r.t. tts_module parameters —
        gradients flow through here for the policy update.
        """
        T_gen = generated_codes.shape[0]
        T_inp = inputs["code"].shape[1]
        T     = min(T_gen, T_inp)

        # Replace ground-truth codes with generated codes (teacher-force)
        codes_tf = inputs["code"].clone()
        codes_tf[:, :T] = generated_codes[:T].unsqueeze(0)

        # Force training=True so RVQEARTTSModel always runs the loss-computation path.
        # The reference model is in eval() to disable dropout, but we still need MoG NLL.
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
            training=True,   # force training path → c_loss/k_loss always computed
        )
        # c_loss + k_loss = MoG NLL (higher = less likely under this model)
        nll = out.c_loss + out.k_loss
        return nll

    # ── training step ─────────────────────────────────────────────────────────

    def training_step(self, batch: dict, batch_idx: int):
        self._ensure_ref_and_reward()
        self.tts_model.train()
        self.audio_codec.eval()       # codec always frozen
        self._ref_tts.eval()

        inputs = self.prepare_inputs(batch)
        B      = inputs["code"].shape[0]

        # ref_en_texts: target EN text for BLEU reward.
        # Dataset (DuplexEARTTSDataset) returns this as "target_texts" (output_roles supervisions).
        # "ref_en_texts" is kept as secondary key for compatibility; falls back to empty string
        # (zero reward, no learning signal) if neither key is present — warn loudly.
        ref_en_texts = batch.get("target_texts", batch.get("ref_en_texts", None))
        if ref_en_texts is None:
            logging.warning(
                "[GRPO] Neither 'target_texts' nor 'ref_en_texts' found in batch. "
                "All rewards will be zero (no learning signal). "
                "Check that output_roles supervisions are present in lhotse data."
            )
            ref_en_texts = [""] * B

        all_grpo_losses = []
        all_stats = []

        for b in range(B):
            # Single-example slice (batch dim = 1)
            single = {k: (v[b:b+1] if isinstance(v, torch.Tensor) else v)
                      for k, v in inputs.items()}

            rewards_b:   list[float]         = []
            nll_policy_b: list[torch.Tensor] = []
            nll_ref_b:    list[torch.Tensor] = []

            # ── Phase 1: generate G samples (no_grad) ─────────────────────────
            gen_codes_list: list[torch.Tensor] = []
            for _ in range(self.G):
                codes = self._generate_codes(single)   # [T, NQ], no grad
                gen_codes_list.append(codes)

            # ── Phase 2: compute ASR-BLEU rewards ─────────────────────────────
            audios = [self._codes_to_audio(c) for c in gen_codes_list]
            rewards_b = self._reward_fn(
                audios,
                [ref_en_texts[b]] * self.G,
                sample_rate=self.target_sample_rate,
            )

            # ── Phase 3: NLL recomputation under policy and reference ──────────
            for codes in gen_codes_list:
                # Policy NLL (WITH grad — gradients flow here)
                nll_p = self._compute_nll(single, codes, self.tts_model)
                nll_policy_b.append(nll_p)

                # Reference NLL (no grad)
                with torch.no_grad():
                    nll_r = self._compute_nll(single, codes, self._ref_tts)
                nll_ref_b.append(nll_r)

            # ── Phase 4: GRPO loss for this example ───────────────────────────
            nll_policy_t = torch.stack(nll_policy_b)   # [G]
            nll_ref_t    = torch.stack(nll_ref_b)       # [G]
            rewards_t    = torch.tensor(rewards_b, device=self.device, dtype=torch.float32)

            if self.reward_clip is not None:
                rewards_t = rewards_t.clamp(-self.reward_clip, self.reward_clip)

            grpo_loss, stats = grpo_policy_loss(
                nll_policy=nll_policy_t,
                nll_ref=nll_ref_t,
                rewards=rewards_t,
                kl_coeff=self.kl_coeff,
                eps_clip=self.eps_clip,
            )
            all_grpo_losses.append(grpo_loss)
            all_stats.append(stats)

        # ── Average over batch ────────────────────────────────────────────────
        loss = torch.stack(all_grpo_losses).mean()

        # Aggregate stats (mean over batch)
        log_dict: dict = {"loss": loss}
        for key in all_stats[0]:
            log_dict[key] = sum(s[key] for s in all_stats) / len(all_stats)
        log_dict["learning_rate"] = (
            self.trainer.optimizers[0].param_groups[0]["lr"]
            if self.trainer is not None else 0.0
        )

        self.log_dict(log_dict, on_step=True, on_epoch=False)
        return loss

    # ── validation inherits from DuplexEARTTS (ASR-BLEU, WER, SECS) ──────────


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

@hydra_runner(config_path="conf", config_name="grpo_voicetranslate_v1")
def train(cfg):
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
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
        model = VoiceTranslateGRPO(OmegaConf.to_container(cfg, resolve=True))

        # Load pretrained TTS codec if specified
        if model.cfg.get("pretrained_tts_model", None):
            state = load_checkpoint(model.cfg.pretrained_tts_model)
            state = set_model_dict_for_partial_init(state, model.tts_model.state_dict())
            model.tts_model.load_state_dict(state, strict=True)

        # Load SFT checkpoint as GRPO policy init (most important)
        if model.cfg.get("pretrained_model", None):
            logging.info(f"[GRPO] Loading SFT policy init: {model.cfg.pretrained_model}")
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
        num_delay_speech_tokens=cfg.model.get("num_delay_speech_tokens", 2),
    )
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
