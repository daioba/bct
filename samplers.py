# Modified from S-FLM (https://github.com/jdeschena/s-flm) for BCT: BCT-E, BCT-P and BCT-L in the S-FLM sampler.
"""Implementations of the samplers.

Content:
  - Pure implementations of posteriors, and sampling utils
  - Sampler classes. Looping over step, after state init.
  - Dataclasses naming convention / use case:
      - State: sampler data persisted across steps.
      - Context: data passed to the denoising neural network.
      - View: data derived from state and used only within a
              single step to pass variables around.
"""

import abc
import math
import torch
import torch.nn.functional as F
from dataclass_patch import dataclass
import candi_utils
import utils


def sample_categorical(categorical_probs, gumbel_noise=None):
  """Sample from categorical using the Gumbel trick"""
  if gumbel_noise is None:
    gumbel_noise = (
      1e-10
      - (torch.rand_like(categorical_probs) + 1e-10).log()
    )
  return (categorical_probs / gumbel_noise).argmax(dim=-1)


def _normalize_posterior_inputs(x0_probs, xt, alpha_s, 
                                alpha_t, use_float64=False):
  if use_float64:
    x0_probs = x0_probs.to(torch.float64)
  if alpha_s.ndim == 2:
    alpha_s = alpha_s.unsqueeze(-1)
  if alpha_t.ndim == 2:
    alpha_t = alpha_t.unsqueeze(-1)
  return x0_probs, xt, alpha_s, alpha_t


@torch.compile
def absorbing_posterior_probs(x0_probs, xt, alpha_s, alpha_t,
                              mask_index, vocab_size,
                              use_float64=False):
  """Posterior q(x_s | x_t, x_0) for absorbing diffusion"""
  x0_probs, xt, alpha_s, alpha_t = _normalize_posterior_inputs(
    x0_probs, xt, alpha_s, alpha_t, use_float64=use_float64)
  denoise_prob = (alpha_s - alpha_t) / (1 - alpha_t)
  xt_one_hot = F.one_hot(xt, vocab_size).to(x0_probs)
  is_masked = (xt == mask_index).to(x0_probs.dtype).unsqueeze(-1)
  masked_posterior = (
    xt_one_hot * (1 - denoise_prob)
    + x0_probs * denoise_prob)
  return xt_one_hot * (1 - is_masked) + masked_posterior * is_masked


@torch.compile
def sample_absorbing_posterior(x0_probs, xt, alpha_s, alpha_t,
                               mask_index,
                               noise_removal_step=False):
  """Sample from absorbing posterior without materializing it"""
  sampled_x0 = sample_categorical(x0_probs)

  if noise_removal_step:
    should_denoise = torch.ones_like(sampled_x0, dtype=torch.bool)
  else:
    denoise_prob = (alpha_s - alpha_t) / (1 - alpha_t)
    should_denoise = torch.rand_like(
      sampled_x0, dtype=torch.float64) < denoise_prob

  is_masked = (xt == mask_index)
  should_denoise_mask = is_masked & should_denoise
  return torch.where(should_denoise_mask, sampled_x0, xt)


def _expand_alpha_like(alpha, ref):
  """Expand alpha to match the dimension of ref. One of
     (B, 1), (B, L), (B, L, 1). Output should have ndim=2"""
  if alpha.ndim == 3:
    alpha = alpha.squeeze(-1)
  if alpha.ndim != 2:
    raise ValueError(f'Unexpected alpha shape: {alpha.shape}')
  if alpha.shape[1] == 1:
    return alpha.expand_as(ref)
  if alpha.shape[1] != ref.shape[1]:
    raise ValueError(
      f'Alpha shape {alpha.shape} not broadcastable to {ref.shape}')
  return alpha


@torch.compile
def uniform_posterior_probs(x0_probs, xt, alpha_s, alpha_t,
                            vocab_size, use_float64=False):
  """Posterior q(x_s | x_t, x_0) for uniform diffusion."""
  x0_probs, xt, alpha_s, alpha_t = _normalize_posterior_inputs(
    x0_probs, xt, alpha_s, alpha_t, use_float64=use_float64)
  alpha_ts = alpha_t / alpha_s
  d_alpha = alpha_s - alpha_t
  xt_one_hot = F.one_hot(xt, vocab_size).to(x0_probs)
  p_xt = torch.gather(x0_probs, -1, xt[..., None])
  numerator = (
    alpha_t * vocab_size * x0_probs * xt_one_hot
    + (alpha_ts - alpha_t) * xt_one_hot
    + d_alpha * x0_probs
    + (1 - alpha_ts) * (1 - alpha_s) / vocab_size)
  denominator = alpha_t * vocab_size * p_xt + (1 - alpha_t)
  return numerator / denominator


@torch.compile
def sample_uniform_posterior(x0_probs, xt, alpha_s, alpha_t,
  vocab_size, noise_removal_step=False):
  """Sample from uniform posterior without materializing it."""
  p_xt = torch.gather(x0_probs, -1, xt.unsqueeze(-1)).squeeze(-1)
  alpha_t = _expand_alpha_like(alpha_t, p_xt).to(x0_probs.dtype)
  alpha_s = _expand_alpha_like(alpha_s, p_xt).to(x0_probs.dtype)
  denominator = alpha_t * vocab_size * p_xt + (1 - alpha_t)

  sampled_x0 = sample_categorical(x0_probs)
  sample_threshold = torch.rand_like(p_xt)

  if noise_removal_step:
    keep_xt_prob = (
      alpha_t * vocab_size * p_xt / denominator).clamp(0.0, 1.0)
    return torch.where(sample_threshold < keep_xt_prob,
                       xt, sampled_x0)

  alpha_ts = alpha_t / alpha_s
  sample_uniform_prob = (
    (1 - alpha_ts) * (1 - alpha_s) / denominator).clamp(0.0, 1.0)
  keep_xt_prob = (
    (alpha_t * vocab_size * p_xt + alpha_ts - alpha_t)
    / denominator).clamp(0.0, 1.0)
  uniform_samples = torch.randint(0, vocab_size, xt.shape,
                                  device=xt.device)
  # priority: uniform > xt > x0
  keep_or_resample = torch.where(
    (sample_uniform_prob + keep_xt_prob).clamp(max=1.0)
    > sample_threshold,
    xt,
    sampled_x0)
  return torch.where(sample_uniform_prob > sample_threshold,
                     uniform_samples,
                     keep_or_resample)


def sample_posterior(process, x0_probs, xt, alpha_s, alpha_t, *,
  mask_index=None, vocab_size, noise_removal_step=False,
  posterior_sampler='fast', use_float64=False):
  if process == 'absorbing':
    return sample_absorbing_posterior(
      x0_probs=x0_probs, xt=xt,
      alpha_s=alpha_s, alpha_t=alpha_t,
      mask_index=mask_index,
      noise_removal_step=noise_removal_step)
  elif process == 'uniform':
    if posterior_sampler == 'fast':
      return sample_uniform_posterior(
        x0_probs=x0_probs, xt=xt,
        alpha_s=alpha_s, alpha_t=alpha_t,
        vocab_size=vocab_size,
        noise_removal_step=noise_removal_step)
    elif posterior_sampler == 'naive':
      posterior_probs = compute_posterior(
        process, x0_probs, xt, alpha_s, alpha_t,
        mask_index=mask_index,
        vocab_size=vocab_size,
        use_float64=use_float64)
      return sample_categorical(posterior_probs)
    else:
      raise ValueError(posterior_sampler)
  else:
    raise ValueError(process)


def compute_posterior(process, x0_probs, xt, alpha_s, alpha_t, *,
  mask_index=None, vocab_size, use_float64=False):
  if process == 'absorbing':
    return absorbing_posterior_probs(
      x0_probs=x0_probs,
      xt=xt,
      alpha_s=alpha_s,
      alpha_t=alpha_t,
      mask_index=mask_index,
      vocab_size=vocab_size,
      use_float64=use_float64)
  if process == 'uniform':
    return uniform_posterior_probs(
      x0_probs=x0_probs,
      xt=xt,
      alpha_s=alpha_s,
      alpha_t=alpha_t,
      vocab_size=vocab_size,
      use_float64=use_float64)
  raise ValueError(f'Unknown process: {process}')


def _maybe_cast_log_probs(log_probs, use_float64):
  if use_float64:
    return log_probs.to(torch.float64)
  return log_probs


def _model_mask_index(model):
  if model.diffusion_type != 'absorbing':
    return None
  if not hasattr(model, 'mask_index'):
    raise AttributeError(
      'Absorbing diffusion models must define model.mask_index')
  return model.mask_index


def _decode_direct_tokens(log_probs, greedy):
  if greedy:
    return log_probs.argmax(dim=-1)
  return sample_categorical(log_probs.exp())


def _decode_posterior_tokens(
    model, log_probs, current_tokens, alpha_s, alpha_t, *,
    greedy, noise_removal_step, use_float64):
  probs = log_probs.exp()
  mask_index = _model_mask_index(model)
  if greedy:
    posterior = compute_posterior(
      model.diffusion_type, probs, current_tokens,
      alpha_s, alpha_t,
      mask_index=mask_index,
      vocab_size=model.vocab_size,
      use_float64=use_float64)
    return posterior.argmax(dim=-1)
  return sample_posterior(
    model.diffusion_type, probs, current_tokens,
    alpha_s, alpha_t,
    mask_index=mask_index,
    vocab_size=model.vocab_size,
    noise_removal_step=noise_removal_step)


def _decode_ancestral_update(
    sampler, model, log_probs, current_tokens, alpha_s, alpha_t, *,
    is_last_step):
  """Generic ancestral transition kernel on an arbitrary token window.

  This is the shared black-box update used by the full-sequence ancestral
  sampler and the block samplers. Scheduling and state transitions stay
  outside this function; it only maps:
    current_tokens_t + model log p(x0 | xt) + (alpha_t, alpha_s)
  to:
    next_tokens_s
  """
  if is_last_step and sampler.noise_removal == 'greedy':
    return log_probs.argmax(dim=-1)
  return _decode_posterior_tokens(
    model, log_probs, current_tokens, alpha_s, alpha_t,
    greedy=False,
    noise_removal_step=is_last_step,
    use_float64=sampler.use_float64)


def _early_stop_token(state, xt, idx, tokenizer):
  """Early stopping for token-by-token generation.

  This is useful for tasks where generation should stop at 
  the first EOS. For example, for GSM8K, we only care about 
  generating one answer.

  Called after a single token has been written to xt[:, idx]. 
  Return true when all sequences are done.
  """
  eos_id = tokenizer.eos_token_id
  pad_id = tokenizer.pad_token_id

  newly_finished = (~state.finished) & (xt[:, idx] == eos_id)
  if newly_finished.any():
    xt[newly_finished, idx + 1:] = pad_id
  state.finished = state.finished | newly_finished
  return bool(state.finished.all())


def _early_stop_block(state, block_start, block_end, tokenizer):
  """Early stopping for block-by-block generation.

  This is useful for tasks where generation should stop at 
  the first EOS. For example, for GSM8K, we only care about 
  generating one answer.

  Called after the current block has been finalized. Return 
  true when all sequences are done
  """
  eos_id = tokenizer.eos_token_id
  pad_id = tokenizer.pad_token_id
  xt = state.xt
  # Detect newly-finished sequences (EOS in block, not already done).
  block = xt[:, block_start: block_end]  # [B, block_size]
  eos_mask = block == eos_id  # [B, block_size]
  eos_mask[state.finished] = False  # exclude already-done rows
  newly_done = eos_mask.any(dim=1)  # [B]

  if newly_done.any():
    eos_pos = eos_mask.int().argmax(dim=1)  # first EOS pos per row
    for b in newly_done.nonzero(as_tuple=False).view(-1):
      ep = eos_pos[b].item()
      xt[b, block_start + ep + 1:] = pad_id
    state.finished = state.finished | newly_done
  return bool(state.finished.all())


# ----------------------------------------------------------
# Sampler classes, actually used to generate
# ----------------------------------------------------------
@dataclass(kw_only=True)
class BaseState:
  prefix_tokens: torch.Tensor = None   # (B, max_prefix_len) or (B, max_prefix_len, V)
  prefix_lengths: torch.Tensor = None  # (B,) int64


class Sampler(abc.ABC):
  """API that all samplers must comply with."""

  @abc.abstractmethod
  def init_state(self, model, num_samples, *, num_steps,
                 eps, prefix_tokens, prefix_lengths):
    ...

  @abc.abstractmethod
  def step(self, model, state):
    ...

  def _validate_prefix_args(self, prefix_tokens, prefix_lengths):
    assert (prefix_tokens is None) == (prefix_lengths is None), \
      'prefix_tokens and prefix_lengths must both be set or both be None'

  def _project_prefix(self, xt, prefix_tokens, prefix_lengths):
    if prefix_tokens is None:
      return xt
    P = prefix_tokens.shape[1]
    positions = torch.arange(P, device=xt.device)
    mask = positions[None, :] < prefix_lengths[:, None]  # (B, P)
    if xt.ndim == 3:
      mask = mask.unsqueeze(-1)
    xt[:, :P] = torch.where(mask, prefix_tokens, xt[:, :P])
    return xt

  def metadata(self, state):
    return {'nfe': state.nfe}


@dataclass(kw_only=True)
class AncestralState(BaseState):
  xt: torch.Tensor
  timesteps: torch.Tensor
  ones: torch.Tensor
  start_idx: int
  step_idx: int
  nfe: int
  done: bool


@dataclass
class AncestralContext:
  temperature: float = 1.0
  kv_cache: bool = False


class AncestralSampler(Sampler):
  """Full-sequence ancestral sampler for MDLM & DUO"""

  def __init__(self, use_float64=False,
               noise_removal='ancestral', steps_policy='full',
               temperature=1.0):
    self.use_float64 = use_float64
    self.noise_removal = noise_removal
    self.steps_policy = steps_policy
    self.temperature = temperature
    assert noise_removal in ('ancestral', 'greedy')
    assert steps_policy in ('full', 'proportional')

  def init_state(self, model, num_samples, *,
                 num_steps=None, eps=1e-5, prefix_tokens=None,
                 prefix_lengths=None):
    self._validate_prefix_args(prefix_tokens, prefix_lengths)
    xt = model.prior_sample(num_samples, model.num_tokens)
    if prefix_tokens is not None:
      start_idx = int(prefix_lengths.min())
      self._project_prefix(xt, prefix_tokens, prefix_lengths)
    else:
      start_idx = 0

    if num_steps is None:
      num_steps = model.config.sampler.steps
    # Adapt number of sampling steps to the number of tokens
    #  left to sample.
    if self.steps_policy == 'proportional':
      gen_len = model.num_tokens - start_idx
      num_steps = int(round(num_steps * gen_len))

    timesteps = torch.linspace(1, eps, num_steps + 1,
                               device=model.device)
    ones = torch.ones(xt.shape[0], 1, dtype=model.dtype,
                      device=model.device)
    state = AncestralState(
      xt=xt, timesteps=timesteps, ones=ones, start_idx=start_idx,
      step_idx=0, nfe=0, done=False,
      prefix_tokens=prefix_tokens,
      prefix_lengths=prefix_lengths)
    return state

  def step(self, model, state):
    num_steps = len(state.timesteps) - 1
    is_last_step = (state.step_idx == num_steps - 1)
    t = state.timesteps[state.step_idx] * state.ones
    s = state.timesteps[state.step_idx + 1] * state.ones

    _, alpha_t = model.noise(t)
    if is_last_step:
      alpha_s = torch.ones_like(alpha_t)
    else:
      _, alpha_s = model.noise(s)
    sigma_t = model._sigma_from_alphat(alpha_t)

    log_probs = _maybe_cast_log_probs(
      model.forward(xt=state.xt, sigma=sigma_t,
                    context=AncestralContext(temperature=self.temperature)),
      self.use_float64)

    current_tokens = state.xt[:, state.start_idx:]
    log_probs_window = log_probs[:, state.start_idx:]

    current_tokens[:] = _decode_ancestral_update(
      self, model, log_probs_window, current_tokens,
      alpha_s, alpha_t, is_last_step=is_last_step)

    self._project_prefix(state.xt, state.prefix_tokens, 
                         state.prefix_lengths)
    state.nfe += 1
    state.step_idx += 1
    if is_last_step:
      state.done = True
    return state

@dataclass(kw_only=True)
class ARState(BaseState):
  xt: torch.Tensor
  ones: torch.Tensor
  zeros: torch.Tensor
  start_idx: int
  token_idx: int
  cached_len: int   # number of tokens committed to the KV cache
  nfe: int
  done: bool
  # Whether each xt has already generated an EOS. Only used 
  #  when early_stopping is True.
  finished: torch.Tensor = None  


@dataclass
class ARContext:
  kv_cache: bool
  temperature: float = 1.0


class ARSampler(Sampler):
  """Autoregressive sampler, generate tokens L2R, feed prefix
     only (suggested to use with causal transformers only)"""

  def __init__(self, use_float64, kv_cache, greedy,
               early_stopping, temperature=1.0):
    self.use_float64 = use_float64
    self.kv_cache = kv_cache
    self.greedy = greedy
    self.early_stopping = early_stopping
    self.temperature = temperature

  def init_state(self, model, num_samples, *,
                 num_steps=None, eps=1e-5, prefix_tokens=None,
                 prefix_lengths=None):
    if prefix_lengths is not None and prefix_lengths.unique().numel() > 1:
      raise NotImplementedError(
        'ARSampler does not support variable-length prefixes')
    xt = torch.zeros(num_samples, model.num_tokens,
                     dtype=torch.long, device=model.device)
    if prefix_tokens is not None:
      self._project_prefix(xt, prefix_tokens, prefix_lengths)
      start_idx = int(prefix_lengths.max())
    else:
      xt[:, 0] = model.tokenizer.bos_token_id
      start_idx = 1
    ones = torch.ones(num_samples, 1, dtype=model.dtype,
                      device=model.device)
    zeros = torch.zeros_like(ones)
    if self.kv_cache:
      model.reset_kv_cache()
    if self.early_stopping:
      finished = torch.zeros(num_samples, dtype=torch.bool,
                             device=model.device)
    else:
      finished = None
    state = ARState(xt=xt, ones=ones, zeros=zeros,
      start_idx=start_idx, token_idx=start_idx, cached_len=0,
      nfe=0, done=(start_idx >= xt.shape[1]), finished=finished)
    return state

  def step(self, model, state):
    x = state.xt
    if self.kv_cache:
      start_idx = state.cached_len
    else:
      start_idx = 0
    end_idx = state.token_idx - 1

    x_input = x[:, start_idx: end_idx + 1]
    log_p = model.forward(xt=x_input, sigma=state.zeros,
      context=ARContext(kv_cache=self.kv_cache,
                        temperature=self.temperature))

    log_probs = _maybe_cast_log_probs(log_p[:, -1], self.use_float64)
    new_tok = _decode_direct_tokens(log_probs, greedy=self.greedy)

    x[:, state.token_idx] = new_tok

    if self.early_stopping:
      _early_stop_token(state, x, state.token_idx, model.tokenizer)

    if self.kv_cache:
      state.cached_len = state.token_idx
    state.nfe += 1
    state.token_idx += 1
    state.done = state.token_idx >= x.shape[1]
    if self.early_stopping and bool(state.finished.all()):
      state.done = True
    return state


@dataclass(kw_only=True)
class SFMState(BaseState):
  # [B, L, d] float sphere embeddings during integration;
  # replaced by [B, L] int token ids at the final decoding step.
  xt: torch.Tensor
  t_schedule: torch.Tensor
  start_idx: int
  step_idx: int
  nfe: int
  done: bool
  prefix_lengths: torch.Tensor = None  # [B] per-sample lengths
  prefix_tokens: torch.Tensor = None  # [B, L]
  prefix_embeds: torch.Tensor = None  # [B, P, d] sphere embeddings of prefix
  ode_evals: int = 0
  forward_calls: int = 0


@dataclass
class SFMContext:
  temperature: float = 0.0
  # Temperature conditioning of a TFT model: tau = -log T_path.
  bct_tau: float = 0.0

@torch.compile
def sfm_compute_velocity(x, E, log_p, mode, eps, cos_omega=None, omega=None,
                         vel_scale=None):
  # `vel_scale`: per-candidate weight w_{a,T}(theta_x) of BCT-P / BCT-L.
  p = log_p.exp()
  if mode == 'exact':
    if E.ndim == 2:  # [V, d]
      ein_fwd = 'bld,vd->blv'
      ein_bwd = 'blv,vd->bld'
    else:           # [B, L, k, d]
      ein_fwd = 'bld,blkd->blk'
      ein_bwd = 'blk,blkd->bld'
    if cos_omega is None:
      cos_omega = torch.einsum(ein_fwd, x, E)
      cos_omega = cos_omega.clamp(-1 + eps, 1 - eps)
    if omega is None:
      omega = torch.acos(cos_omega)
    scale = omega / omega.sin().clamp(min=eps)
    p_scale = p * scale
    if vel_scale is not None:
      p_scale = p_scale * vel_scale
    term1 = torch.einsum(ein_bwd, p_scale, E)
    term2 = x * (p_scale * cos_omega).sum(dim=-1, keepdim=True)
    return term1 - term2
  elif mode == 'sample':
    target_idx = sample_categorical(p)
    if E.ndim == 2:  # [V, d]
      target = E[target_idx]
    else:            # [B, L, k, d]
      B, L = target_idx.shape
      target = E[
        torch.arange(B, device=E.device)[:, None],
        torch.arange(L, device=E.device)[None, :],
        target_idx]
    return utils.log_map(x, target, eps)
  else:
    raise ValueError(f'Unknown velocity mode: {mode}')


# ════════════════════════════════════════════════════════════════
#  BCT-E (endpoint tempering)
#
#      l^{E,T}_x = l_x / T + (1 - 1/T) log q~_a(z|x),
#      log q~_a(z|x) = (d - 2) [ log sin(theta_x / a) - log sin theta_x ],
#      theta_x = arccos<z, e_hat_x> < a * pi.
# ════════════════════════════════════════════════════════════════


def sfm_log_q_unnorm(cos_omega, a, d):
  """log q~_a(z|x), up to a constant; `-inf` outside the support."""
  theta = torch.acos(cos_omega)
  a_t = torch.as_tensor(a, dtype=theta.dtype, device=theta.device)
  in_support = theta < a_t * math.pi
  ratio = torch.where(in_support, theta / a_t, torch.zeros_like(theta))
  small = theta < 1e-12
  log_ratio = torch.where(
    small,
    -torch.log(a_t).expand_as(theta),
    torch.log(torch.sin(ratio).clamp(min=1e-300))
    - torch.log(torch.sin(theta).clamp(min=1e-300)))
  out = (d - 2) * log_ratio
  return torch.where(in_support, out,
                     torch.full_like(out, -float('inf')))


def bct_endpoint_logits(log_p, log_q, T):
  """BCT-E logits; candidates outside the support get probability 0."""
  masked = torch.isneginf(log_q)
  safe_log_q = torch.where(masked, torch.zeros_like(log_q), log_q)
  out = log_p / T + (1.0 - 1.0 / T) * safe_log_q
  return torch.where(masked, torch.full_like(out, -float('inf')), out)


def sfm_slerp_param(alpha_t, invert_time_convention):
  """The SLERP parameter `a` of `algo.SFM.q_xt` (1: noise, 0: clean)."""
  return alpha_t if invert_time_convention else 1.0 - alpha_t


def sfm_step_size(alpha_t, alpha_s, invert_time_convention, eps):
  if invert_time_convention:
    # slerp param = alpha_t, decreasing along schedule
    return (alpha_t - alpha_s) / alpha_t.clamp(min=eps)
  else:
    # slerp param = 1 - alpha_t, decreasing along schedule
    return (alpha_s - alpha_t) / (1 - alpha_t).clamp(min=eps)


# ════════════════════════════════════════════════════════════════
#  BCT-P (path tempering).  The conditional velocity of the tempered bridge
#  q_{a,T}(z|x) ∝ q_a(z|x)^{1/T} is the S-FLM one rescaled by
#
#      w_{a,T}(theta) = a v_{a,T}(theta) / theta,
#      v_{a,T}(theta) = (nu*beta / a^2)
#                       INT_0^theta [s cot(s/a) - Hbar] g(s) ds / g(theta),
#      g(theta) = sin^{nu*beta}(theta/a) sin^{nu*(1-beta)}(theta),
#      nu = d - 2,  beta = 1/T,  Hbar = E_m[theta cot(theta/a)],
#
#  and w == 1 at T=1.
# ════════════════════════════════════════════════════════════════

# Quadrature / interpolation resolution.
BCT_PATH_N_QUAD = 1 << 20
BCT_PATH_N_GRID = 1 << 16
BCT_PATH_PHI_MIN = 1e-9


def _bct_log_sinh_over_x(h):
  """log( sinh(h) / h ), stable for both small and large |h|."""
  ah = h.abs()
  safe = ah.clamp(min=1e-6)
  big = (safe + torch.log1p(-torch.exp(-2.0 * safe)) - math.log(2.0)
         - torch.log(safe))
  return torch.where(ah > 1e-6, big, ah * ah / 6.0)


def _bct_log_g(p, a, nu, beta):
  """`log g_{a,T}(theta)` in `phi = theta/(a*pi)`, up to normalisation."""
  out = nu * beta * torch.log(torch.sin(math.pi * p))
  c = nu * (1.0 - beta)
  if torch.is_tensor(c):
    return out + c * torch.log(torch.sin(a * math.pi * p))
  if c == 0.0:
    return out + 0.0 * a
  return out + c * torch.log(torch.sin(a * math.pi * p))


def _bct_graded_edges(n_side, phi_min, split, device, dtype):
  """Cell edges on [0, 1], geometrically refined towards both endpoints."""
  xi = torch.arange(n_side + 1, device=device, dtype=dtype) / n_side
  left = split * (phi_min / split) ** (1.0 - xi)
  right = 1.0 - (1.0 - split) * (phi_min / (1.0 - split)) ** xi
  return torch.cat([left, right[1:]])


def _bct_signed_cumsum_log(log_abs, sign, *, reverse):
  """Signed exclusive cumulative sum in log space: `(log|S|, sign(S))`."""
  neg_inf = torch.full_like(log_abs[:1], -float('inf'))
  lp = torch.where(sign > 0, log_abs, torch.full_like(log_abs, -float('inf')))
  ln = torch.where(sign < 0, log_abs, torch.full_like(log_abs, -float('inf')))
  if reverse:
    lp = torch.cat([torch.flip(torch.logcumsumexp(torch.flip(lp, [0]), 0), [0]),
                    neg_inf])
    ln = torch.cat([torch.flip(torch.logcumsumexp(torch.flip(ln, [0]), 0), [0]),
                    neg_inf])
  else:
    lp = torch.cat([neg_inf, torch.logcumsumexp(lp, 0)])
    ln = torch.cat([neg_inf, torch.logcumsumexp(ln, 0)])
  hi = torch.maximum(lp, ln)
  lo = torch.minimum(lp, ln)
  sgn = torch.where(lp >= ln, torch.ones_like(lp), -torch.ones_like(lp))
  gap = (lo - hi).clamp(max=-1e-12)
  out = hi + torch.log1p(-torch.exp(gap))
  out = torch.where(torch.isneginf(hi),
                    torch.full_like(out, -float('inf')), out)
  return out, sgn


class BCTPathTable:
  """`w_{a,T}(theta)` for one `(a, d, T)` on a uniform grid in `phi`."""

  __slots__ = ('a', 'd', 'T', 'n', 'w', 'Hbar')

  def __init__(self, a, d, T, n, w, Hbar):
    self.a, self.d, self.T = float(a), int(d), float(T)
    self.n, self.w, self.Hbar = int(n), w, float(Hbar)

  def weight(self, theta):
    """`theta -> w_{a,T}(theta)`; 1 outside the support."""
    phi = theta.div(self.a * math.pi)
    phi.clamp_(0.0, 1.0).mul_(self.n)
    idx = phi.floor().clamp_(0, self.n - 1)
    frac = phi.sub_(idx)
    idx = idx.to(torch.long)
    lo = self.w[idx]
    idx += 1
    hi = self.w[idx]
    del idx
    lo.lerp_(hi, frac)
    return lo


def bct_path_table(a, d, T, *, n_grid=BCT_PATH_N_GRID, n_quad=BCT_PATH_N_QUAD,
                   phi_min=BCT_PATH_PHI_MIN, device='cpu',
                   dtype=torch.float64):
  """The `w_{a,T}` table for one `(a, d, T)`, by quadrature in log space."""
  a = float(a)
  nu = float(d - 2)
  beta = 1.0 / float(T)
  edges = _bct_graded_edges(int(n_quad), phi_min, 0.5, device, dtype)
  lo, hi = edges[:-1], edges[1:]
  ctr = 0.5 * (lo + hi)
  wid = hi - lo

  def _log_g(p):
    return _bct_log_g(p, a, nu, beta)

  e_lo, e_hi, e_c = _log_g(lo), _log_g(hi), _log_g(ctr)
  log_cell = e_c + torch.log(wid) + _bct_log_sinh_over_x(0.5 * (e_hi - e_lo))
  # f(theta) = theta cot(theta/a)
  f_c = ((a * math.pi) * ctr * torch.cos(math.pi * ctr)
         / torch.sin(math.pi * ctr))

  wts = torch.exp(log_cell - log_cell.max())
  Hbar = (f_c * wts).sum() / wts.sum()

  fh = f_c - Hbar
  pos = fh > 0
  if not bool((pos[:-1] | ~pos[1:]).all()):
    raise RuntimeError('[f - Hbar] changes sign more than once; the BCT-P '
                       'quadrature grid is not usable for (a=%r, d=%r, T=%r)'
                       % (a, d, T))
  term = torch.log(fh.abs().clamp(min=1e-300)) + log_cell
  sign = torch.sign(fh)

  logS_f, sgn_f = _bct_signed_cumsum_log(term, sign, reverse=False)
  logS_b, sgn_b = _bct_signed_cumsum_log(term, sign, reverse=True)
  sgn_b = -sgn_b   # S_b(phi) = -INT_phi^1, since the total integral is 0

  cross = int((fh > 0).sum().item())
  use_f = torch.arange(edges.numel(), device=device) <= cross
  logS = torch.where(use_f, logS_f, logS_b)
  sgn = torch.where(use_f, sgn_f, sgn_b)

  log_w = (logS - _log_g(edges) + math.log(nu * beta / a)
           - torch.log(edges))
  w_knots = sgn * torch.exp(log_w)
  # Endpoints analytically: w -> nu*beta (a - Hbar) / (a (nu+1)) as phi -> 0,
  # and w -> 1 (a < 1) or 1/T (a = 1) as phi -> 1.
  w0 = nu * beta / (a * (nu + 1.0)) * (a - float(Hbar))
  p_edge = nu * beta + (nu * (1.0 - beta) if a >= 1.0 else 0.0)
  w1 = nu * beta / p_edge
  w_knots[0] = w0
  w_knots[-1] = w1
  phi_knots = torch.cat([edges.new_zeros(1), edges, edges.new_ones(1)])
  w_knots = torch.cat([edges.new_tensor([w0]), w_knots,
                       edges.new_tensor([w1])])

  # Resample onto a uniform grid in phi.
  n = int(n_grid)
  out_phi = torch.arange(n + 1, device=device, dtype=dtype) / n
  idx = torch.searchsorted(phi_knots.contiguous(), out_phi.contiguous()).clamp(
    1, phi_knots.numel() - 1)
  p0, p1 = phi_knots[idx - 1], phi_knots[idx]
  frac = ((out_phi - p0) / (p1 - p0).clamp(min=1e-300)).clamp(0.0, 1.0)
  w = torch.lerp(w_knots[idx - 1], w_knots[idx], frac)
  return BCTPathTable(a, d, T, n, w, Hbar)


# ── Tempered-flow training (TFT): sampling z ~ q_{a,T}(.|x)
#
#  z = cos(theta) e_x + sin(theta) u, with theta = Q_{a,T}(omega) the quantile
#  transport of the polar angle omega of the uniform source; at T=1,
#  theta = a*omega, which is upstream `utils.slerp`.

# Quadrature resolution.
BCT_CORRUPT_N_QUAD = 1 << 18
BCT_CORRUPT_PHI_MIN = 1e-9

def _bct_edge_cdf(edges, a, nu, beta):
  """Normalised CDF of `g_{a,T}` at the quadrature edges."""
  lo, hi = edges[:-1], edges[1:]
  ctr = 0.5 * (lo + hi)
  wid = hi - lo
  e_lo = _bct_log_g(lo, a, nu, beta)
  e_hi = _bct_log_g(hi, a, nu, beta)
  e_c = _bct_log_g(ctr, a, nu, beta)
  log_cell = e_c + torch.log(wid) + _bct_log_sinh_over_x(0.5 * (e_hi - e_lo))
  log_cdf = torch.logcumsumexp(log_cell, dim=-1)
  log_cdf = log_cdf - log_cdf[..., -1:]
  cdf = torch.exp(log_cdf)
  zero = cdf.new_zeros(cdf.shape[:-1] + (1,))
  cdf = torch.cat([zero, cdf], dim=-1)
  cdf[..., -1] = 1.0
  return cdf



def bct_corruption_transport(psi, a, T, d, *, n_quad=BCT_CORRUPT_N_QUAD,
                             phi_min=BCT_CORRUPT_PHI_MIN, row_chunk=16):
  """Quantile transport `psi = omega/pi -> phi = theta/(a*pi)` for a
  per-example `(a, T)`; the identity at `T = 1`."""
  device, dtype = psi.device, psi.dtype
  a_f = a.reshape(-1).to(dtype)
  T_f = T.reshape(-1).to(dtype)
  n_rows = int(psi.shape[0])
  if a_f.numel() == 1:
    a_f = a_f.expand(n_rows)
  if T_f.numel() == 1:
    T_f = T_f.expand(n_rows)
  if a_f.numel() != n_rows or T_f.numel() != n_rows:
    raise ValueError(
      f'bct_corruption_transport needs one (a, T) per row of psi: psi has '
      f'{n_rows} rows but a has {a_f.numel()} and T has {T_f.numel()}.')
  if float(a_f.min()) <= 0.0:
    raise ValueError('bct_corruption_transport needs a > 0 (the a = 0 row is '
                     'the source density and is handled separately).')
  if float(T_f.min()) <= 0.0:
    raise ValueError('bct_corruption_transport needs T > 0.')
  nu = float(d - 2)
  beta_f = 1.0 / T_f

  edges = _bct_graded_edges(int(n_quad), phi_min, 0.5, device, dtype)
  cdf_src = _bct_edge_cdf(edges, torch.zeros((), device=device, dtype=dtype),
                          nu, 1.0)                            # [n_quad+1]

  flat = psi.reshape(n_rows, -1)                              # [B, M]
  j = torch.searchsorted(edges.contiguous(), flat.contiguous()).clamp(
    1, edges.numel() - 1)
  e0, e1 = edges[j - 1], edges[j]
  frac = ((flat - e0) / (e1 - e0).clamp(min=1e-300)).clamp(0.0, 1.0)
  u = torch.lerp(cdf_src[j - 1], cdf_src[j], frac)            # [B, M]
  del j, e0, e1, frac

  out = torch.empty_like(flat)
  for s0 in range(0, n_rows, int(row_chunk)):
    s1 = min(s0 + int(row_chunk), n_rows)
    cdf_tgt = _bct_edge_cdf(edges, a_f[s0:s1, None], nu,
                            beta_f[s0:s1, None])              # [k, n_quad+1]
    u_blk = u[s0:s1].contiguous()
    k = torch.searchsorted(cdf_tgt.contiguous(), u_blk).clamp(
      1, cdf_tgt.shape[-1] - 1)
    f0 = torch.gather(cdf_tgt, -1, k - 1)
    f1 = torch.gather(cdf_tgt, -1, k)
    w = ((u_blk - f0) / (f1 - f0).clamp(min=1e-300)).clamp(0.0, 1.0)
    out[s0:s1] = torch.lerp(edges[k - 1], edges[k], w)
    del cdf_tgt, u_blk, k, f0, f1, w
  return out.reshape(psi.shape).clamp(0.0, 1.0)


def bct_trained_path_logits(log_p, log_q, T, strength=1.0):
  """BCT-L logits for a TFT model,

      l = log_p / T + strength * (1/T)(1 - 1/T) log_q,

  where `log_p` is the model's log-softmax at temperature 1, `log_q` is
  `log q~_a(z|x)` and `strength` is lambda.
  """
  if strength == 1.0:
    return bct_endpoint_logits(log_p, log_q / T, T)
  masked = torch.isneginf(log_q)
  safe_log_q = torch.where(masked, torch.zeros_like(log_q), log_q)
  out = log_p / T + strength * (1.0 - 1.0 / T) * (safe_log_q / T)
  if strength == 0.0:
    return out
  return torch.where(masked, torch.full_like(out, -float('inf')), out)


class SFMSampler(Sampler):
  def __init__(self, noise_removal, velocity, use_float64,
               slerp_float64, eps, temperature, p_nucleus,
               top_k,
               top_k_velocity,
               invert_time_convention,
               bct_mode='none', bct_T=1.0, bct_correction_strength=1.0):
    # BCT sampling modes: none (standard S-FLM sampling), endpoint (BCT-E),
    # path (BCT-P) and trained_path (BCT-L for a TFT model, with lambda =
    # bct_correction_strength).
    self.bct_mode = bct_mode
    self.bct_T = bct_T
    self.bct_correction_strength = float(bct_correction_strength)
    if self.bct_correction_strength != 1.0:
      if bct_mode != 'trained_path':
        raise ValueError(
          'sampler.bct.correction_strength != 1 is only defined for '
          f'bct.mode = trained_path (got {bct_mode}).')
      if self.bct_correction_strength < 0.0:
        raise ValueError('sampler.bct.correction_strength must be >= 0, got '
                         f'{self.bct_correction_strength}.')
    # Support diagnostics, reported through `metadata()`.
    self._bct_n_oos = None
    self._bct_margin_min = None
    self.bct_n_candidates = 0
    # BCT-P / BCT-L: running summary of the velocity weight w_{a,T}.
    self._bct_w_min = None
    self._bct_w_max = None
    self._bct_w_sum = None
    self._bct_w_count = 0
    if bct_mode not in ('none', 'endpoint', 'path', 'trained_path'):
      raise ValueError(f'Unknown sampler.bct.mode: {bct_mode}')
    if bct_mode != 'none':
      if bct_T <= 0.0:
        raise ValueError(f'sampler.bct.T must be > 0, got {bct_T}')
    if bct_mode == 'endpoint':
      if temperature != 1.0:
        raise ValueError(
          'sampler.bct.mode = endpoint requires sampler.temperature == 1.0 '
          f'(got {temperature}); the 1/T is applied by BCT-E itself.')
    if bct_mode == 'trained_path':
      if temperature != 1.0:
        raise ValueError(
          'sampler.bct.mode = trained_path requires sampler.temperature == '
          f'1.0 (got {temperature}); the 1/T is applied inside the BCT-L '
          'correction, not by sampler.temperature.')
      if velocity != 'exact':
        raise ValueError(
          'sampler.bct.mode = trained_path needs velocity=exact (got '
          f'{velocity}).')
      if top_k_velocity > 0:
        raise ValueError(
          'sampler.bct.mode = trained_path needs top_k_velocity=-1 (got '
          f'{top_k_velocity}).')
    if bct_mode == 'path':
      if temperature != bct_T:
        raise ValueError(
          f'sampler.bct.mode = {bct_mode} requires sampler.temperature == '
          f'sampler.bct.T (got temperature={temperature}, T={bct_T}); the '
          'BCT-P posterior is softmax(l/T), i.e. standard tempering, and the '
          'tempered conditional velocity must use that same T.')
      if velocity != 'exact':
        raise ValueError(
          f'sampler.bct.mode = {bct_mode} needs velocity=exact (got '
          f'{velocity}): the tempered conditional velocity rescales every '
          'candidate, which the sampled single-candidate branch does not '
          'carry.')
      if top_k_velocity > 0:
        raise ValueError(
          f'sampler.bct.mode = {bct_mode} needs top_k_velocity=-1 (got '
          f'{top_k_velocity}): the truncated `E` no longer matches the '
          'full-vocabulary polar angles the weight table is looked up with.')
    self.noise_removal = noise_removal
    self.velocity = velocity
    self.use_float64 = use_float64
    self.slerp_float64 = slerp_float64
    self.eps = eps
    self.temperature = temperature
    self.p_nucleus = p_nucleus
    self.top_k = top_k
    self.top_k_velocity = top_k_velocity
    self.invert_time_convention = invert_time_convention

  def init_state(self, model, num_samples, *,
                 num_steps=None, eps=1e-5, prefix_tokens=None,
                 prefix_lengths=None):
    self._validate_prefix_args(prefix_tokens, prefix_lengths)
    xt = torch.randn(num_samples, model.num_tokens,
      model.backbone.embed_dim, device=model.device,
      dtype=torch.float32)
    xt = utils.sphere_normalize(xt)

    prefix_embeds = None
    if prefix_tokens is not None:
      prefix_embeds = model.backbone.get_sphere_embeddings(
        prefix_tokens)
      self._project_prefix(xt, prefix_embeds, prefix_lengths)
      start_idx = int(prefix_lengths.min())
    else:
      start_idx = 0

    if num_steps is None:
      num_steps = model.config.sampler.steps

    if self.invert_time_convention:
      t_schedule = torch.linspace(eps, 1.0, num_steps + 1,
                                  device=model.device)
    else:
      t_schedule = torch.linspace(1.0, eps, num_steps + 1,
                                  device=model.device)
    state = SFMState(xt=xt, t_schedule=t_schedule,
      start_idx=start_idx, step_idx=0, nfe=0, done=False,
      prefix_lengths=prefix_lengths,
      prefix_embeds=prefix_embeds,
      prefix_tokens=prefix_tokens)
    return state

  def metadata(self, state):
    md = {'nfe': state.nfe,
          'ode_evals': state.ode_evals,
          'forward_calls': state.forward_calls}
    if self.bct_mode != 'none':
      md['bct_mode'] = self.bct_mode
      md['bct_T'] = self.bct_T
      md['bct_n_out_of_support'] = (0 if self._bct_n_oos is None
                                    else int(self._bct_n_oos))
      md['bct_n_candidates'] = self.bct_n_candidates
      md['bct_support_margin_min'] = (float('inf')
                                      if self._bct_margin_min is None
                                      else float(self._bct_margin_min))
    if self.bct_correction_strength != 1.0:
      md['bct_correction_strength'] = self.bct_correction_strength
    if self.bct_mode in ('path', 'trained_path'):
      md['bct_w_min'] = (None if self._bct_w_min is None
                         else float(self._bct_w_min))
      md['bct_w_max'] = (None if self._bct_w_max is None
                         else float(self._bct_w_max))
      md['bct_w_mean'] = (None if self._bct_w_sum is None
                          else float(self._bct_w_sum) / self._bct_w_count)
    return md

  def _last_step_decode(self, state, log_p):
    if self.noise_removal == 'greedy':
      tokens = log_p.argmax(dim=-1)
    elif self.noise_removal == 'ancestral':
      tokens = sample_categorical(log_p.exp())
    else:
      raise ValueError(self.noise_removal)

    if state.prefix_embeds is not None:
      self._project_prefix(tokens, state.prefix_tokens, 
                           state.prefix_lengths)
    state.xt = tokens  # replace continuous [B,L,d] with int [B,L]
    state.done = True
    return state
  
  def _select_topk(self, log_p, E, k):
    log_p_k, top_idxs = torch.topk(log_p, k, dim=-1)
    return torch.log_softmax(log_p_k, dim=-1), F.embedding(top_idxs, E)

  def _compute_velocity(self, x, E, log_p, cos_omega=None, omega=None,
                        vel_scale=None):
    return sfm_compute_velocity(
      x, E, log_p, mode=self.velocity, eps=self.eps,
      cos_omega=cos_omega, omega=omega, vel_scale=vel_scale)

  def _bct_record_support(self, cos_omega, a):
    """Support diagnostics for `metadata()`."""
    a_f = float(a)
    cos_cut = math.cos(min(a_f * math.pi, math.pi))
    n_oos = (cos_omega <= cos_cut).sum()
    margin = a_f * math.pi - torch.acos(cos_omega.min())
    self._bct_n_oos = (n_oos if self._bct_n_oos is None
                       else self._bct_n_oos + n_oos)
    self._bct_margin_min = (margin if self._bct_margin_min is None
                            else torch.minimum(self._bct_margin_min, margin))
    self.bct_n_candidates += cos_omega.numel()

  def _bct_record_weight(self, w):
    w_min, w_max = w.min(), w.max()
    w_sum = w.sum(dtype=torch.float64)
    self._bct_w_min = (w_min if self._bct_w_min is None
                       else torch.minimum(self._bct_w_min, w_min))
    self._bct_w_max = (w_max if self._bct_w_max is None
                       else torch.maximum(self._bct_w_max, w_max))
    self._bct_w_sum = (w_sum if self._bct_w_sum is None
                       else self._bct_w_sum + w_sum)
    self._bct_w_count += w.numel()

  def _model_tau(self):
    """tau = -log T_path: T for `path` / `trained_path`, 1 otherwise."""
    if self.bct_mode in ('path', 'trained_path'):
      return -math.log(float(self.bct_T))
    return 0.0

  def _get_step_size(self, model, state):
    _, alpha_t = model.noise(state.t_schedule[state.step_idx])
    _, alpha_s = model.noise(state.t_schedule[state.step_idx + 1])
    return sfm_step_size(
      alpha_t, alpha_s, self.invert_time_convention, self.eps)

  def step(self, model, state):
    num_steps = len(state.t_schedule) - 1
    is_last_step = (state.step_idx == num_steps - 1)

    _, alpha_t = model.noise(state.t_schedule[state.step_idx])
    sigma_t = model._sigma_from_alphat(alpha_t).reshape(-1, 1)

    context = SFMContext(temperature=self.temperature,
                         bct_tau=self._model_tau())
    log_p = model.forward(xt=state.xt, sigma=sigma_t,
                          context=context)
    if self.use_float64:
      log_p = log_p.to(torch.float64)
    state.nfe += 1
    state.forward_calls += 1

    if self.p_nucleus != 1.0 or self.top_k != -1:
      log_p = utils.top_k_top_p_filtering(log_p, 
        top_k=self.top_k, top_p=self.p_nucleus).log_softmax(-1)

    if is_last_step:
      return self._last_step_decode(state, log_p)
    # Arguments to compute the velocity field:
    #  v = sum_k p_k * log_{x}(e_k).
    log_p_window = log_p[:, state.start_idx:]  # [B, L, V]
    E = utils.sphere_normalize(
      model.backbone.sphere_embed.weight.detach())  # [V, d]
    x = state.xt[:, state.start_idx:].to(E)  # [B, L, d]

    if self.slerp_float64:
      E = E.to(torch.float64)
      x = x.to(torch.float64)

    # BCT: `none` is upstream S-FLM.  BCT-E corrects the posterior, BCT-P
    # rescales the conditional velocity, and BCT-L does both for a TFT model.
    cos_omega = None
    omega = None
    vel_scale = None
    if self.bct_mode in ('endpoint', 'path', 'trained_path'):
      a = sfm_slerp_param(alpha_t, self.invert_time_convention)
      cos_omega = torch.einsum('bld,vd->blv', x, E).clamp(
        -1 + self.eps, 1 - self.eps)
      self._bct_record_support(cos_omega, a)

    if self.bct_mode == 'endpoint':
      log_q = sfm_log_q_unnorm(cos_omega, a, model.backbone.embed_dim)
      dt_ = torch.promote_types(log_p_window.dtype, log_q.dtype)
      log_p_window = torch.log_softmax(
        bct_endpoint_logits(log_p_window.to(dt_), log_q.to(dt_),
                            self.bct_T), dim=-1)
      del log_q
    elif self.bct_mode == 'trained_path':
      log_q = sfm_log_q_unnorm(cos_omega, a, model.backbone.embed_dim)
      dt_ = torch.promote_types(log_p_window.dtype, log_q.dtype)
      log_p_in = log_p_window.to(dt_)
      log_p_window = torch.log_softmax(
        bct_trained_path_logits(log_p_in, log_q.to(dt_),
                                self.bct_T,
                                strength=self.bct_correction_strength),
        dim=-1)
      del log_q
      omega = torch.acos(cos_omega)
      table = bct_path_table(a, model.backbone.embed_dim, self.bct_T,
                             device=omega.device, dtype=omega.dtype)
      vel_scale = table.weight(omega)
      self._bct_record_weight(vel_scale)
    elif self.bct_mode == 'path':
      omega = torch.acos(cos_omega)
      table = bct_path_table(a, model.backbone.embed_dim, self.bct_T,
                             device=omega.device, dtype=omega.dtype)
      vel_scale = table.weight(omega)
      self._bct_record_weight(vel_scale)

    if self.top_k_velocity > 0:
      log_p_v, E = self._select_topk(log_p_window, E, self.top_k_velocity)
      cos_omega = None
      omega = None
    else:
      log_p_v = log_p_window

    vel = self._compute_velocity(x, E, log_p_v, cos_omega=cos_omega,
                                 omega=omega, vel_scale=vel_scale)
    state.ode_evals += 1

    dt = self._get_step_size(model, state)
    x_new = utils.exp_map(x, dt * vel, self.eps)
    state.xt[:, state.start_idx:] = x_new.to(state.xt.dtype)
    self._project_prefix(
      state.xt, state.prefix_embeds, state.prefix_lengths)
    state.step_idx += 1
    return state

# ════════════════════════════════════════════════════════════════
#  FLM Samplers
# ════════════════════════════════════════════════════════════════

# ── Pure step functions (usable from distillation training) ──
def flm_euler_step(z, x_1_pred_probs, gamma_t, gamma_d):
  """One Euler ODE step for FLM.

  Args:
    z: current state (B, L, V)
    x_1_pred_probs: exp(log_softmax(model_output)) (B, L, V)
    gamma_t: current gamma (B,) or scalar
    gamma_d: step size in gamma space (B,) or scalar
  Returns:
    updated z (B, L, V)
  """
  if not isinstance(gamma_t, (int, float)):
    gamma_t = gamma_t.view(-1, 1, 1)
  if not isinstance(gamma_d, (int, float)):
    gamma_d = gamma_d.view(-1, 1, 1)
  v = (x_1_pred_probs - z) / (1.0 - gamma_t + 1e-5)
  return z + gamma_d * v


@dataclass(kw_only=True)
class FLMState(BaseState):
  xt: torch.Tensor        # (B, L, V) continuous
  t_vals: torch.Tensor    # (num_steps+1,) time grid
  step_idx: int
  nfe: int
  done: bool
  start_idx: int          # prefix length


@dataclass
class FLMContext:
  temperature: float = 1.0


class FLMEulerSampler(Sampler):
  """Euler ODE sampler for base FLM.

  NOTE: unlike the original FLM codebase, we cast to float64 by
  default for consistency with MDLM / DUO samplers.
  """

  def __init__(self, use_float64=False, temperature=1.0):
    self.use_float64 = use_float64
    self.temperature = temperature

  def init_state(self, model, num_samples, *,
                 num_steps=None, eps=1e-5, prefix_tokens=None,
                 prefix_lengths=None):
    if num_steps is None:
      num_steps = model.config.sampler.steps
    V = model.vocab_size
    L = model.num_tokens
    device = model.device

    self._validate_prefix_args(prefix_tokens, prefix_lengths)
    z = torch.randn(
      num_samples, L, V, device=device, dtype=model.dtype)
    prefix_onehot = None
    if prefix_tokens is not None:
      prefix_onehot = F.one_hot(
        prefix_tokens, V).float().to(device)
      start_idx = int(prefix_lengths.min())
      self._project_prefix(z, prefix_onehot, prefix_lengths)
    else:
      start_idx = 0

    t_vals = torch.linspace(0.0, 1.0, num_steps + 1,
                            device=device)
    state = FLMState(
      xt=z, t_vals=t_vals, step_idx=0,
      nfe=0, done=False, start_idx=start_idx,
      prefix_tokens=prefix_onehot,
      prefix_lengths=prefix_lengths)
    return state

  def step(self, model, state):
    num_steps = len(state.t_vals) - 1
    is_last = state.step_idx == num_steps - 1
    B = state.xt.shape[0]

    t_curr = state.t_vals[state.step_idx]
    t_next = state.t_vals[state.step_idx + 1]
    t_in = t_curr.expand(B)

    gamma_t = model._alpha_t_to_gamma(t_in)
    gamma_next = model._alpha_t_to_gamma(t_next.expand(B))
    gamma_d = gamma_next - gamma_t

    log_p = model.forward(
      xt=state.xt, sigma=t_in.unsqueeze(-1),
      context=FLMContext(temperature=self.temperature))
    state.nfe += 1
    if self.use_float64:
      log_p = log_p.to(torch.float64)
    x_1_pred = log_p.exp()

    if is_last:
      state.xt = log_p.argmax(dim=-1)
      state.done = True
    else:
      state.xt = flm_euler_step(
        state.xt, x_1_pred, gamma_t, gamma_d)
      self._project_prefix(
        state.xt, state.prefix_tokens, state.prefix_lengths)

    state.step_idx += 1
    return state


# ── CANDI sampler ─────────────────────────────────────────────

@dataclass(kw_only=True)
class CANDIState(BaseState):
  xt: torch.Tensor           # (B,L,V) continuous or (B,L) tokens
  clean_mask: torch.Tensor   # (B,L)
  timesteps: torch.Tensor    # (num_steps+1,)
  sigmas: torch.Tensor       # (num_steps+1,)
  dt: float
  step_idx: int
  nfe: int
  done: bool
  mode: str                  # 'nocache', 'cached', 'pure_cont'
  embedding_cache: torch.Tensor = None
  denoised: torch.Tensor = None
  # Precomputed prefix data (None when no prefix)
  prefix_mask: torch.Tensor = None       # (B, L) bool
  prefix_onehot: torch.Tensor = None     # (B, P, V) for nocache/pure_cont
  prefix_embeddings: torch.Tensor = None # (B, P, D) for cached mode


class CANDISampler(Sampler):
  """Sampler for CANDI (hybrid continuous-discrete diffusion).

  Supports three modes (auto-detected from the model config):
    - nocache:   alternates continuous + discrete steps
    - cached:    continuous step in embedding space + optimized discrete
    - pure_cont: only continuous denoising steps
  """

  def __init__(self):
    pass

  def init_state(self, model, num_samples, *,
                 num_steps=None, eps=1e-5,
                 prefix_tokens=None,
                 prefix_lengths=None):
    self._validate_prefix_args(prefix_tokens, prefix_lengths)
    if num_steps is None:
      num_steps = model.config.sampler.steps

    x = model.prior_sample(num_samples, model.num_tokens)
    timesteps = torch.linspace(
      0.999, eps, num_steps + 1, device=model.device)
    dt = (1 - eps) / num_steps
    embedding_cache = None

    # Determine mode and sigma schedule
    if model.pure_continuous or model.is_embed:
      mode = 'pure_cont'
      sigmas = model.get_continuous_noise_sched_pure_cont(
        timesteps)
      clean_mask = torch.zeros(
        num_samples, model.num_tokens, device=x.device)
    elif model.candi_sampler == 'cached':
      mode = 'cached'
      if model.use_percentile_scheduling:
        sigmas = model.get_continuous_from_discrete_noise(
          timesteps)
      else:
        sigmas = candi_utils.inference_sigmas(
          num_steps + 1, model.sigma_min,
          model.sigma_max).to(model.device)
      embedding_cache = model.backbone.get_embedding(x)
      clean_mask = torch.zeros(
        num_samples, model.num_tokens,
        device=x.device, dtype=torch.bool)
      x = x.argmax(dim=-1)  # convert to token IDs
    else:
      mode = 'nocache'
      if model.use_percentile_scheduling:
        sigmas = model.get_continuous_from_discrete_noise(
          timesteps)
      else:
        sigmas = candi_utils.inference_sigmas(
          num_steps + 1, model.sigma_min,
          model.sigma_max).to(model.device)
      clean_mask = torch.zeros(
        num_samples, model.num_tokens, device=x.device)

    # Precompute prefix data
    prefix_mask_full = None
    prefix_onehot = None
    prefix_embeddings = None
    if prefix_tokens is not None:
      P = prefix_tokens.shape[1]
      L = model.num_tokens
      positions = torch.arange(L, device=x.device)
      prefix_mask_full = (positions[None, :]
                          < prefix_lengths[:, None])  # (B, L)
      if mode == 'cached':
        prefix_embeddings = model.backbone.get_embedding(
          prefix_tokens)  # (B, P, D)
        x[:, :P] = torch.where(
          prefix_mask_full[:, :P],
          prefix_tokens, x[:, :P])
        embedding_cache[:, :P] = torch.where(
          prefix_mask_full[:, :P].unsqueeze(-1),
          prefix_embeddings, embedding_cache[:, :P])
      else:
        V = x.shape[-1]
        prefix_onehot = F.one_hot(
          prefix_tokens, V).float().to(x.device)  # (B, P, V)
        x[:, :P] = torch.where(
          prefix_mask_full[:, :P].unsqueeze(-1),
          prefix_onehot, x[:, :P])
      clean_mask[prefix_mask_full] = True

    state = CANDIState(
      xt=x, clean_mask=clean_mask,
      timesteps=timesteps, sigmas=sigmas, dt=dt,
      step_idx=0, nfe=0, done=False, mode=mode,
      embedding_cache=embedding_cache,
      prefix_tokens=prefix_tokens,
      prefix_lengths=prefix_lengths,
      prefix_mask=prefix_mask_full,
      prefix_onehot=prefix_onehot,
      prefix_embeddings=prefix_embeddings)
    return state

  def step(self, model, state):
    i = state.step_idx
    num_steps = len(state.timesteps) - 1
    t = state.timesteps[i]
    sigma_s = state.sigmas[i]
    sigma_t = state.sigmas[i + 1]

    if state.mode == 'nocache':
      x_cont, p_x0 = model._continuous_step(
        state.xt, t, sigma_s=sigma_s, sigma_t=sigma_t,
        clean_mask=state.clean_mask)
      state.xt, state.clean_mask = model._discrete_step(
        x_cont, p_x0, t, state.dt,
        prev_clean_mask=state.clean_mask)

    elif state.mode == 'cached':
      state.embedding_cache, x0_hat = (
        model._continuous_step_cache(
          state.xt, t, sigma_s=sigma_s, sigma_t=sigma_t,
          clean_mask=state.clean_mask.float(),
          embedding_cache=state.embedding_cache))
      state.xt, state.clean_mask = (
        model._discrete_step_optimized(
          x0_hat, state.xt, t, state.dt,
          prev_clean_mask=state.clean_mask))

    elif state.mode == 'pure_cont':
      state.xt, state.denoised = model._continuous_step(
        state.xt, t, sigma_s=sigma_s, sigma_t=sigma_t,
        clean_mask=state.clean_mask,
        is_embed=model.is_embed)

    # Restore prefix after each step
    if state.prefix_mask is not None:
      P = state.prefix_tokens.shape[1]
      m = state.prefix_mask[:, :P]
      if state.mode == 'cached':
        state.xt[:, :P] = torch.where(
          m, state.prefix_tokens, state.xt[:, :P])
        state.embedding_cache[:, :P] = torch.where(
          m.unsqueeze(-1),
          state.prefix_embeddings,
          state.embedding_cache[:, :P])
      else:
        self._project_prefix(
          state.xt, state.prefix_onehot,
          state.prefix_lengths)
      state.clean_mask[state.prefix_mask] = True

    state.nfe += 1
    state.step_idx += 1
    if state.step_idx >= num_steps:
      # Convert to token IDs
      if state.mode == 'pure_cont' and model.is_embed:
        state.xt = state.denoised.argmax(dim=-1)
      elif state.mode != 'cached':
        # nocache and pure_cont (non-embed) have continuous xt
        state.xt = state.xt.argmax(dim=-1)
      # cached mode: xt is already token IDs
      state.done = True
    return state


def run_sampler(sampler, model, num_samples, *,
               num_steps=None, eps=1e-5, prefix_tokens=None,
               prefix_lengths=None):
  state = sampler.init_state(model, num_samples,
                             num_steps=num_steps, eps=eps,
                             prefix_tokens=prefix_tokens,
                             prefix_lengths=prefix_lengths)
  while not state.done:
    state = sampler.step(model, state)
  return state.xt, sampler.metadata(state)


def get_sampler(config):
  s = config.sampler

  if s.predictor == 'ancestral':
    return AncestralSampler(
      use_float64=s.use_float64,
      noise_removal=s.noise_removal,
      steps_policy=s.steps_policy,
      temperature=s.temperature)

  if s.predictor == 'ar':
    return ARSampler(
      use_float64=s.use_float64,
      kv_cache=s.use_kv_cache,
      greedy=s.greedy,
      early_stopping=s.early_stopping,
      temperature=s.temperature)

  if s.predictor == 'sfm':
    return SFMSampler(noise_removal=s.noise_removal,
      velocity=s.velocity, use_float64=s.use_float64,
      slerp_float64=config.algo.slerp_precision=='float64',
      eps=config.algo.eps, temperature=s.temperature,
      p_nucleus=s.p_nucleus, top_k=s.top_k,
      top_k_velocity=s.top_k_velocity,
      invert_time_convention=config.algo.invert_time_convention,
      bct_mode=s.get('bct', {}).get('mode', 'none'),
      bct_T=s.get('bct', {}).get('T', 1.0),
      bct_correction_strength=s.get('bct', {}).get(
        'correction_strength', 1.0))

  if s.predictor == 'flm_euler':
    return FLMEulerSampler(
      use_float64=s.use_float64,
      temperature=s.temperature)

  if s.predictor == 'candi':
    return CANDISampler()

  raise ValueError(f'Unknown sampler predictor: {s.predictor}')
