# Modified from S-FLM (https://github.com/jdeschena/s-flm) for BCT: tempered-flow training (TFT) and checkpoint adaptation in `SFM`.
import math

import torch
import collections
from dataclass_patch import dataclass

import samplers
import trainer_base
from trainer_base import FLMTrainingContext, CANDITrainingContext
import torch.nn.functional as F
import utils
import flm_utils
import candi_utils
import models.flm_dit


class AR(trainer_base.TrainerBase):
  def __init__(self, config, tokenizer):
    vocab_size = tokenizer.vocab_size
    if (not hasattr(tokenizer, 'mask_token')
        or tokenizer.mask_token is None):
      self.mask_index = vocab_size
      vocab_size += 1
    else:
      self.mask_index = tokenizer.mask_token_id
    super().__init__(config, tokenizer,
                     vocab_size=vocab_size)
    self.save_hyperparameters()
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert not self.config.algo.time_conditioning
    assert self.config.prior.type == 'none'

  def _process_model_input(self, x0, valid_tokens):
    input_tokens = x0[:, :-1]
    output_tokens = x0[:, 1:]
    valid_tokens = valid_tokens[:, 1:]
    return input_tokens, output_tokens, valid_tokens

  def _process_model_output(
      self, model_output, xt, sigma, context=None):
    del xt, sigma, context
    model_output[:, :, self.mask_index] = self.neg_infinity
    return torch.log_softmax(model_output, dim=-1)

  def _process_sigma(self, sigma, context=None):
    return None

  def nll(self, input_tokens, output_tokens, context,
          current_accumulation_step, train_mode,
          valid_tokens=None):
    del train_mode, current_accumulation_step, valid_tokens, context

    x0 = input_tokens
    output = self.forward(xt=x0, sigma=None)
    per_token_nll =  - output.gather(
      -1, output_tokens[:, :, None])[:, :, 0]
    return per_token_nll, None


class MDLM(trainer_base.AbsorbingState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self.post_process_mode = config.algo.post_process_mode
    self._validate_configuration()

  def _process_model_output_log_probs(
      self, model_output, xt, sigma):
    del sigma
    model_output[:, :, self.mask_index] += self.neg_infinity

    # Normalize the model_output such that x.exp() is
    # a probability distribution over vocab_size.
    model_output = model_output - torch.logsumexp(
      model_output, dim=-1, keepdim=True)
    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    model_output[unmasked_indices] = self.neg_infinity
    model_output[unmasked_indices, xt[unmasked_indices]] = 0
    return model_output

  def _process_model_output_logits(self, model_output, xt, sigma):
    del sigma
    model_output[..., self.mask_index] = self.neg_infinity
    xt_unsq = xt.unsqueeze(-1)
    unmasked = (xt != self.mask_index)[..., None]
    deterministic = model_output.new_full(
      model_output.shape, self.neg_infinity)
    deterministic.scatter_(-1, xt_unsq, 0.0)
    # For masked positions, return raw logits (unnormalized).
    return torch.where(unmasked, deterministic, model_output)

  def _validate_configuration(self):
    super()._validate_configuration()
    if self.post_process_mode not in {
      'log_probs', 'logits'}:
      raise ValueError(self.post_process_mode)

  def _process_model_output(
      self, model_output, xt, sigma, context=None):
    del context
    if self.post_process_mode == 'log_probs':
      return self._process_model_output_log_probs(
        model_output, xt, sigma)
    elif self.post_process_mode == 'logits':
      return self._process_model_output_logits(
        model_output, xt, sigma)
    else:
      raise ValueError(self.post_process_mode)

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False, context=None,
                    train_mode=False):
    del context, xt, train_mode
    log_p_theta = torch.gather(
      input=log_x_theta,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    if low_var:
      loss_coefficient = -1
    else:
      loss_coefficient = dalpha_t / (1 - alpha_t)
    return loss_coefficient * log_p_theta

  def nll(self, x0, output_tokens, context,
          current_accumulation_step=None, train_mode=False,
          valid_tokens=None):
    if self.post_process_mode == 'logits':
      raise ValueError(
        'post_process_mode=logits is sampling-only. '
        'Use log_probs mode for training or evaluation.')
    return super().nll(
      x0=x0,
      output_tokens=output_tokens,
      context=context,
      current_accumulation_step=current_accumulation_step,
      train_mode=train_mode,
      valid_tokens=valid_tokens)


class DUO_BASE(trainer_base.UniformState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def on_save_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher'))
    super().on_save_checkpoint(checkpoint)

  def on_load_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher'))
    super().on_load_checkpoint(checkpoint)

  def _process_model_output(
      self, model_output, xt, sigma, context=None):
    del xt, sigma, context
    return model_output.log_softmax(dim=-1)

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False, context=None,
                    train_mode=False):
    del context, train_mode
    assert alpha_t.ndim == 2
    assert x0.ndim == 2
    assert xt.ndim == 2
    assert not torch.is_tensor(dalpha_t) or dalpha_t.ndim == 2
    x_reconst = log_x_theta.exp()
    x_bar_theta = self.vocab_size * alpha_t[
        :, :, None] * x_reconst + 1 - alpha_t[:, :, None]
    coeff = dalpha_t / (self.vocab_size * alpha_t)
    x_eq_xt = (x0 == xt).float()
    x_neq_xt = 1 - x_eq_xt
    xbar_xt = (1 - alpha_t) + self.vocab_size * alpha_t * x_eq_xt
    xbar_theta_xt = torch.gather(
      x_bar_theta, -1, xt.unsqueeze(-1)).squeeze(-1)
    xbar_theta_x = torch.gather(
      x_bar_theta, -1, x0.unsqueeze(-1)).squeeze(-1)
    if low_var:
      term1 = 0
    else:
      term1 = self.vocab_size * (1 / xbar_xt
                                 - 1 / xbar_theta_xt)

    const = (1 - alpha_t) / (self.vocab_size * alpha_t
                             + 1 - alpha_t)
    term2_coefs = x_eq_xt * const + x_neq_xt
    term2_offset = ((self.vocab_size - 1) * const * x_eq_xt
                    - (1 / const) * x_neq_xt) * const.log()
    term2_theta = - term2_coefs * (
      x_bar_theta.log().sum(-1)
      - self.vocab_size * xbar_theta_xt.log())
    term2_theta = (
      term2_theta
      - self.vocab_size * alpha_t / (1 - alpha_t) * (
        xbar_theta_x.log() - xbar_theta_xt.log()) * x_neq_xt)
    term2 = term2_theta + term2_offset
    diffusion_loss = coeff * (term1 - term2)
    assert diffusion_loss.ndim == 2
    return diffusion_loss


class SFM(trainer_base.Diffusion):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self.eps = config.algo.eps
    self.renormalize_weights = config.algo.renormalize_weights
    self.invert_time_convention = config.algo.invert_time_convention
    # Tempered-flow training (TFT); `none` is upstream S-FLM training.
    self.bct_train_T_dist = str(config.algo.get('bct_train_T_dist', 'none'))
    self.bct_train_p_one = float(config.algo.get('bct_train_p_one', 0.0))
    self.bct_train_p_low = float(config.algo.get('bct_train_p_low', 0.0))
    self.bct_train_T_low = float(config.algo.get('bct_train_T_low', 0.1))
    self.bct_train_beta_min = float(
      config.algo.get('bct_train_beta_min', 1.25))
    self.bct_train_beta_max = float(
      config.algo.get('bct_train_beta_max', 5.0))
    # Whether the model is conditioned on T is a property of the backbone.
    self.bct_temp_conditioning = bool(
      config.model.get('temp_conditioning', False))
    self._validate_configuration()

  def _validate_configuration(self):
    if self.invert_time_convention and self.config.noise.adaptive:
      raise ValueError('Adaptive noise schedule requires '
                       'invert_time_convention=false '
                       '(MDLM-like convention).')
    backbone_type = self.config.model.type
    if backbone_type == 'sphere-arch' and not self.renormalize_weights:
      raise ValueError('Backbone sphere-arch requires '
                       'algo.renormalize_weights=True.')
    if self.bct_train_T_dist not in ('none', 'mixture'):
      raise ValueError("algo.bct_train_T_dist must be 'none' or 'mixture', "
                       f'got {self.bct_train_T_dist}.')
    if self.bct_train_T_dist == 'mixture':
      if not (0.0 <= self.bct_train_p_one <= 1.0
              and 0.0 <= self.bct_train_p_low <= 1.0
              and self.bct_train_p_one + self.bct_train_p_low <= 1.0):
        raise ValueError(
          'algo.bct_train_p_one + algo.bct_train_p_low must lie in [0, 1] '
          f'(got {self.bct_train_p_one} + {self.bct_train_p_low}).')
      if not 1.0 <= self.bct_train_beta_min <= self.bct_train_beta_max:
        raise ValueError(
          'algo.bct_train_beta_min/max must satisfy '
          f'1 <= min <= max (got {self.bct_train_beta_min}, '
          f'{self.bct_train_beta_max}).')
      if self.bct_train_T_low <= 0:
        raise ValueError('algo.bct_train_T_low must be > 0.')
    if (self.bct_temp_conditioning
        and self.config.model.type not in ('sphere-arch', 'sphere-dit')):
      raise ValueError(
        'model.temp_conditioning is only implemented for '
        'model.type=sphere-arch / sphere-dit '
        f'(got {self.config.model.type}).')

  def _process_model_output(self, model_output, xt, sigma,
                            context=None):
    return model_output.float().log_softmax(-1)

  def _sample_prior(self, e_clean):
    e_noisy = torch.randn_like(e_clean)
    return utils.sphere_normalize(e_noisy)

  def q_xt(self, x, alpha_t, use_pure_noise, valid_tokens=None,
           bct_T=None):
    e_clean = self.backbone.get_sphere_embeddings(x)  # [B, L, d]
    e_noisy = self._sample_prior(e_clean)

    if use_pure_noise:
      x_t = e_noisy
    else:
      slerp_t = alpha_t if self.invert_time_convention else 1 - alpha_t
      if bct_T is not None:
        # TFT: one temperature per sequence.
        x_t = self._slerp_tempered(e_clean, e_noisy, slerp_t, bct_T)
      else:
        x_t = self._slerp(e_clean, e_noisy, slerp_t)

    if valid_tokens is not None:
      x_t = torch.where(valid_tokens.bool().unsqueeze(-1),
                        x_t, e_clean)
    return x_t

  def optimizer_step(self, *args, **kwargs):
    out = super().optimizer_step(*args, **kwargs)
    if self.renormalize_weights:
      self.backbone.renormalize_weights()
    return out

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t,
                    low_var=False, context=None, train_mode=False):
    del xt, alpha_t, dalpha_t, low_var, context

    ce_loss = -log_x_theta.gather(
      -1, x0.unsqueeze(-1)).squeeze(-1)

    return ce_loss

  def _slerp(self, clean, noisy, alpha_t):
    # alpha_t = 0 -> clean, alpha_t = 1 -> noisy
    orig_dtype = None
    if self.config.algo.slerp_precision == 'float64':
      orig_dtype = clean.dtype
      alpha_t = alpha_t.to(torch.float64)
      clean = clean.to(torch.float64)
      noisy = noisy.to(torch.float64)
    out = utils.slerp(
      clean=clean,
      noisy=noisy,
      alpha_t=alpha_t,
      eps=self.eps)

    if orig_dtype is not None:
      out = out.to(orig_dtype)
    return out

  def _slerp_tempered(self, clean, noisy, a, T):
    """Draws z ~ q_{a,T_i}(.|x_i) exactly by quantile transport, with a
    per-sequence temperature; `utils.slerp` at T = 1."""
    orig_dtype = None
    if self.config.algo.slerp_precision == 'float64':
      orig_dtype = clean.dtype
      a = a.to(torch.float64)
      clean = clean.to(torch.float64)
      noisy = noisy.to(torch.float64)
    a = a.reshape(a.shape[0], *([1] * (clean.ndim - 1)))
    cos_omega = (noisy * clean).sum(dim=-1, keepdim=True)
    cos_omega = cos_omega.clamp(-1 + self.eps, 1 - self.eps)
    omega = torch.acos(cos_omega)
    psi = omega / math.pi
    d = clean.shape[-1]
    phi = samplers.bct_corruption_transport(
      psi, a.reshape(-1), T.reshape(-1).to(psi.dtype), d)
    alpha_eff = a * (phi / psi.clamp(min=1e-300))
    sin_omega = omega.sin().clamp(min=self.eps)
    coeff_clean = ((1 - alpha_eff) * omega).sin() / sin_omega
    coeff_noisy = (alpha_eff * omega).sin() / sin_omega
    out = coeff_clean * clean + coeff_noisy * noisy
    if orig_dtype is not None:
      out = out.to(orig_dtype)
    return out

  def _bct_sample_train_T(self, n, device):
    """The TFT training temperature, one per sequence (None for `none`).

    `mixture`: an atom at T = 1, an atom at `bct_train_T_low`, and otherwise
    beta = 1/T uniform.
    """
    if self.bct_train_T_dist == 'none':
      return None
    u = torch.rand(n, device=device, dtype=torch.float64)
    beta = (self.bct_train_beta_min
            + (self.bct_train_beta_max - self.bct_train_beta_min) * u)
    T = 1.0 / beta
    r = torch.rand(n, device=device, dtype=torch.float64)
    p1 = self.bct_train_p_one
    plow = self.bct_train_p_low
    T = torch.where(r < p1, torch.ones_like(T), T)
    if plow > 0.0:
      T = torch.where((r >= p1) & (r < p1 + plow),
                      torch.full_like(T, self.bct_train_T_low), T)
    return T

  # Resuming a released checkpoint with the temperature-conditioning
  # parameters added: the checkpoint is re-keyed by parameter name, so that the
  # existing parameters keep their EMA and optimizer state.
  def _bct_param_names(self):
    """Parameter names in `_get_parameters()` order (as seen by EMA/AdamW)."""
    names = [f'backbone.{n}' for n, _ in self.backbone.named_parameters()]
    names += [f'noise.{n}' for n, _ in self.noise.named_parameters()]
    n_grad = sum(1 for p in self._get_parameters() if p.requires_grad)
    if n_grad != len(names):
      raise RuntimeError(
        'Checkpoint adaptation assumes every optimised parameter '
        f'requires grad ({n_grad} vs {len(names)}); it does not.')
    return names

  def on_load_checkpoint(self, checkpoint):
    self._bct_extend_checkpoint(checkpoint)
    super().on_load_checkpoint(checkpoint)

  def _bct_extend_checkpoint(self, checkpoint):
    sd = checkpoint.get('state_dict')
    if sd is None:
      return
    has_temp_in_ckpt = any('temp_modulation.' in k for k in sd)
    if has_temp_in_ckpt and not self.bct_temp_conditioning:
      raise RuntimeError(
        'Checkpoint carries temperature-conditioning parameters but '
        'model.temp_conditioning is False; the load would silently drop them.')
    names = self._bct_param_names()
    missing = [n for n in names if n not in sd]
    if not missing:
      return
    if self.config.mode != 'train':
      raise RuntimeError(
        'model.temp_conditioning is True but the checkpoint has no '
        'temperature-conditioning parameters; refusing to evaluate a freshly '
        f'initialised conditioning (missing e.g. {missing[:3]}).')
    allowed = ('temp_map.', 'temp_modulation.')
    unexpected = [n for n in missing if not any(k in n for k in allowed)]
    if unexpected:
      raise RuntimeError(
        'Checkpoint is missing parameters that are not temperature-'
        f'conditioning ones; refusing to invent state for them: '
        f'{unexpected[:8]}')
    cur = dict(self.named_parameters())
    old_names = [n for n in names if n not in missing]
    print(f'[SFM.on_load_checkpoint] adapting checkpoint: '
          f'{len(old_names)} existing parameters keep their optimizer / EMA '
          f'state, {len(missing)} new temperature-conditioning parameters '
          f'get fresh state.')

    for n in missing:
      sd[n] = cur[n].detach().clone().cpu()

    ema = checkpoint.get('ema')
    if ema is not None:
      shadow = ema['shadow_params']
      if len(shadow) != len(old_names):
        raise RuntimeError(
          f'EMA shadow has {len(shadow)} entries but the checkpoint has '
          f'{len(old_names)} parameters; cannot align by name.')
      by_name = dict(zip(old_names, shadow))
      ema['shadow_params'] = [
        by_name[n] if n in by_name else cur[n].detach().clone().cpu()
        for n in names]

    for opt in checkpoint.get('optimizer_states', []):
      groups = opt['param_groups']
      if len(groups) != 1:
        raise RuntimeError(
          f'Expected one param group (configure_optimizers builds one), got '
          f'{len(groups)}.')
      old_ids = list(groups[0]['params'])
      if len(old_ids) != len(old_names):
        raise RuntimeError(
          f'Optimizer group has {len(old_ids)} params but the checkpoint has '
          f'{len(old_names)}; cannot align by name.')
      id_to_name = dict(zip(old_ids, old_names))
      new_index = {n: i for i, n in enumerate(names)}
      opt['state'] = {new_index[id_to_name[i]]: st
                      for i, st in opt['state'].items()}
      groups[0]['params'] = list(range(len(names)))

  def _bct_train_tau(self, T, n, device):
    """tau = -log T, the value the model is conditioned on (0 when T is None)."""
    if T is None:
      return torch.zeros(n, device=device, dtype=torch.float64)
    return -torch.log(T.reshape(-1).to(torch.float64))

  def nll(self, x0, output_tokens, context,
          current_accumulation_step=None, train_mode=False,
          valid_tokens=None):
    del output_tokens
    t = self._sample_t(x0.shape[0], current_accumulation_step)
    assert t.shape[0] == x0.shape[0]
    use_pure_noise = self._use_pure_noise(
      train_mode=train_mode, context=context)
    if use_pure_noise:
      t = torch.ones_like(t)

    dalpha_t, alpha_t = self.noise(t)
    alpha_t = alpha_t.unsqueeze(-1)
    dalpha_t = dalpha_t.unsqueeze(-1)

    # TFT: `bct_T` is None unless `algo.bct_train_T_dist=mixture`.
    bct_T = self._bct_sample_train_T(x0.shape[0], x0.device)
    xt = self.q_xt(
      x0, alpha_t, use_pure_noise=use_pure_noise,
      valid_tokens=valid_tokens, bct_T=bct_T)
    if self.bct_temp_conditioning and context is not None:
      context.bct_tau = self._bct_train_tau(
        bct_T, x0.shape[0], x0.device)

    sigma = self._sigma_from_alphat(alpha_t)
    log_x_theta = self.forward(
      x0=x0, xt=xt, sigma=sigma, context=context)
    utils.print_nans(log_x_theta, 'model_output')

    loss = self.nll_per_token(
      log_x_theta=log_x_theta, xt=xt, x0=x0,
      alpha_t=alpha_t, dalpha_t=dalpha_t,
      low_var=train_mode and self.loss_type == 'low_var',
      context=context, train_mode=train_mode)

    return loss, t


# ════════════════════════════════════════════════════════════════
#  FLM (Flow Language Model)
#  https://arxiv.org/abs/2602.16813
# ════════════════════════════════════════════════════════════════
class FLMBase(trainer_base.Diffusion):
  """Base class for Flow Language Model algorithms.

  NOTE: self.noise is inherited from TrainerBase but UNUSED by FLM.
  FLM uses LUT-based gamma/alpha mapping that is a fixed function
  of vocab size K, independent of the noise schedule.
  """

  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self.t_min = config.algo.t_min
    self.t_max = config.algo.t_max
    self.ignore_bos = getattr(config.algo, 'ignore_bos', False)
    # Output-logit softcap (upstream FLM uses 30.0). None ↔ disabled —
    # default for backward-compat with checkpoints trained without it.
    self.cap_value = getattr(config.algo, 'cap_value', None)
    self.lut_a2g, self.lut_g2a = flm_utils.build_luts(K=self.vocab_size)

  def _validate_configuration(self):
    pass  # FLM has no configuration constraints from parent

  def _process_model_output(self, model_output, xt, sigma,
                            context=None):
    del xt, sigma
    if (context is not None
        and getattr(context, 'skip_softmax', False)):
      return model_output
    if self.cap_value is not None:
      model_output = self.cap_value * torch.tanh(
        model_output / self.cap_value)
    return model_output.log_softmax(dim=-1)

  def nll(self, input_tokens, output_tokens, context,
          current_accumulation_step=None, train_mode=False,
          valid_tokens=None):
    # Note: technically, not an NLL, just implemented that way
    #  to plug-in the trainer_base API.
    raise NotImplementedError

  def _sample_t_interval(self, n, accum_step,
                         t_min=None, t_max=None):
    if t_min is None:
      t_min = self.t_min
    if t_max is None:
      t_max = self.t_max
    if accum_step is not None:
      batch_dim = n
      n = self.config.loader.global_batch_size
    _eps_t = torch.rand(n, device=self.device)
    if self.antithetic_sampling:
      offset = torch.arange(n, device=self.device) / n
      _eps_t = (_eps_t / n + offset) % 1
      perm = torch.randperm(n, device=self.device)
      _eps_t = _eps_t[perm]
    t = (t_max - t_min) * _eps_t + t_min
    if accum_step is not None:
      t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
      t = t.chunk(
        self.trainer.num_devices)[self.trainer.local_rank]
      t = t.chunk(
        self.trainer.accumulate_grad_batches)[accum_step]
      t = t[:batch_dim]
    return t

  def _alpha_t_to_gamma(self, alpha_t):
    return flm_utils.alpha_to_gamma(alpha_t, self.lut_a2g)

  def _gamma_to_alphat(self, gamma_t):
    return flm_utils.gamma_to_alpha(gamma_t, self.lut_g2a)

  def corrupt_continuous(self, x0, t, valid_tokens=None):
    """Corrupt data x0 at time t via linear interpolation.

    Prompt positions (valid_tokens=0) stay clean.
    """
    t = t.unsqueeze(-1).unsqueeze(-1)
    target_data = F.one_hot(x0, self.vocab_size).float()
    noise = torch.randn_like(target_data, dtype=torch.float32)
    x_t = (1 - t) * noise + t * target_data
    if valid_tokens is not None:
      mask = valid_tokens.bool().unsqueeze(-1)
      x_t = torch.where(mask, x_t, target_data)
    return x_t, target_data

  def forward_no_softmax(self, xt, t, t_prime=None):
    """Forward through backbone WITHOUT log-softmax.

    Used by distillation for the residual network output.
    """
    ctx = FLMTrainingContext(skip_softmax=True)
    if t_prime is not None:
      sigma_prime = self._process_sigma(t_prime.unsqueeze(-1))
      ctx = FLMTrainingContext(
        sigma_prime=sigma_prime, skip_softmax=True)
    return self.forward(xt=xt, sigma=t.unsqueeze(-1), context=ctx)

  def teacher_forward(self, teacher_model, xt, t):
    """Forward through a teacher model (returns log-probs)."""
    sigma = self._process_sigma(t.unsqueeze(-1))
    with torch.no_grad():
      with torch.amp.autocast('cuda', dtype=torch.float32):
        model_output = teacher_model(None, xt, sigma, None)
    return model_output.log_softmax(dim=-1)

  def teacher_forward_no_softmax(self, teacher_model, xt, t,
                                 t_prime=None):
    """Forward through teacher WITHOUT log-softmax."""
    sigma = self._process_sigma(t.unsqueeze(-1))
    sigma_prime = None
    if t_prime is not None:
      sigma_prime = self._process_sigma(t_prime.unsqueeze(-1))
    ctx = FLMTrainingContext(sigma_prime=sigma_prime)
    with torch.no_grad():
      with torch.amp.autocast('cuda', dtype=torch.float32):
        model_output = teacher_model(None, xt, sigma, ctx)
    return model_output

  # ── checkpoint ────────────────────────────────────────────

  def load_state_dict(self, state_dict, strict=True):
    return super().load_state_dict(state_dict, strict=False)

  def on_save_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher'))
    super().on_save_checkpoint(checkpoint)

  def _filter_checkpoint_state_dict(self, state_dict):
    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
      if k.startswith('teacher'):
        continue
      new_key = k.replace('._orig_mod.', '.')
      new_state_dict[new_key] = v
    return new_state_dict

  # ── teacher model utilities ───────────────────────────────

  def _extract_ema_state_dict(self, model, checkpoint):
    """Extract EMA parameters from checkpoint."""
    ema_state = checkpoint.get('ema', None)
    if not ema_state:
      print("Warning: No EMA found, using regular state_dict")
      return {
        k.replace('backbone.', '').replace('._orig_mod.', ''): v
        for k, v in checkpoint['state_dict'].items()
        if k.startswith('backbone.')}

    new_sd = collections.OrderedDict()
    shadow_params = ema_state['shadow_params']
    param_names = [n for n, p in model.named_parameters()
                   if p.requires_grad]
    min_len = min(len(shadow_params), len(param_names))
    for name, val in zip(param_names[:min_len],
                         shadow_params[:min_len]):
      new_sd[name] = val
    for k, v in checkpoint['state_dict'].items():
      clean_k = (k.replace('backbone.', '')
                  .replace('._orig_mod.', ''))
      if (clean_k not in new_sd
          and clean_k in dict(model.named_parameters())):
        new_sd[clean_k] = v
    return new_sd

  def _load_teacher_model(self, path, use_plain_config=True):
    """Load a frozen teacher model from checkpoint.

    Uses FLMDIT (not the standard DIT) so that the teacher's
    EmbeddingLayer does NOT apply softmax to continuous input.
    """
    print(f"Loading teacher model from: {path}")
    if use_plain_config:
      saved = (self.config.algo.double_temb,
               self.config.algo.learnable_loss_weighting)
      self.config.algo.double_temb = False
      self.config.algo.learnable_loss_weighting = False

    model = models.flm_dit.FLMDIT(
      self.config, vocab_size=self.vocab_size)

    if use_plain_config:
      (self.config.algo.double_temb,
       self.config.algo.learnable_loss_weighting) = saved

    checkpoint = torch.load(
      path, map_location='cpu', weights_only=False)
    state_dict = self._extract_ema_state_dict(model, checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(self.device).eval()
    for param in model.parameters():
      param.requires_grad = False
    return model

  def _copy_teacher_weights_to_student(self, teacher_dict):
    """Copy teacher weights to student, zero-init sigma_map_prime."""
    with torch.no_grad():
      student_dict = self.backbone.state_dict()
      for name, param in teacher_dict.items():
        if name in student_dict:
          student_dict[name].copy_(param)
      if (hasattr(self.backbone, 'sigma_map_prime')
          and self.backbone.sigma_map_prime is not None):
        for name, param in (
            self.backbone.sigma_map_prime.named_parameters()):
          if 'mlp.2' in name:
            param.zero_()

  @staticmethod
  def _zero_init_module(module):
    for m in module.modules():
      if isinstance(m, torch.nn.Linear):
        m.weight.data.zero_()
        if m.bias is not None:
          m.bias.data.zero_()


class FLM(FLMBase):
  """Flow Language Model — continuous-time flow matching."""

  def on_load_checkpoint(self, checkpoint):
    if 'state_dict' in checkpoint:
      checkpoint['state_dict'] = (
        self._filter_checkpoint_state_dict(
          checkpoint['state_dict']))
    super().on_load_checkpoint(checkpoint)

  def nll(self, x0, output_tokens, context,
           current_accumulation_step=None, train_mode=False,
           valid_tokens=None):
    del output_tokens
    B = x0.shape[0]
    t = self._sample_t_interval(
      B, current_accumulation_step,
      t_min=self.t_min, t_max=self.t_max)
    c_t = self._alpha_t_to_gamma(t)
    x_t, target_data = self.corrupt_continuous(
      x0, c_t, valid_tokens=valid_tokens)
    f = self.forward(xt=x_t, sigma=t.unsqueeze(-1))
    tfm_loss = -(target_data * f).sum(dim=-1)

    if self.config.algo.learnable_loss_weighting:
      loss_weight = self.backbone.learnable_loss_weighting(t)
      loss_weight = loss_weight.unsqueeze(-1)
      tfm_loss = (torch.exp(-loss_weight) * tfm_loss
                  + loss_weight)
    return tfm_loss, None


# ──────────────────────────────────────────────────────────────────
# CANDI: Continuous and Discrete Diffusion
# ──────────────────────────────────────────────────────────────────

class CANDI(trainer_base.Diffusion):
  """Hybrid continuous-discrete diffusion (CANDI).

  Supports three modes via config flags:
    - Hybrid    (pure_continuous=False, is_embed=False)
    - One-hot   (pure_continuous=True,  is_embed=False)
    - Embedding (pure_continuous=True,  is_embed=True)
  """

  def __init__(self, config, tokenizer):
    # Mask token at end of vocab; total vocab = len(tokenizer) + 1
    self.mask_index = len(tokenizer)
    vocab_size = len(tokenizer) + 1
    super().__init__(config, tokenizer, vocab_size=vocab_size)
    self.save_hyperparameters()

    # Config flags
    self.pure_continuous = config.algo.pure_continuous
    self.is_embed = config.algo.is_embed
    self.candi_sampler = config.algo.sampler
    self.step_size = config.algo.step_size
    self.temp = config.algo.temp
    self.sigma_min = config.algo.sigma_min
    self.sigma_max = config.algo.sigma_max
    self.min_percentile = config.algo.min_percentile
    self.max_percentile = config.algo.max_percentile
    self.use_percentile_scheduling = (
      config.algo.use_percentile_scheduling)
    self.ignore_bos = getattr(config.algo, 'ignore_bos', False)

    # Noise-mapping lookup tables (buffers auto-transfer device)
    cont_noise = torch.linspace(0.2, 4.0, 1000)
    disc_noise = candi_utils.expected_rank(
      d=self.vocab_size - 1,
      a=torch.tensor(1.0),
      sigma=cont_noise)
    self.register_buffer('cont_noise_table', cont_noise)
    self.register_buffer('disc_noise_table', disc_noise)

  def _validate_configuration(self):
    assert self.time_conditioning, (
      'CANDI requires time_conditioning=True')

  # ── Post-processing ────────────────────────────────────────────

  def _process_model_output(self, model_output, xt, sigma,
                            context=None):
    reveal_mask = context.reveal_mask if context else None
    # For revealed positions we need original token IDs.
    # In embed mode xt.argmax(-1) is over the embedding dim
    # (wrong), so prefer x0_tokens from context when available.
    if context and context.x0_tokens is not None:
      xt_tokens = context.x0_tokens
    elif xt.ndim == 2:
      xt_tokens = xt
    else:
      xt_tokens = xt.argmax(dim=-1)
    # Temperature + log-softmax
    model_output = model_output / self.temp
    model_output = (model_output
                    - torch.logsumexp(
                        model_output, dim=-1, keepdim=True))
    # Force revealed positions: log_prob=0 at true token
    if reveal_mask is not None:
      rm = reveal_mask.bool()
      model_output[rm] = self.neg_infinity
      model_output[rm, xt_tokens[rm]] = 0
    return model_output

  # ── Noise helpers ──────────────────────────────────────────────

  def get_continuous_from_discrete_noise(self, discrete_noise):
    """Map discrete error rate -> continuous sigma."""
    if self.is_embed:
      return candi_utils.training_sigma_ve(
        discrete_noise, self.sigma_min, self.sigma_max)
    target = (discrete_noise
              * (self.max_percentile - self.min_percentile)
              + self.min_percentile)
    return candi_utils.sigma_from_time_vectorized(
      target, self.cont_noise_table, self.disc_noise_table)

  def get_continuous_noise_sched_pure_cont(self, timesteps):
    """Noise schedule for pure-continuous inference."""
    target = (timesteps
              * (self.max_percentile - self.min_percentile)
              + self.min_percentile)
    return candi_utils.sigma_from_time_vectorized(
      target, self.cont_noise_table, self.disc_noise_table)

  # ── Forward corruption ─────────────────────────────────────────

  def discrete_noising(self, x, alpha_t):
    """Replace tokens with uniform random, prob = 1 - alpha_t."""
    move = torch.rand(*x.shape, device=x.device) < 1 - alpha_t
    uniform = torch.randint(
      0, self.vocab_size, x.shape, device=x.device)
    xt = torch.where(move, uniform, x)
    if self.ignore_bos:
      xt[:, 0] = x[:, 0]
    return xt

  def q_xt(self, x, alpha_t, use_pure_noise=False,
           valid_tokens=None):
    """Hybrid corruption: discrete + continuous noise.

    Args:
      x: (B, L) int token IDs.
      alpha_t: (B, 1) signal level.
      valid_tokens: (B, L) optional. 1=generate (noise this),
          0=prompt (keep clean).

    Returns a dict with keys: xt, reveal_mask,
    discrete_noise, continuous_noise, (is_embed).
    Prompt positions (valid_tokens=0) get no noise and
    reveal_mask=1 (the model sees them as clean/revealed).
    """
    del use_pure_noise  # Not used by CANDI
    disc_xt = self.discrete_noising(x, alpha_t)
    reveal_mask = (disc_xt == x).float()
    discrete_noise = (1 - alpha_t).squeeze()
    continuous_noise = self.get_continuous_from_discrete_noise(
      discrete_noise)
    onehot = F.one_hot(
      x, num_classes=self.vocab_size - 1).float()

    if self.pure_continuous:
      if self.is_embed:
        clean_emb = self.backbone.get_embedding(x)
        noise = (torch.randn_like(clean_emb)
                 * continuous_noise[:, None, None])
        xt = clean_emb + noise
        rm = torch.zeros_like(reveal_mask)
      else:
        noise = (torch.randn_like(onehot)
                 * continuous_noise[:, None, None])
        xt = onehot + noise
        rm = torch.zeros_like(reveal_mask)
      # Keep prompt positions clean
      if valid_tokens is not None:
        prompt = ~valid_tokens.bool()
        clean = (clean_emb if self.is_embed
                 else onehot)
        xt = torch.where(
          prompt.unsqueeze(-1), clean, xt)
        rm = torch.where(prompt, torch.ones_like(rm), rm)
      return {
        'xt': xt, 'reveal_mask': rm,
        'discrete_noise': discrete_noise,
        'continuous_noise': continuous_noise,
        'is_embed': self.is_embed}

    # Hybrid: discrete + continuous
    gauss = (continuous_noise[:, None, None]
             * torch.randn_like(onehot))
    xt_cont = onehot + gauss
    # Keep prompt positions clean (before computing xt)
    if valid_tokens is not None:
      prompt = ~valid_tokens.bool()
      reveal_mask = torch.where(
        prompt, torch.ones_like(reveal_mask), reveal_mask)
    xt = (onehot * reveal_mask.unsqueeze(-1)
          + (1 - reveal_mask).unsqueeze(-1) * xt_cont)
    return {
      'xt': xt,
      'reveal_mask': reveal_mask,
      'discrete_noise': discrete_noise,
      'continuous_noise': continuous_noise}

  # ── Loss ───────────────────────────────────────────────────────

  def nll_per_token(self, model_output, xt, x0, alpha_t,
                    dalpha_t, low_var=False, context=None,
                    train_mode=False, **kwargs):
    # reveal_mask handling is in _process_model_output, not here.
    del xt, low_var, context, train_mode
    log_p = torch.gather(
      model_output, dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    if self.pure_continuous:
      return -log_p
    return log_p * dalpha_t / (1 - alpha_t)

  def nll(self, x0, output_tokens, context,
          current_accumulation_step=None, train_mode=False,
          valid_tokens=None):
    del output_tokens, context, train_mode
    t = self._sample_t(x0.shape[0], current_accumulation_step)
    assert t.shape[0] == x0.shape[0]
    dalpha_t, alpha_t = self.noise(t)
    alpha_t = alpha_t.unsqueeze(-1)
    dalpha_t = dalpha_t.unsqueeze(-1)
    assert alpha_t.ndim == 2
    assert dalpha_t.ndim == 2
    sigma = self._sigma_from_alphat(alpha_t)

    noisy = self.q_xt(x0, alpha_t, valid_tokens=valid_tokens)
    ctx = CANDITrainingContext(
      reveal_mask=noisy['reveal_mask'],
      continuous_noise=noisy['continuous_noise'],
      is_embed=noisy.get('is_embed', False),
      x0_tokens=x0)
    log_x_theta = self.forward(
      xt=noisy['xt'], sigma=sigma, context=ctx)

    utils.print_nans(log_x_theta, 'model_output')
    loss = self.nll_per_token(
      model_output=log_x_theta,
      xt=noisy['xt'],
      x0=x0,
      alpha_t=alpha_t,
      dalpha_t=dalpha_t,
      reveal_mask=noisy['reveal_mask'])
    return loss, None

  # ── Sampling ───────────────────────────────────────────────────

  def prior_sample(self, *batch_dims):
    """Gaussian prior noise."""
    if self.is_embed:
      dim = self.config.model.hidden_size
      return (torch.randn(*batch_dims, dim,
                           device=self.device)
              * self.sigma_max)
    sigma = self.get_continuous_from_discrete_noise(
      torch.tensor(0.999, device=self.device))
    return (torch.randn(*batch_dims, self.vocab_size - 1,
                         dtype=torch.float32,
                         device=self.device)
            * sigma)

  # ── Sampling sub-steps (called by CANDISampler) ─────────────

  @torch.no_grad()
  def _continuous_step(self, x, time_t, sigma_s, sigma_t,
                       clean_mask=None, is_embed=False):
    """Score-based denoising: x_new = x - dt * (x - x0_hat)/s^2."""
    B = x.shape[0]
    dt_cont = torch.ones(B, device=x.device) * (
      sigma_s - sigma_t).item()
    t_vec = torch.ones(B, device=x.device) * time_t.item()
    sigma_t_vec = torch.ones(
      B, device=x.device) * sigma_t.item()
    if clean_mask is None:
      clean_mask = torch.zeros(
        x.shape[:-1], device=x.device)

    # Time embedding via noise schedule (consistent with training)
    _, alpha_t = self.noise(t_vec)
    sigma_emb = self._sigma_from_alphat(alpha_t.unsqueeze(-1)).squeeze(-1)
    ctx = CANDITrainingContext(
      reveal_mask=clean_mask,
      continuous_noise=sigma_t_vec,
      is_embed=is_embed)
    cond_denoised = self.forward(
      xt=x, sigma=sigma_emb.unsqueeze(-1),
      context=ctx).double()
    denoised = cond_denoised.exp()

    if self.is_embed:
      x0_hat = self.backbone.get_embedding(denoised)
    else:
      x0_hat = denoised
    d = (x - x0_hat) / (sigma_t_vec[:, None, None] ** 2)
    x_cont = x - dt_cont[:, None, None] * d
    return x_cont, denoised

  @torch.no_grad()
  def _continuous_step_cache(self, x, time_t, sigma_s,
                             sigma_t, clean_mask,
                             embedding_cache):
    """Cached denoising in embedding space."""
    B = x.shape[0]
    dt = sigma_s - sigma_t
    t_vec = torch.ones(B, device=x.device) * time_t.item()
    sigma_t_vec = torch.ones(
      B, device=x.device) * sigma_t.item()
    if clean_mask is None:
      clean_mask = torch.zeros(
        x.shape[:-1], device=x.device)

    # Time embedding via noise schedule (consistent with training)
    _, alpha_t = self.noise(t_vec)
    sigma_emb = self._sigma_from_alphat(alpha_t.unsqueeze(-1)).squeeze(-1)
    ctx = CANDITrainingContext(
      reveal_mask=clean_mask,
      continuous_noise=sigma_t_vec,
      is_embed=False,
      embedding_cache=embedding_cache)
    cond_denoised = self.forward(
      xt=x, sigma=sigma_emb.unsqueeze(-1),
      context=ctx).double()

    denoised = cond_denoised.exp()
    x0_hat = candi_utils.sample_categorical(denoised)
    embedding_hat = self.backbone.get_embedding(x0_hat)
    d = (embedding_cache - embedding_hat) / (sigma_t ** 2)
    new_cache = (embedding_cache
                 - dt * d * self.step_size)
    return new_cache, x0_hat

  def _discrete_step(self, x_sigma, p_x0, t, dt,
                     prev_clean_mask,
                     noise_removal_step=False):
    """Categorical discrete unmasking step."""
    s = 0.0 if noise_removal_step else t - dt
    B, L = x_sigma.shape[0], x_sigma.shape[1]
    t_val = t.item() if isinstance(t, torch.Tensor) else t
    s_val = s.item() if isinstance(s, torch.Tensor) else s

    mask_probs = torch.ones(
      B, L, 1, device=x_sigma.device) * s_val
    unmasked_probs = p_x0 * (t_val - s_val)
    q_xs = torch.cat([unmasked_probs, mask_probs], dim=-1)
    _x = candi_utils.sample_categorical(q_xs)
    new_clean = (
      prev_clean_mask.bool() | (_x != self.mask_index)
    ).float()

    old_tokens = x_sigma.argmax(dim=-1)
    sampled_real = torch.where(
      _x != self.mask_index, _x, old_tokens)
    updated = torch.where(
      prev_clean_mask.bool(), old_tokens, sampled_real)
    updated_x = F.one_hot(
      updated,
      num_classes=x_sigma.shape[-1]).float().to(
        x_sigma.device)
    updated_x = (updated_x * new_clean.unsqueeze(-1)
                 + (1 - new_clean).unsqueeze(-1) * x_sigma)
    return updated_x, new_clean

  def _discrete_step_optimized(self, x0_hat, xt, t, dt,
                               prev_clean_mask,
                               noise_removal_step=False):
    """Greedy discrete step for cached sampling.

    Note: mutates xt in-place for efficiency.
    """
    s = 0 if noise_removal_step else t - dt
    unmask = (torch.rand(prev_clean_mask.shape,
                         device=prev_clean_mask.device)
              < (t - s) / t)
    xt[~prev_clean_mask] = x0_hat[~prev_clean_mask]
    new_clean = prev_clean_mask | unmask
    return xt, new_clean
