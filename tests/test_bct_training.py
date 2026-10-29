"""Tempered-flow training (TFT): exact tempered corruption, temperature
conditioning, and resuming a released checkpoint with new parameters.

The corruption tests run on CPU; the model-level tests need CUDA.
"""
import copy
import math

import pytest
import torch

import algo
import dataloader
import samplers
import trainer_base
import utils
from bct_helpers import (BATCH, DEVICE, LENGTH, VOCAB, cuda_only,
                         sfm_config)

DT = torch.float64
MULTI_KW = dict(n_quad=1 << 15)
TEMP_KEYS = ('temp_map.', 'temp_modulation.')


def test_mixed_T_rows_do_not_leak_into_each_other():
  d = 64
  psi = torch.rand(8, 32, 1, dtype=DT) * 0.4 + 0.3
  a = torch.rand(8, dtype=DT) * 0.1 + 0.88
  T = torch.tensor([1.0, 0.1, 0.4, 0.2, 0.8, 1.0, 0.5, 0.3], dtype=DT)
  mixed = samplers.bct_corruption_transport(psi, a, T, d, **MULTI_KW)
  for i in range(8):
    alone = samplers.bct_corruption_transport(
      psi[i:i + 1], a[i:i + 1], T[i:i + 1], d, **MULTI_KW)
    assert float((mixed[i:i + 1] - alone).abs().max()) < 1e-15


def test_per_example_transport_rejects_degenerate_arguments():
  d = 64
  psi = torch.full((1, 4, 1), 0.5, dtype=DT)
  with pytest.raises(ValueError, match='a > 0'):
    samplers.bct_corruption_transport(
      psi, torch.zeros(1, dtype=DT), torch.ones(1, dtype=DT), d, **MULTI_KW)
  with pytest.raises(ValueError, match='T > 0'):
    samplers.bct_corruption_transport(
      psi, torch.tensor([0.9], dtype=DT), torch.zeros(1, dtype=DT), d,
      **MULTI_KW)


def test_multi_T_corruption_at_T1_is_utils_slerp():
  d, B, L = 64, 6, 32
  torch.manual_seed(0)
  clean = utils.sphere_normalize(torch.randn(B, L, d, dtype=DT))
  noisy = utils.sphere_normalize(torch.randn(B, L, d, dtype=DT))
  a = torch.rand(B, dtype=DT) * 0.1 + 0.88
  cos_omega = (noisy * clean).sum(-1, keepdim=True).clamp(-1 + 1e-6,
                                                          1 - 1e-6)
  omega = torch.acos(cos_omega)
  psi = omega / math.pi
  phi = samplers.bct_corruption_transport(
    psi, a, torch.ones(B, dtype=DT), d, **MULTI_KW)
  a_b = a.reshape(B, 1, 1)
  alpha_eff = a_b * (phi / psi.clamp(min=1e-300))
  sin_omega = omega.sin().clamp(min=1e-6)
  got = (((1 - alpha_eff) * omega).sin() / sin_omega) * clean \
      + ((alpha_eff * omega).sin() / sin_omega) * noisy
  want = utils.slerp(clean=clean, noisy=noisy, alpha_t=a.reshape(B, 1),
                     eps=1e-6)
  assert float((got - want).abs().max()) < 1e-12


# ── sphere-arch ─────────────────────────────────────────────────────────
def _sfm(temp_conditioning=False, seed=0, **algo_overrides):
  torch.manual_seed(seed)
  model = algo.SFM(sfm_config('sphere-arch', temp_conditioning,
                              **algo_overrides),
                   dataloader.SudokuTokenizer())
  model.eval()
  return model.to(DEVICE)


def _inputs(model, seed=0):
  g = torch.Generator(device='cpu').manual_seed(seed)
  x0 = torch.randint(0, VOCAB, (BATCH, LENGTH), generator=g).to(DEVICE)
  xt = model.backbone.get_sphere_embeddings(x0)
  sigma = torch.full((BATCH, 1), 0.7, device=DEVICE)
  return x0, xt, sigma


def test_temp_conditioning_adds_only_the_expected_parameters():
  plain = {n for n, _ in _sfm(False).named_parameters()}
  cond = {n for n, _ in _sfm(True).named_parameters()}
  extra = cond - plain
  assert plain <= cond
  assert extra
  assert all(any(k in n for k in TEMP_KEYS) for n in extra)
  # one modulation per block plus one shared embedder
  n_blocks = _sfm(True).backbone.n_blocks
  assert sum('temp_modulation.weight' in n for n in extra) == n_blocks


def test_untrained_conditioning_leaves_the_model_unchanged():
  model = _sfm(True)
  x0, xt, sigma = _inputs(model)
  base = model.forward(x0=x0, xt=xt, sigma=sigma,
                       context=trainer_base.TrainingContext())
  for T in (1.0, 0.4, 0.1):
    tau = torch.full((BATCH,), -torch.tensor(T).log().item(), device=DEVICE)
    out = model.forward(x0=x0, xt=xt, sigma=sigma,
                        context=trainer_base.TrainingContext(bct_tau=tau))
    assert torch.equal(base, out)


def test_tau_gate_pins_T1_to_the_unconditioned_model():
  model = _sfm(True)
  with torch.no_grad():
    for block in model.backbone.blocks:
      block.temp_modulation.weight.normal_(std=0.1)
      block.temp_modulation.bias.normal_(std=0.1)
  x0, xt, sigma = _inputs(model)
  base = model.forward(x0=x0, xt=xt, sigma=sigma,
                       context=trainer_base.TrainingContext())
  at_one = model.forward(
    x0=x0, xt=xt, sigma=sigma,
    context=trainer_base.TrainingContext(
      bct_tau=torch.zeros(BATCH, device=DEVICE)))
  assert torch.equal(base, at_one)
  at_low = model.forward(
    x0=x0, xt=xt, sigma=sigma,
    context=trainer_base.TrainingContext(
      bct_tau=torch.full((BATCH,), 2.302585, device=DEVICE)))
  assert not torch.allclose(base, at_low)


def test_conditioning_is_per_sequence():
  model = _sfm(True)
  with torch.no_grad():
    for block in model.backbone.blocks:
      block.temp_modulation.weight.normal_(std=0.1)
  x0, xt, sigma = _inputs(model)
  tau = torch.tensor([0.0, 2.3, 0.9, 1.6], device=DEVICE)[:BATCH]
  mixed = model.forward(x0=x0, xt=xt, sigma=sigma,
                        context=trainer_base.TrainingContext(bct_tau=tau))
  for i in range(BATCH):
    uniform = model.forward(
      x0=x0, xt=xt, sigma=sigma,
      context=trainer_base.TrainingContext(
        bct_tau=torch.full_like(tau, float(tau[i]))))
    assert torch.equal(mixed[i], uniform[i])


def test_a_model_without_conditioning_ignores_tau():
  model = _sfm(False)
  x0, xt, sigma = _inputs(model)
  base = model.forward(x0=x0, xt=xt, sigma=sigma,
                       context=trainer_base.TrainingContext())
  out = model.forward(
    x0=x0, xt=xt, sigma=sigma,
    context=trainer_base.TrainingContext(
      bct_tau=torch.full((BATCH,), 2.3, device=DEVICE)))
  assert torch.equal(base, out)


def test_no_distribution_draws_nothing():
  model = _sfm(False, bct_train_T_dist='none')
  assert model._bct_sample_train_T(64, DEVICE) is None


def test_mixture_has_the_configured_atoms_and_beta_range():
  model = _sfm(True, bct_train_T_dist='mixture', bct_train_p_one=0.2,
               bct_train_p_low=0.1, bct_train_T_low=0.1,
               bct_train_beta_min=1.25, bct_train_beta_max=5.0)
  torch.manual_seed(0)
  T = model._bct_sample_train_T(200000, DEVICE)
  p_one = float((T == 1.0).double().mean())
  p_low = float((T == 0.1).double().mean())
  assert abs(p_one - 0.2) < 0.01
  assert abs(p_low - 0.1) < 0.01
  cont = T[(T != 1.0) & (T != 0.1)]
  beta = 1.0 / cont
  assert float(beta.min()) >= 1.25 - 1e-9
  assert float(beta.max()) <= 5.0 + 1e-9
  # uniform in beta, not in T
  assert abs(float(beta.mean()) - 3.125) < 0.02


def test_mixture_without_low_mass_never_draws_the_low_atom():
  model = _sfm(True, bct_train_T_dist='mixture', bct_train_p_one=0.2,
               bct_train_p_low=0.0, bct_train_T_low=0.1)
  torch.manual_seed(0)
  T = model._bct_sample_train_T(100000, DEVICE)
  assert float((T == 0.1).double().mean()) == 0.0
  assert abs(float((T == 1.0).double().mean()) - 0.2) < 0.01
  assert float(T[T != 1.0].min()) >= 0.2 - 1e-9


def test_tau_is_minus_log_T():
  model = _sfm(True, bct_train_T_dist='mixture', bct_train_p_one=0.2,
               bct_train_p_low=0.1)
  T = model._bct_sample_train_T(1024, DEVICE)
  tau = model._bct_train_tau(T, 1024, DEVICE)
  assert torch.allclose(tau, -T.log())
  assert float(tau[T == 1.0].abs().max()) == 0.0


def test_multi_T_corruption_at_T1_matches_the_untempered_one():
  model = _sfm(False)
  x0 = torch.randint(0, VOCAB, (BATCH, LENGTH), device=DEVICE)
  alpha_t = torch.full((BATCH, 1), 0.1, device=DEVICE)
  torch.manual_seed(3)
  plain = model.q_xt(x0, alpha_t, use_pure_noise=False)
  torch.manual_seed(3)
  multi = model.q_xt(x0, alpha_t, use_pure_noise=False,
                     bct_T=torch.ones(BATCH, device=DEVICE))
  assert float((plain - multi).abs().max()) < 1e-10


def _fake_checkpoint(model):
  """The three positionally-indexed structures Lightning restores."""
  names = model._bct_param_names()
  sd = {k: v.detach().clone().cpu()
        for k, v in model.state_dict().items()}
  params = [dict(model.named_parameters())[n] for n in names]
  return {
    'state_dict': sd,
    'ema': {'decay': 0.9999, 'num_updates': 250000,
            'shadow_params': [torch.full_like(p, float(i)).cpu()
                              for i, p in enumerate(params)]},
    'optimizer_states': [{
      'state': {i: {'step': torch.tensor(250000.0),
                    'exp_avg': torch.full_like(p, float(i)).cpu(),
                    'exp_avg_sq': torch.full_like(p, float(i)).cpu()}
                for i, p in enumerate(params)},
      'param_groups': [{'lr': 3e-4, 'initial_lr': 3e-4,
                        'params': list(range(len(params)))}]}],
    'loops': {'fit_loop': {
      'epoch_progress': {'current': {'completed': 12}},
      'epoch_loop.batch_progress': {'current': {'completed': 17089}}}},
  }


def test_resume_without_new_parameters_is_untouched():
  model = _sfm(False)
  ck = _fake_checkpoint(model)
  before = copy.deepcopy(ck['optimizer_states'][0]['param_groups'][0]
                         ['params'])
  model.on_load_checkpoint(ck)
  assert ck['optimizer_states'][0]['param_groups'][0]['params'] == before
  assert len(ck['ema']['shadow_params']) == len(model._bct_param_names())


def test_resume_realigns_ema_and_optimizer_state_by_name():
  plain = _sfm(False)
  ck = _fake_checkpoint(plain)
  old_names = plain._bct_param_names()
  marker = {n: i for i, n in enumerate(old_names)}

  cond = _sfm(True)
  cond.on_load_checkpoint(ck)
  new_names = cond._bct_param_names()
  assert len(new_names) > len(old_names)

  state = ck['optimizer_states'][0]['state']
  groups = ck['optimizer_states'][0]['param_groups'][0]
  assert groups['params'] == list(range(len(new_names)))
  assert groups['lr'] == 3e-4 and groups['initial_lr'] == 3e-4
  for i, n in enumerate(new_names):
    shadow = ck['ema']['shadow_params'][i]
    if n in marker:
      assert i in state
      assert float(state[i]['exp_avg'].flatten()[0]) == marker[n]
      assert float(state[i]['step']) == 250000.0
      assert float(shadow.flatten()[0]) == marker[n]
    else:
      assert i not in state
      assert torch.equal(shadow,
                         dict(cond.named_parameters())[n].detach().cpu())
      if 'temp_modulation.' in n:
        assert float(shadow.abs().max()) == 0.0
  assert set(ck['state_dict']) - set(plain.state_dict()) == (
    set(new_names) - set(old_names))


def test_resume_refuses_a_mismatch_it_cannot_repair():
  plain = _sfm(False)
  ck = _fake_checkpoint(plain)
  victim = 'backbone.lm_head.weight'
  assert victim in ck['state_dict']
  del ck['state_dict'][victim]
  with pytest.raises(RuntimeError, match='not temperature-conditioning'):
    _sfm(True).on_load_checkpoint(ck)


def test_conditioned_checkpoint_into_unconditioned_model_raises():
  cond = _sfm(True)
  ck = _fake_checkpoint(cond)
  with pytest.raises(RuntimeError, match='temp_conditioning is False'):
    _sfm(False).on_load_checkpoint(ck)


def test_unconditioned_checkpoint_into_conditioned_model_raises_at_eval():
  plain = _sfm(False)
  ck = _fake_checkpoint(plain)
  model = _sfm(True)
  model.config.mode = 'gsm8k_eval'
  with pytest.raises(RuntimeError, match='freshly'):
    model.on_load_checkpoint(ck)


# ── sphere-dit ──────────────────────────────────────────────────────────
def _sfm_dit(temp_conditioning=False, seed=0):
  torch.manual_seed(seed)
  model = algo.SFM(sfm_config('sphere-dit', temp_conditioning),
                   dataloader.SudokuTokenizer())
  with torch.no_grad():
    head = model.backbone.output_layer.linear
    head.weight.normal_(std=head.weight.shape[1] ** -0.5)
    head.bias.normal_(std=0.1)
  model.eval()
  return model.to(DEVICE)


def _perturb(model, std=0.1):
  with torch.no_grad():
    for n, p in model.backbone.named_parameters():
      if 'temp_modulation.' in n:
        p.normal_(std=std)


def _tau(value):
  return torch.full((BATCH,), float(value), device=DEVICE)


def test_temp_conditioning_adds_only_the_expected_parameters_dit():
  plain = {n for n, _ in _sfm_dit(False).named_parameters()}
  model = _sfm_dit(True)
  cond = {n for n, _ in model.named_parameters()}
  extra = cond - plain
  assert plain <= cond
  assert all(any(k in n for k in TEMP_KEYS) for n in extra)
  n_blocks = len(model.backbone.blocks)
  # one per block + one for the final layer
  assert sum(n.endswith('temp_modulation.weight') for n in extra) == (
    n_blocks + 1)
  assert 'backbone.output_layer.temp_modulation.weight' in extra
  assert sum(n.startswith('backbone.temp_map.') for n in extra) > 0


def test_default_backbone_is_unchanged_by_the_flag_dit():
  cfg = sfm_config('sphere-dit', False)
  del cfg.model['temp_conditioning']
  torch.manual_seed(0)
  bare = algo.SFM(cfg, dataloader.SudokuTokenizer())
  flagged = _sfm_dit(False)
  assert ([n for n, _ in bare.named_parameters()]
          == [n for n, _ in flagged.named_parameters()])


def test_untrained_conditioning_leaves_the_model_unchanged_dit():
  model = _sfm_dit(True)
  x0, xt, sigma = _inputs(model)
  base = model.forward(x0=x0, xt=xt, sigma=sigma,
                       context=trainer_base.TrainingContext())
  for T in (1.0, 0.4, 0.1):
    out = model.forward(
      x0=x0, xt=xt, sigma=sigma,
      context=trainer_base.TrainingContext(
        bct_tau=_tau(-torch.tensor(T).log().item())))
    assert torch.equal(base, out)


def test_tau_gate_pins_T1_to_the_unconditioned_model_dit():
  model = _sfm_dit(True)
  _perturb(model)
  x0, xt, sigma = _inputs(model)
  base = model.forward(x0=x0, xt=xt, sigma=sigma,
                       context=trainer_base.TrainingContext())
  at_one = model.forward(x0=x0, xt=xt, sigma=sigma,
                         context=trainer_base.TrainingContext(
                           bct_tau=_tau(0.0)))
  assert torch.equal(base, at_one)
  at_low = model.forward(x0=x0, xt=xt, sigma=sigma,
                         context=trainer_base.TrainingContext(
                           bct_tau=_tau(2.302585)))
  assert not torch.allclose(base, at_low)


def test_final_layer_modulation_is_live_dit():
  model = _sfm_dit(True)
  with torch.no_grad():
    model.backbone.output_layer.temp_modulation.bias.normal_(std=0.1)
  x0, xt, sigma = _inputs(model)
  base = model.forward(x0=x0, xt=xt, sigma=sigma,
                       context=trainer_base.TrainingContext())
  out = model.forward(x0=x0, xt=xt, sigma=sigma,
                      context=trainer_base.TrainingContext(
                        bct_tau=_tau(1.6)))
  assert not torch.allclose(base, out)


def test_conditioning_is_per_sequence_dit():
  model = _sfm_dit(True)
  _perturb(model)
  x0, xt, sigma = _inputs(model)
  tau = torch.tensor([0.0, 2.3, 0.9, 1.6], device=DEVICE)[:BATCH]
  mixed = model.forward(x0=x0, xt=xt, sigma=sigma,
                        context=trainer_base.TrainingContext(bct_tau=tau))
  for i in range(BATCH):
    uniform = model.forward(
      x0=x0, xt=xt, sigma=sigma,
      context=trainer_base.TrainingContext(
        bct_tau=torch.full_like(tau, float(tau[i]))))
    assert torch.equal(mixed[i], uniform[i])


def test_a_model_without_conditioning_ignores_tau_dit():
  model = _sfm_dit(False)
  x0, xt, sigma = _inputs(model)
  base = model.forward(x0=x0, xt=xt, sigma=sigma,
                       context=trainer_base.TrainingContext())
  out = model.forward(x0=x0, xt=xt, sigma=sigma,
                      context=trainer_base.TrainingContext(
                        bct_tau=_tau(2.3)))
  assert torch.equal(base, out)


def test_resume_realigns_ema_and_optimizer_state_by_name_dit():
  plain = _sfm_dit(False)
  ck = _fake_checkpoint(plain)
  old_names = plain._bct_param_names()
  marker = {n: i for i, n in enumerate(old_names)}

  cond = _sfm_dit(True)
  cond.on_load_checkpoint(ck)
  new_names = cond._bct_param_names()
  assert len(new_names) > len(old_names)
  state = ck['optimizer_states'][0]['state']
  assert ck['optimizer_states'][0]['param_groups'][0]['params'] == list(
    range(len(new_names)))
  for i, n in enumerate(new_names):
    shadow = ck['ema']['shadow_params'][i]
    if n in marker:
      assert float(state[i]['exp_avg'].flatten()[0]) == marker[n]
      assert float(shadow.flatten()[0]) == marker[n]
    else:
      assert i not in state
      assert torch.equal(shadow,
                         dict(cond.named_parameters())[n].detach().cpu())
      if 'temp_modulation.' in n:
        assert float(shadow.abs().max()) == 0.0
  assert set(ck['state_dict']) - set(plain.state_dict()) == (
    set(new_names) - set(old_names))


def test_conditioned_checkpoint_into_unconditioned_model_raises_dit():
  ck = _fake_checkpoint(_sfm_dit(True))
  with pytest.raises(RuntimeError, match='temp_conditioning is False'):
    _sfm_dit(False).on_load_checkpoint(ck)


def test_unconditioned_checkpoint_into_conditioned_model_raises_at_eval_dit():
  ck = _fake_checkpoint(_sfm_dit(False))
  model = _sfm_dit(True)
  model.config.mode = 'gsm8k_eval'
  with pytest.raises(RuntimeError, match='freshly'):
    model.on_load_checkpoint(ck)
