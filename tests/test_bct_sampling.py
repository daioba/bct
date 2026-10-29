"""BCT sampling: BCT-E, BCT-P and BCT-L on top of the S-FLM sampler.

Pure-math tests run on CPU; end-to-end tests through `samplers.SFMSampler`
need CUDA (the rotary embedding is a Triton kernel).
"""
import math

import pytest
import torch

import algo
import dataloader
import samplers
import utils
from bct_helpers import BATCH, DEVICE, PROMPT_LEN, cuda_only, prefix, sfm_config

EPS = 1e-6
TS = (1.0, 0.5, 0.25, 0.1, 0.05, 2.0)
DT = torch.float64
A_BAND = (0.8865, 0.9090, 0.95, 0.999)
TEST_N_QUAD = 1 << 16


def _cos(shape, seed=0, dtype=torch.float64):
  g = torch.Generator().manual_seed(seed)
  return (2 * torch.rand(shape, generator=g, dtype=dtype) - 1) * 0.999


def test_log_q_is_identically_zero_at_a_equals_one():
  cos = _cos((3, 4, 64), seed=7)
  lq = samplers.sfm_log_q_unnorm(cos, 1.0, 512)
  assert torch.equal(lq, torch.zeros_like(lq))


def test_log_q_carries_the_d_minus_2_coefficient():
  cos = _cos((2, 3, 16), seed=11)
  a = 0.9
  base = samplers.sfm_log_q_unnorm(cos, a, 3)      # (d-2) = 1
  for d in (16, 512, 768):
    scaled = samplers.sfm_log_q_unnorm(cos, a, d)
    assert torch.allclose(scaled, (d - 2) * base, rtol=1e-12, atol=1e-9)


def test_theta_to_zero_is_finite_and_is_the_analytic_limit():
  a, d = 0.879, 768
  cos = torch.tensor([1.0, 1.0 - 1e-15, 1.0 - 1e-12, 1.0 - 1e-8],
                     dtype=torch.float64)
  lq = samplers.sfm_log_q_unnorm(cos, a, d)
  assert torch.isfinite(lq).all()
  assert torch.allclose(lq, torch.full_like(lq, (d - 2) * -math.log(a)),
                        rtol=0, atol=1e-4)


def test_support_boundary_is_hard_and_produces_no_nan():
  a, d = 0.9, 512
  boundary = math.cos(a * math.pi)
  theta = torch.tensor(
    [0.0, 0.5, a * math.pi - 1e-6, a * math.pi - 1e-12,
     a * math.pi, a * math.pi + 1e-12, math.pi], dtype=torch.float64)
  lq = samplers.sfm_log_q_unnorm(torch.cos(theta), a, d)
  assert not torch.isnan(lq).any()
  inside = theta < a * math.pi
  assert torch.isfinite(lq[inside]).all()
  assert torch.isneginf(lq[~inside]).all()
  assert math.isclose(float(torch.cos(theta)[4]), boundary, abs_tol=1e-12)


def _logits_and_logq(seed=0, V=64, a=0.9, d=512):
  g = torch.Generator().manual_seed(seed)
  logits = torch.log_softmax(
    torch.randn(3, 5, V, generator=g, dtype=torch.float64) * 3.0, dim=-1)
  cos = (2 * torch.rand((3, 5, V), generator=g, dtype=torch.float64) - 1)
  return logits, samplers.sfm_log_q_unnorm(cos.clamp(-0.9999, 0.9999), a, d)


def test_T_equals_one_is_the_identity():
  logits, lq = _logits_and_logq()
  out = samplers.bct_endpoint_logits(logits, lq, 1.0)
  masked = torch.isneginf(lq)
  assert torch.equal(out[~masked], logits[~masked])


@pytest.mark.parametrize('T', TS)
def test_candidate_independent_constants_cancel(T):
  logits, lq = _logits_and_logq(seed=3)
  ref = torch.log_softmax(samplers.bct_endpoint_logits(logits, lq, T), -1)
  shifted = torch.log_softmax(
    samplers.bct_endpoint_logits(logits + 12.5, lq - 7.25, T), -1)
  finite = torch.isfinite(ref)
  assert torch.allclose(ref[finite], shifted[finite], rtol=0, atol=1e-10)


@pytest.mark.parametrize('T', TS)
def test_out_of_support_has_probability_zero_at_every_T(T):
  logits, lq = _logits_and_logq(seed=5, a=0.5)
  masked = torch.isneginf(lq)
  assert masked.any(), 'test is vacuous unless something is out of support'
  assert (~masked).any()
  out = samplers.bct_endpoint_logits(logits, lq, T)
  p = torch.softmax(out, dim=-1)
  assert torch.equal(p[masked], torch.zeros_like(p[masked]))
  assert torch.isfinite(out[~masked]).all()
  assert not torch.isnan(p).any()
  assert torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)))


@pytest.mark.parametrize('T', TS)
def test_matches_a_from_scratch_bayes_calculation(T):
  g = torch.Generator().manual_seed(23)
  V, a, d = 48, 0.6, 32
  log_pi = torch.log_softmax(
    torch.randn(7, V, generator=g, dtype=torch.float64) * 2.0, -1)
  cos = (2 * torch.rand((7, V), generator=g, dtype=torch.float64) - 1)
  lq = samplers.sfm_log_q_unnorm(cos.clamp(-0.9999, 0.9999), a, d)
  assert torch.isneginf(lq).any()

  finite_lq = torch.where(torch.isneginf(lq), torch.zeros_like(lq), lq)
  log_r = torch.log_softmax(log_pi + finite_lq, dim=-1)  # r on the support
  want = torch.softmax(
    torch.where(torch.isneginf(lq),
                torch.full_like(lq, -float('inf')), log_pi / T + finite_lq),
    dim=-1)
  got = torch.softmax(samplers.bct_endpoint_logits(log_r, lq, T), dim=-1)
  assert torch.allclose(got, want, rtol=0, atol=1e-10)


def test_slerp_param_follows_the_repo_time_convention():
  alpha = torch.tensor(0.121, dtype=torch.float64)
  assert samplers.sfm_slerp_param(alpha, True) == alpha
  assert torch.isclose(samplers.sfm_slerp_param(alpha, False),
                       torch.tensor(0.879, dtype=torch.float64))


@pytest.mark.parametrize('d', [8, 768])
@pytest.mark.parametrize('a', A_BAND)
def test_path_weight_is_one_at_T1(a, d):
  tb = samplers.bct_path_table(a, d, 1.0, n_quad=TEST_N_QUAD, dtype=DT)
  theta = torch.linspace(1e-6, a * math.pi * (1 - 1e-9), 4096, dtype=DT)
  assert float((tb.weight(theta) - 1.0).abs().max()) < 1e-4


def test_path_table_resolution():
  assert samplers.BCT_PATH_N_QUAD >= 1 << 18
  assert samplers.BCT_PATH_N_GRID >= 1 << 16
  assert samplers.BCT_PATH_PHI_MIN <= 1e-9


def test_path_weight_does_not_mutate_its_argument():
  tb = samplers.bct_path_table(0.95, 768, 0.25, n_quad=TEST_N_QUAD,
                               dtype=DT)
  theta = torch.linspace(0.1, 2.5, 257, dtype=DT)
  before = theta.clone()
  tb.weight(theta)
  assert torch.equal(theta, before)


def test_trained_path_logits_reduce_to_identity_at_T1():
  torch.manual_seed(0)
  log_p = torch.log_softmax(torch.randn(3, 4, 7, dtype=DT), -1)
  log_q = torch.randn(3, 4, 7, dtype=DT)
  out = samplers.bct_trained_path_logits(log_p, log_q, 1.0)
  assert torch.allclose(out, log_p, atol=0, rtol=0)


def test_trained_path_logits_recover_the_tempered_prior():
  torch.manual_seed(0)
  T, V = 0.4, 32
  log_pi = torch.log_softmax(torch.randn(V, dtype=DT), -1)
  log_q = torch.randn(V, dtype=DT) * 3.0            # log q~_a
  log_s = torch.log_softmax(log_pi + log_q / T, -1)  # log q_{a,T} = log q_a / T
  got = torch.log_softmax(
    samplers.bct_trained_path_logits(log_s, log_q, T), -1)
  want = torch.log_softmax(log_pi / T + log_q / T, -1)
  assert float((got - want).abs().max()) < 1e-10


def test_trained_path_logits_kill_out_of_support_candidates():
  log_p = torch.log_softmax(torch.zeros(1, 1, 4, dtype=DT), -1)
  log_q = torch.tensor([[[0.0, -float('inf'), 1.0, -float('inf')]]], dtype=DT)
  out = samplers.bct_trained_path_logits(log_p, log_q, 0.4)
  assert bool(torch.isneginf(out[..., [1, 3]]).all())
  assert bool(torch.isfinite(out[..., [0, 2]]).all())


def test_correction_strength_one_is_bit_identical_to_exact_bctl():
  torch.manual_seed(0)
  log_p = torch.log_softmax(torch.randn(3, 4, 9, dtype=DT), -1)
  log_q = torch.randn(3, 4, 9, dtype=DT) * 5.0
  log_q[0, 0, 2] = -float('inf')
  for T in (0.4, 0.2, 0.1):
    ref = samplers.bct_endpoint_logits(log_p, log_q / T, T)
    got = samplers.bct_trained_path_logits(log_p, log_q, T, strength=1.0)
    assert torch.equal(torch.isneginf(ref), torch.isneginf(got))
    fin = torch.isfinite(ref)
    assert torch.equal(ref[fin], got[fin])


def test_correction_strength_zero_is_the_path_posterior():
  torch.manual_seed(1)
  log_p = torch.log_softmax(torch.randn(2, 3, 11, dtype=DT), -1)
  log_q = torch.randn(2, 3, 11, dtype=DT) * 5.0
  log_q[1, 2, 4] = -float('inf')
  T = 0.2
  zero = samplers.bct_trained_path_logits(log_p, log_q, T, strength=0.0)
  assert torch.allclose(torch.log_softmax(zero, -1),
                        torch.log_softmax(log_p / T, -1), atol=1e-12, rtol=0)
  half = samplers.bct_trained_path_logits(log_p, log_q, T, strength=0.5)
  full = samplers.bct_trained_path_logits(log_p, log_q, T, strength=1.0)
  fin = torch.isfinite(full)
  assert torch.allclose(half[fin], 0.5 * (zero[fin] + full[fin]),
                        atol=1e-9, rtol=0)
  assert bool(torch.isneginf(half[1, 2, 4]))


def _sflm_model():
  config = sfm_config('sphere-dit')
  torch.manual_seed(0)
  model = algo.SFM(config, dataloader.SudokuTokenizer())
  model.eval()
  model = model.to(DEVICE)
  generator = torch.Generator(device='cpu').manual_seed(1234)
  with torch.no_grad():
    for param in model.parameters():
      param.add_(torch.randn(param.shape, generator=generator,
                             dtype=torch.float32).to(param.device) * 0.02)
  return model


def _sampler(**overrides):
  kwargs = dict(noise_removal='greedy', velocity='exact', use_float64=True,
                slerp_float64=True, eps=EPS, temperature=1.0, p_nucleus=1.0,
                top_k=-1, top_k_velocity=-1, invert_time_convention=False)
  kwargs.update(overrides)
  return samplers.SFMSampler(**kwargs)


def _run(model, sampler, steps=6, seed=3, stop_before_decode=False):
  """One trajectory under a pinned RNG state."""
  prefix_tokens, prefix_lengths = prefix()
  torch.manual_seed(seed)
  state = sampler.init_state(model, BATCH, num_steps=steps,
                             prefix_tokens=prefix_tokens,
                             prefix_lengths=prefix_lengths)
  last = steps - 1
  while not state.done:
    if stop_before_decode and state.step_idx == last:
      return state
    state = sampler.step(model, state)
  return state


@cuda_only
def test_mode_none_is_bit_identical_to_upstream_sflm():
  model = _sflm_model()
  a = _run(model, _sampler())
  b = _run(model, _sampler(bct_mode='none', bct_T=0.05))
  assert torch.equal(a.xt, b.xt)


@cuda_only
def test_T_equals_one_reproduces_sflm_sampling_end_to_end():
  model = _sflm_model()
  a = _run(model, _sampler())
  b = _run(model, _sampler(bct_mode='endpoint', bct_T=1.0))
  assert torch.equal(a.xt, b.xt)


@cuda_only
def test_cos_omega_reuse_does_not_change_the_velocity():
  torch.manual_seed(0)
  x = utils.sphere_normalize(
    torch.randn(2, 5, 32, dtype=torch.float64, device=DEVICE))
  E = utils.sphere_normalize(
    torch.randn(11, 32, dtype=torch.float64, device=DEVICE))
  log_p = torch.log_softmax(
    torch.randn(2, 5, 11, dtype=torch.float64, device=DEVICE), -1)
  cos = torch.einsum('bld,vd->blv', x, E).clamp(-1 + EPS, 1 - EPS)
  want = samplers.sfm_compute_velocity(x, E, log_p, 'exact', EPS)
  got = samplers.sfm_compute_velocity(x, E, log_p, 'exact', EPS,
                                      cos_omega=cos)
  assert torch.allclose(got, want, rtol=0, atol=1e-12)


@cuda_only
def test_low_T_actually_changes_the_trajectory():
  model = _sflm_model()
  base = _run(model, _sampler(), stop_before_decode=True)
  low = _run(model, _sampler(bct_mode='endpoint', bct_T=0.1),
             stop_before_decode=True)
  assert not torch.equal(base.xt, low.xt)


@cuda_only
def test_bct_differs_from_standard_tempering_at_the_same_T():
  model = _sflm_model()
  standard = _run(model, _sampler(temperature=0.1), stop_before_decode=True)
  bct = _run(model, _sampler(bct_mode='endpoint', bct_T=0.1),
             stop_before_decode=True)
  assert not torch.equal(standard.xt, bct.xt)


@cuda_only
def test_conditioning_positions_are_untouched_by_the_correction():
  model = _sflm_model()
  prefix_tokens, _ = prefix()
  state = _run(model, _sampler(bct_mode='endpoint', bct_T=0.05))
  assert torch.equal(state.xt[:, :PROMPT_LEN], prefix_tokens)


@cuda_only
def test_support_diagnostics_are_recorded():
  model = _sflm_model()
  sampler = _sampler(bct_mode='endpoint', bct_T=0.5)
  state = _run(model, sampler)
  md = sampler.metadata(state)
  assert md['bct_mode'] == 'endpoint' and md['bct_T'] == 0.5
  assert md['bct_n_candidates'] > 0
  assert md['bct_n_out_of_support'] >= 0
  assert math.isfinite(md['bct_support_margin_min'])


def test_standard_and_bct_tempering_cannot_be_composed():
  with pytest.raises(ValueError, match='temperature'):
    _sampler(bct_mode='endpoint', bct_T=0.5, temperature=0.5)


def test_unknown_mode_and_bad_T_are_rejected():
  with pytest.raises(ValueError, match='bct.mode'):
    _sampler(bct_mode='full_path')
  with pytest.raises(ValueError, match='must be > 0'):
    _sampler(bct_mode='endpoint', bct_T=0.0)


@pytest.mark.parametrize('mode', ['path'])
def test_path_requires_matching_standard_temperature(mode):
  with pytest.raises(ValueError, match='temperature'):
    _sampler(bct_mode=mode, bct_T=0.25, temperature=1.0)
  with pytest.raises(ValueError, match='temperature'):
    _sampler(bct_mode=mode, bct_T=0.25, temperature=0.5)
  _sampler(bct_mode=mode, bct_T=0.25, temperature=0.25)


@pytest.mark.parametrize('mode', ['path'])
def test_path_rejects_velocity_modes_it_cannot_weight(mode):
  with pytest.raises(ValueError, match='velocity=exact'):
    _sampler(bct_mode=mode, bct_T=0.5, temperature=0.5,
                  velocity='sample')
  with pytest.raises(ValueError, match='top_k_velocity'):
    _sampler(bct_mode=mode, bct_T=0.5, temperature=0.5,
                  top_k_velocity=1)


@cuda_only
def test_path_at_T1_reproduces_sflm_sampling_end_to_end():
  model = _sflm_model()
  a = _run(model, _sampler())
  b = _run(model, _sampler(bct_mode='path', bct_T=1.0, temperature=1.0))
  assert torch.equal(a.xt, b.xt)


@cuda_only
def test_path_differs_from_standard_tempering_at_the_same_T():
  model = _sflm_model()
  standard = _run(model, _sampler(temperature=0.1), stop_before_decode=True)
  path = _run(model, _sampler(bct_mode='path', bct_T=0.1,
                                   temperature=0.1), stop_before_decode=True)
  assert not torch.equal(standard.xt, path.xt)


@cuda_only
def test_path_keeps_the_standard_posterior():
  model = _sflm_model()
  path = _run(model, _sampler(bct_mode='path', bct_T=0.1,
                                   temperature=0.1), stop_before_decode=True)
  endpoint = _run(model, _sampler(bct_mode='endpoint', bct_T=0.1),
                  stop_before_decode=True)
  assert not torch.equal(path.xt, endpoint.xt)


@cuda_only
def test_path_weight_diagnostics_are_recorded():
  model = _sflm_model()
  sampler = _sampler(bct_mode='path', bct_T=0.25, temperature=0.25)
  state = _run(model, sampler)
  md = sampler.metadata(state)
  assert md['bct_mode'] == 'path'
  assert md['bct_w_min'] is not None and md['bct_w_max'] is not None
  assert md['bct_w_min'] <= md['bct_w_mean'] <= md['bct_w_max']
  assert md['bct_w_max'] > 1.0        # T < 1 speeds the flow up


def test_correction_strength_is_rejected_outside_trained_path():
  with pytest.raises(ValueError, match='correction_strength'):
    _sampler(bct_mode='path', bct_T=0.4, temperature=0.4,
             bct_correction_strength=0.5)
  with pytest.raises(ValueError, match='correction_strength'):
    _sampler(bct_mode='trained_path', bct_T=0.4, bct_correction_strength=-0.1)
  _sampler(bct_mode='trained_path', bct_T=0.4, bct_correction_strength=0.5)


def test_trained_path_requires_temperature_one():
  with pytest.raises(ValueError, match='trained_path requires'):
    _sampler(bct_mode='trained_path', bct_T=0.4, temperature=0.4)
  _sampler(bct_mode='trained_path', bct_T=0.4, temperature=1.0)


def test_trained_path_rejects_velocity_modes_it_cannot_weight():
  with pytest.raises(ValueError, match='velocity=exact'):
    _sampler(bct_mode='trained_path', bct_T=0.4, velocity='sample')
  with pytest.raises(ValueError, match='top_k_velocity'):
    _sampler(bct_mode='trained_path', bct_T=0.4, top_k_velocity=1)
