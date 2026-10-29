"""Tiny Sudoku-flavoured S-FLM configs shared by the BCT tests."""
import omegaconf
import pytest
import torch

DEVICE = 'cuda'
VOCAB = 12
DIM = 32
LENGTH = 16
BATCH = 4
PROMPT_LEN = LENGTH // 2

cuda_only = pytest.mark.skipif(
  not torch.cuda.is_available(),
  reason='the rotary embedding is a Triton kernel and needs CUDA')


def model_config(model_type, temp_conditioning=False):
  model = {
    'name': 'test',
    'type': model_type,
    'hidden_size': DIM,
    'cond_dim': 16,
    'length': LENGTH,
    'n_blocks': 2,
    'n_heads': 4,
    'dropout': 0.0,
    'init': 'ngpt',
    'eps': 1e-6,
    'pretrained_ckpt_path': None,
    'temp_conditioning': temp_conditioning,
  }
  if model_type == 'sphere-dit':
    model['learn_temperature_scaling'] = False
  else:
    model['use_time_alpha'] = True
    model['use_time_token'] = False
    model['mlp_type'] = 'gelu'
    model['normalize_input_embed'] = False
  return model


def sfm_config(model_type='sphere-dit', temp_conditioning=False, **algo):
  """A complete Hydra-shaped config for `algo.SFM`."""
  config = omegaconf.OmegaConf.create({
    'mode': 'train',
    'model': model_config(model_type, temp_conditioning),
    'algo': {
      'name': 'sfm', 'diffusion_type': 'sphere', 'backbone': model_type,
      'parameterization': 'mean',
      'time_conditioning': True, 'loss_type': 'ce', 'T': 0,
      'causal_attention': False, 'adaLN': True, 'slerp_precision': 'float64',
      'eps': 1e-6, 'invert_time_convention': False,
      'renormalize_weights': model_type == 'sphere-arch',
      'bct_train_T_dist': 'none', 'bct_train_p_one': 0.0,
      'bct_train_p_low': 0.0, 'bct_train_T_low': 0.1,
      'bct_train_beta_min': 1.25, 'bct_train_beta_max': 5.0,
    },
    'noise': {'type': 'log-linear', 'eps': 1e-3, 'alpha_min': None,
              'alpha_max': 0.093, 'adaptive': False},
    'sampler': {'predictor': 'sfm', 'steps': 8, 'noise_removal': 'greedy',
                'use_float64': True, 'velocity': 'exact', 'p_nucleus': 1.0,
                'top_k': -1, 'top_k_velocity': -1, 'temperature': 1.0,
                'num_sample_batches': 1, 'num_sample_log': 1},
    'training': {'ema': 0.0, 'antithetic_sampling': True,
                 'importance_sampling': False, 'sampling_eps': 1e-3,
                 'change_of_variables': False, 'loss_precision': 'float32',
                 'finetune_path': '', 'resample': False},
    'eval': {'gen_ppl_eval_model_name_or_path': 'gpt2-large',
             'perplexity_batch_size': 8},
    'optim': {'weight_decay': 0, 'lr': 3e-4, 'beta1': 0.9, 'beta2': 0.999,
              'eps': 1e-8},
    'loader': {'global_batch_size': BATCH, 'batch_size': BATCH,
               'eval_batch_size': BATCH, 'num_workers': 0,
               'pin_memory': False},
    'trainer': {'devices': 1, 'num_nodes': 1, 'accumulate_grad_batches': 1},
    'neg_infinity_mode': 'large-finite',
  })
  for k, v in algo.items():
    config.algo[k] = v
  return config


def prefix(seed=0):
  """A clean conditioning prefix (the first half of the sequence)."""
  generator = torch.Generator().manual_seed(seed)
  x0 = torch.randint(0, VOCAB, (BATCH, LENGTH), generator=generator)
  prefix_tokens = x0[:, :PROMPT_LEN].contiguous().to(DEVICE)
  prefix_lengths = torch.full((BATCH,), PROMPT_LEN, device=DEVICE,
                              dtype=torch.long)
  return prefix_tokens, prefix_lengths
