#!/usr/bin/env python
"""Print the evaluation commands behind each result of the paper.

Each printed line is one evaluation run (generation and grading) with the
checkpoint, method, temperature, lambda, seed and output directory filled in,
for example

  CKPT_PATH=... BACKBONE=sphere-arch METHOD=bctp T=0.4 SEED=1 \\
  OUTPUT_DIR=outputs/gsm8k/sarch/released/bctp_T0.4_seed1 bash scripts/bct/eval_gsm8k.sh

Nothing is run.  Run the lines (one after another, or one job per line on a
cluster), then aggregate them with `scripts/bct/summarize.py`, which expects
exactly these output directories.

  python scripts/bct/list_runs.py gsm8k_main > runs.txt

Results:
  gsm8k_main      GSM8K main table and figure            (312 runs, 8 GPUs each)
  passk           Pass@K                                 (128 runs, 8 GPUs each)
  sudoku          Sudoku-Hard                            (168 runs, 1 GPU each)
  lambda_gsm8k    held-out selection of lambda, TinyGSM  ( 30 runs, 8 GPUs each)
  lambda_sudoku   held-out selection of lambda, Sudoku   ( 15 runs, 1 GPU each)
  dlm             MDLM / Duo with standard tempering     ( 16 runs, 8 GPUs each)
  top1            Top-1 reference of the main table      (  2 runs, 8 GPUs each)
  all             all of the above, without duplicates

`lambda_gsm8k` needs the held-out problems built by
`scripts/bct/prepare_tinygsm_heldout.py` (`--heldout-dir`).  With
`--dit-tft500-ckpt`, it also lists the runs of the DiT TFT checkpoint after 500
updates (the last column of the held-out table; see `train_tinygsm.sh`).
"""
import argparse
import os
import sys

TEMPS = (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
SEEDS = (1, 11, 21)
PASSK_SEEDS = tuple(1 + 10 * j for j in range(8))
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
SELECTION_TEMPS = (0.4, 0.2, 0.1)
# lambda selected on held-out problems (see `summarize.py lambda`)
LAMBDA_STAR = {'sarch': 0.5, 'sdit': 0.25, 'sudoku': 1.0}

BACKBONES = {'sarch': ('sphere-arch', 'sphere_arch_truncated_adaptive_no_renorm'),
             'sdit': ('sphere-dit', 'sphere_dit_truncated_adaptive_no_renorm')}


def num(x):
  """1.0 -> '1.0', 0.4 -> '0.4', 0.25 -> '0.25' (as in the run names)."""
  return f'{x:.1f}' if round(x, 1) == x else f'{x:g}'


def run_name(method, T, seed, lam=None):
  name = f'{method}_T{num(T)}'
  if lam is not None:
    name += f'_l{num(lam)}'
  return f'{name}_seed{seed}'


class Lister:

  def __init__(self, ckpt_dir, out_dir, heldout_dir, dit_tft500_ckpt):
    self.ckpt = ckpt_dir
    self.out = out_dir
    self.heldout = heldout_dir
    self.dit_tft500 = dit_tft500_ckpt
    self.lines = []
    self.seen = set()

  def add(self, script, out_rel, **env):
    out = os.path.join(self.out, out_rel)
    if out in self.seen:
      return
    self.seen.add(out)
    assigns = ' '.join(f'{k}={v}' for k, v in env.items())
    self.lines.append(f'{assigns} OUTPUT_DIR={out} bash {script}')

  def gsm8k_ckpt(self, bb, model):
    name = BACKBONES[bb][1]
    if model == 'released':
      return os.path.join(self.ckpt, 's-flm', 'tinygsm', 'sfm', f'{name}.ckpt')
    return os.path.join(self.ckpt, 'bct', 'tinygsm', 'sfm', f'{name}_{model}.ckpt')

  def gsm8k(self, bb, model, method, T, seed, lam=None, **extra):
    env = {'CKPT_PATH': self.gsm8k_ckpt(bb, model),
           'BACKBONE': BACKBONES[bb][0]}
    if model == 'tft':
      env['TFT'] = 1
    env.update(METHOD=method, T=num(T))
    if lam is not None:
      env['LAMBDA'] = num(lam)
    env['SEED'] = seed
    env.update(extra)
    self.add('scripts/bct/eval_gsm8k.sh',
             f'gsm8k/{bb}/{model}/{run_name(method, T, seed, lam)}', **env)

  def sudoku(self, model, method, T, seed, lam=None, heldout=False):
    suffix = {'base': '', 'cont': '_cont', 'tft': '_tft'}[model]
    env = {'CKPT_PATH': os.path.join(self.ckpt, 'bct', 'sudoku_hard', 'sfm',
                                     f'sfm_truncated_adaptive{suffix}.ckpt')}
    if model == 'tft':
      env['TFT'] = 1
    env.update(METHOD=method, T=num(T))
    if lam is not None:
      env['LAMBDA'] = num(lam)
    env['SEED'] = seed
    if heldout:
      env['EVAL_SET'] = 'heldout'
    root = 'sudoku_hard_heldout' if heldout else 'sudoku_hard'
    self.add('scripts/bct/eval_sudoku_hard.sh',
             f'{root}/{model}/{run_name(method, T, seed, lam)}', **env)

  # ── the results of the paper ───────────────────────────────────────────
  def gsm8k_main(self):
    for bb in BACKBONES:
      for model in ('released', 'cont', 'tft'):
        for seed in SEEDS:
          # T = 1: every BCT correction is the identity.
          self.gsm8k(bb, model, 'standard', 1.0, seed)
          for T in TEMPS[1:]:
            if model == 'tft':
              self.gsm8k(bb, model, 'bctl', T, seed, LAMBDA_STAR[bb])
            else:
              for method in ('standard', 'bcte', 'bctp'):
                self.gsm8k(bb, model, method, T, seed)

  def passk(self):
    for seed in PASSK_SEEDS:
      for model in ('cont', 'tft'):
        self.gsm8k('sarch', model, 'standard', 1.0, seed)
        for T in TEMPS[1:]:
          if model == 'cont':
            self.gsm8k('sarch', model, 'standard', T, seed)
          else:
            self.gsm8k('sarch', model, 'bctl', T, seed, LAMBDA_STAR['sarch'])

  def sudoku_all(self):
    for seed in SEEDS:
      for T in TEMPS:
        for model in ('base', 'cont'):
          for method in ('standard', 'bcte', 'bctp'):
            self.sudoku(model, method, T, seed)
        self.sudoku('tft', 'bctl', T, seed, LAMBDA_STAR['sudoku'])

  def lambda_gsm8k(self):
    if self.heldout is None:
      sys.exit('lambda_gsm8k needs --heldout-dir')
    models = [(bb, 'tft', self.gsm8k_ckpt(bb, 'tft')) for bb in BACKBONES]
    if self.dit_tft500:
      models.append(('sdit', 'tft500', self.dit_tft500))
    for bb, model, ckpt in models:
      for lam in LAMBDAS:
        for T in SELECTION_TEMPS:
          self.add('scripts/bct/eval_gsm8k.sh',
                   f'gsm8k_heldout/{bb}/{model}/{run_name("bctl", T, 1, lam)}',
                   CKPT_PATH=ckpt, BACKBONE=BACKBONES[bb][0], TFT=1,
                   METHOD='bctl', T=num(T), LAMBDA=num(lam), SEED=1,
                   EVAL_SET='tinygsm_heldout', HELDOUT_DIR=self.heldout)

  def lambda_sudoku(self):
    for lam in LAMBDAS:
      for T in SELECTION_TEMPS:
        self.sudoku('tft', 'bctl', T, 1, lam, heldout=True)

  def dlm(self):
    for model in ('mdlm', 'duo'):
      for T in TEMPS:
        self.add('scripts/bct/eval_gsm8k_dlm.sh',
                 f'gsm8k_dlm/{model}/T{num(T)}_seed1', MODEL=model,
                 CKPT_PATH=os.path.join(self.ckpt, 's-flm', 'tinygsm',
                                        f'{model}.ckpt'),
                 T=num(T), SEED=1)

  def top1(self):
    for bb in BACKBONES:
      self.add('scripts/bct/eval_gsm8k.sh', f'gsm8k/{bb}/released/top1_seed1',
               CKPT_PATH=self.gsm8k_ckpt(bb, 'released'),
               BACKBONE=BACKBONES[bb][0], METHOD='top1', SEED=1)


TARGETS = {'gsm8k_main': 'gsm8k_main', 'passk': 'passk', 'sudoku': 'sudoku_all',
           'lambda_gsm8k': 'lambda_gsm8k', 'lambda_sudoku': 'lambda_sudoku',
           'dlm': 'dlm', 'top1': 'top1'}


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawTextHelpFormatter)
  p.add_argument('result', choices=list(TARGETS) + ['all'])
  p.add_argument('--ckpt-dir', default='checkpoints',
                 help='holds s-flm/ (jdeschena/s-flm) and bct/ (daisuk30ba/BCT)')
  p.add_argument('--out', default='outputs')
  p.add_argument('--heldout-dir', default='outputs/tinygsm_heldout')
  p.add_argument('--dit-tft500-ckpt', default=None)
  a = p.parse_args()
  lister = Lister(a.ckpt_dir, a.out, a.heldout_dir, a.dit_tft500_ckpt)
  for name in (TARGETS if a.result == 'all' else [a.result]):
    getattr(lister, TARGETS[name])()
  print('\n'.join(lister.lines))


if __name__ == '__main__':
  main()
