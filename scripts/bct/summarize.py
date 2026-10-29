#!/usr/bin/env python
"""Aggregate evaluation runs into the tables of the paper.

All subcommands read `results.json` files written by the evaluation scripts in
`scripts/bct/`, laid out under one root directory as described in the README:

  <root>/gsm8k/{sarch,sdit}/{released,cont,tft}/<method>_T<T>[_l<lambda>]_seed<S>/
  <root>/gsm8k_heldout/{sarch,sdit}/<model>/bctl_T<T>_l<lambda>_seed1/
  <root>/gsm8k_dlm/{mdlm,duo}/T<T>_seed<S>/
  <root>/sudoku_hard/{base,cont,tft}/<method>_T<T>[_l<lambda>]_seed<S>/
  <root>/sudoku_hard_heldout/tft/bctl_T<T>_l<lambda>_seed1/

Subcommands:
  gsm8k    GSM8K accuracy, mean and sample SD over generation seeds (main
           table and figure).  At T = 1 every BCT correction is the identity, so the
           T = 1 entry of every method is the `standard_T1.0` run of the same
           checkpoint.
  sudoku   Sudoku-Hard exact solve rate over generation seeds.
  dlm      Standard temperature scaling on MDLM / Duo (seed 1).
  lambda   Held-out selection of lambda for BCT-L: the score of a
           lambda is the mean held-out accuracy over T in {0.4, 0.2, 0.1};
           the selected lambda maximises it, and an exact tie goes to the
           larger lambda.
  passk    Pass@K on GSM8K from eight generations per problem (generation
           seeds 1, 11, ..., 71), with problem-level bootstrap intervals.
           This regrades every generation (it executes the generated
           programs), so run it where you run the evaluation.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
  os.path.abspath(__file__))))
sys.path.insert(0, REPO)

TEMPS = (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
SEEDS = (1, 11, 21)
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
SELECTION_TEMPS = (0.4, 0.2, 0.1)
PASSK_SEEDS = tuple(1 + 10 * j for j in range(8))
PASSK_KS = (1, 2, 4, 8)


def tname(x):
  """1.0 -> '1.0', 0.4 -> '0.4', 0.25 -> '0.25' (as in the run names)."""
  return f'{x:.1f}' if round(x, 1) == x else f'{x:g}'


def run_name(method, T, seed, lam=None):
  name = f'{method}_T{tname(T)}'
  if lam is not None:
    name += f'_l{tname(lam)}'
  return f'{name}_seed{seed}'


def load(path):
  with open(os.path.join(path, 'results.json')) as f:
    return json.load(f)


def counts(paths):
  """[(num_correct, num_total)] for the given run directories, or None."""
  out = []
  for p in paths:
    if not os.path.exists(os.path.join(p, 'results.json')):
      return None
    r = load(p)
    out.append((int(r['num_correct']), int(r['num_total'])))
  if len({n for _, n in out}) != 1:
    raise SystemExit(f'runs have different numbers of problems: {paths}')
  return out


def mean_sd(cs):
  acc = [100.0 * c / n for c, n in cs]
  sd = float(np.std(acc, ddof=1)) if len(acc) > 1 else 0.0
  return float(np.mean(acc)), sd


def fmt(ms):
  return '--' if ms is None else f'{ms[0]:.1f} ± {ms[1]:.1f}'


def markdown(header, rows):
  lines = ['| ' + ' | '.join(header) + ' |',
           '|' + '---|' * len(header)]
  lines += ['| ' + ' | '.join(r) + ' |' for r in rows]
  return '\n'.join(lines)


# ── GSM8K ─────────────────────────────────────────────────────────────────
GSM8K_COLUMNS = (
  ('TFT BCT-L', 'tft', 'bctl'),
  ('Cont. Standard', 'cont', 'standard'),
  ('Cont. BCT-E', 'cont', 'bcte'),
  ('Cont. BCT-P', 'cont', 'bctp'),
  ('Standard', 'released', 'standard'),
  ('BCT-E', 'released', 'bcte'),
  ('BCT-P', 'released', 'bctp'),
)


def gsm8k_dirs(root, bb, model, method, T, seeds, lam):
  base = os.path.join(root, 'gsm8k', bb, model)
  if T == 1.0:
    return [os.path.join(base, f'standard_T1.0_seed{s}') for s in seeds]
  lam_ = lam if method == 'bctl' else None
  return [os.path.join(base, run_name(method, T, s, lam_)) for s in seeds]


def cmd_gsm8k(a):
  out = {}
  lams = {'sarch': a.lambda_sarch, 'sdit': a.lambda_sdit}
  for bb in ('sarch', 'sdit'):
    rows = []
    for T in TEMPS:
      row = [tname(T)]
      for label, model, method in GSM8K_COLUMNS:
        cs = counts(gsm8k_dirs(a.root, bb, model, method, T, a.seeds,
                               lams[bb]))
        ms = None if cs is None else mean_sd(cs)
        row.append(fmt(ms))
        out[f'{bb}|{label}|T={tname(T)}'] = (
          None if cs is None else {'correct': [c for c, _ in cs],
                                   'mean_pct': ms[0], 'sd_pct': ms[1]})
      rows.append(row)
    print(f'\nGSM8K, {bb} (lambda = {lams[bb]}, seeds {list(a.seeds)})')
    print(markdown(['T'] + [c[0] for c in GSM8K_COLUMNS], rows))
  return out


# ── Sudoku-Hard ───────────────────────────────────────────────────────────
SUDOKU_COLUMNS = (
  ('Standard', 'base', 'standard'),
  ('BCT-E', 'base', 'bcte'),
  ('BCT-P', 'base', 'bctp'),
  ('Cont. Standard', 'cont', 'standard'),
  ('Cont. BCT-E', 'cont', 'bcte'),
  ('Cont. BCT-P', 'cont', 'bctp'),
  ('TFT BCT-L', 'tft', 'bctl'),
)


def cmd_sudoku(a):
  out = {}
  rows = []
  for T in TEMPS:
    row = [tname(T)]
    for label, model, method in SUDOKU_COLUMNS:
      lam = a.lam if method == 'bctl' else None
      dirs = [os.path.join(a.root, 'sudoku_hard', model,
                           run_name(method, T, s, lam)) for s in a.seeds]
      cs = counts(dirs)
      ms = None if cs is None else mean_sd(cs)
      row.append(fmt(ms))
      out[f'{label}|T={tname(T)}'] = (
        None if cs is None else {'correct': [c for c, _ in cs],
                                 'mean_pct': ms[0], 'sd_pct': ms[1]})
    rows.append(row)
  print(f'\nSudoku-Hard exact solve rate (lambda = {a.lam}, '
        f'seeds {list(a.seeds)})')
  print(markdown(['T'] + [c[0] for c in SUDOKU_COLUMNS], rows))
  return out


# ── MDLM / Duo ────────────────────────────────────────────────────────────
def cmd_dlm(a):
  out = {}
  rows = []
  cols = (('MDLM', lambda T: os.path.join(a.root, 'gsm8k_dlm', 'mdlm',
                                          f'T{tname(T)}_seed{a.seed}')),
          ('Duo', lambda T: os.path.join(a.root, 'gsm8k_dlm', 'duo',
                                         f'T{tname(T)}_seed{a.seed}')),
          # S-FLM with standard tempering, for reference (Table 1, seed 1).
          ('S-FLM S-arch', lambda T: gsm8k_dirs(a.root, 'sarch', 'released',
                                                'standard', T, [a.seed],
                                                None)[0]),
          ('S-FLM DiT', lambda T: gsm8k_dirs(a.root, 'sdit', 'released',
                                             'standard', T, [a.seed],
                                             None)[0]))
  for T in TEMPS:
    row = [tname(T)]
    for name, path_of in cols:
      cs = counts([path_of(T)])
      if cs is None:
        row.append('--')
        out[f'{name}|T={tname(T)}'] = None
        continue
      c, n = cs[0]
      row.append(f'{100.0 * c / n:.1f} ({c})')
      out[f'{name}|T={tname(T)}'] = {'correct': c, 'acc_pct': 100.0 * c / n}
    rows.append(row)
  print(f'\nStandard temperature scaling on MDLM / Duo (seed {a.seed})')
  print(markdown(['T'] + [c[0] for c in cols], rows))
  return out


# ── held-out selection of lambda ──────────────────────────────────────────
def select_lambda(correct_sum):
  """{lambda: summed correct answers} -> selected lambda (ties: larger)."""
  best = max(correct_sum.values())
  return max(lam for lam, v in correct_sum.items() if v == best)


def lambda_table(base, label):
  grid, sums, n_total = {}, {}, None
  for lam in LAMBDAS:
    sums[lam] = 0
    for T in SELECTION_TEMPS:
      cs = counts([os.path.join(base, run_name('bctl', T, 1, lam))])
      if cs is None:
        return None
      c, n = cs[0]
      n_total = n
      grid[(lam, T)] = c
      sums[lam] += c
  lam_star = select_lambda(sums)
  rows = []
  for lam in LAMBDAS:
    score = 100.0 * sums[lam] / (len(SELECTION_TEMPS) * n_total)
    mark = ' *' if lam == lam_star else ''
    rows.append([f'{tname(lam)}{mark}'] + [str(grid[(lam, T)])
                                      for T in SELECTION_TEMPS]
                + [f'{score:.2f}'])
  print(f'\n{label}: held-out lambda sweep (correct out of {n_total}); '
        f'selected lambda = {tname(lam_star)}')
  print(markdown(['lambda'] + [f'T={tname(T)}' for T in SELECTION_TEMPS]
                 + ['score (%)'], rows))
  return {'lambda_star': lam_star, 'n_total': n_total,
          'correct': {f'{tname(lam)}|T={tname(T)}': c
                      for (lam, T), c in grid.items()},
          'score_pct': {tname(lam): 100.0 * s / (len(SELECTION_TEMPS)
                                                 * n_total)
                        for lam, s in sums.items()}}


def cmd_lambda(a):
  out = {}
  if a.task == 'gsm8k':
    for bb in ('sarch', 'sdit'):
      root = os.path.join(a.root, 'gsm8k_heldout', bb)
      if not os.path.isdir(root):
        continue
      for model in sorted(os.listdir(root)):
        res = lambda_table(os.path.join(root, model), f'GSM8K {bb} {model}')
        if res is not None:
          out[f'{bb}|{model}'] = res
  else:
    res = lambda_table(os.path.join(a.root, 'sudoku_hard_heldout', 'tft'),
                       'Sudoku-Hard tft')
    if res is not None:
      out['tft'] = res
  return out


# ── Pass@K ────────────────────────────────────────────────────────────────
def pass_at_k(n, c, k):
  """Unbiased pass@k of one problem with c correct out of n generations."""
  if not 0 <= c <= n or not 1 <= k <= n:
    raise ValueError(f'pass_at_k(n={n}, c={c}, k={k})')
  if n - c < k:
    return 1.0
  return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def problem_order(n_total, world):
  """Dataset index of each record of results.json (rank-major)."""
  order = []
  for r in range(world):
    order += list(range(r, n_total, world))
  return order


def _grade(args):
  i, response, gt, timeout = args
  import warnings
  import sandbox_gsm8k
  warnings.filterwarnings('ignore', category=SyntaxWarning)
  return i, bool(sandbox_gsm8k.evaluate_samples(response, gt,
                                                timeout_s=timeout))


def graded_vector(run, world, timeout, workers):
  """Correctness per problem, in dataset order, regraded from the records."""
  import multiprocessing as mp
  r = load(run)
  recs = r['records']
  with mp.Pool(workers) as pool:
    res = dict(pool.map(_grade, [(i, x['response'], x['response_ground_truth'],
                                  timeout) for i, x in enumerate(recs)]))
  ok = np.array([res[i] for i in range(len(recs))], dtype=bool)
  if int(ok.sum()) != int(r['num_correct']):
    raise SystemExit(f'regrading {run} gives {int(ok.sum())} correct, '
                     f'results.json says {r["num_correct"]}')
  order = problem_order(len(recs), world)
  vec = np.zeros(len(recs), dtype=bool)
  vec[order] = ok
  prompts = [None] * len(recs)
  for pos, p in enumerate(order):
    prompts[p] = recs[pos]['prompt']
  return vec, prompts


def cmd_passk(a):
  sys.set_int_max_str_digits(0)
  methods = {
    'Cont. Standard': lambda T, s: os.path.join(
      a.root, 'gsm8k', 'sarch', 'cont',
      f'standard_T1.0_seed{s}' if T == 1.0 else run_name('standard', T, s)),
    'TFT BCT-L': lambda T, s: os.path.join(
      a.root, 'gsm8k', 'sarch', 'tft',
      f'standard_T1.0_seed{s}' if T == 1.0
      else run_name('bctl', T, s, a.lam)),
  }
  cvec = {}
  for name, path_of in methods.items():
    for T in TEMPS:
      vecs, ref_prompts = [], None
      for s in PASSK_SEEDS:
        v, prompts = graded_vector(path_of(T, s), a.world, a.timeout,
                                   a.workers)
        if ref_prompts is None:
          ref_prompts = prompts
        elif prompts != ref_prompts:
          raise SystemExit(f'problem order differs across seeds: {name} {T}')
        vecs.append(v)
      cvec[(name, T)] = np.sum(vecs, axis=0)
  n = len(PASSK_SEEDS)
  n_prob = len(next(iter(cvec.values())))
  rng = np.random.default_rng(0)
  boot_idx = rng.integers(0, n_prob, size=(a.bootstrap, n_prob))
  out, table = {}, {}
  for (name, T), c in cvec.items():
    for k in PASSK_KS:
      v = np.array([pass_at_k(n, int(x), k) for x in c])
      bm = v[boot_idx].mean(axis=1)
      table[(name, T, k)] = v
      out[f'{name}|T={tname(T)}|K={k}'] = {
        'pct': 100.0 * v.mean(),
        'ci95_pct': [100.0 * float(np.percentile(bm, 2.5)),
                     100.0 * float(np.percentile(bm, 97.5))]}
  for T in TEMPS:
    for k in PASSK_KS:
      dv = table[('TFT BCT-L', T, k)] - table[('Cont. Standard', T, k)]
      bm = dv[boot_idx].mean(axis=1)
      out[f'gap|T={tname(T)}|K={k}'] = {
        'pct': 100.0 * dv.mean(),
        'ci95_pct': [100.0 * float(np.percentile(bm, 2.5)),
                     100.0 * float(np.percentile(bm, 97.5))]}
  for name in methods:
    rows = [[tname(T)] + [f'{out[f"{name}|T={tname(T)}|K={k}"]["pct"]:.1f}'
                          for k in PASSK_KS] for T in TEMPS]
    print(f'\nPass@K (%), S-arch, {name}, n = {n} generations per problem')
    print(markdown(['T'] + [f'K={k}' for k in PASSK_KS], rows))
  return out


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawTextHelpFormatter)
  sub = p.add_subparsers(dest='cmd', required=True)
  g = sub.add_parser('gsm8k')
  g.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
  g.add_argument('--lambda-sarch', type=float, default=0.5)
  g.add_argument('--lambda-sdit', type=float, default=0.25)
  s = sub.add_parser('sudoku')
  s.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
  s.add_argument('--lam', type=float, default=1.0)
  d = sub.add_parser('dlm')
  d.add_argument('--seed', type=int, default=1)
  l = sub.add_parser('lambda')
  l.add_argument('--task', choices=['gsm8k', 'sudoku'], required=True)
  k = sub.add_parser('passk')
  k.add_argument('--lam', type=float, default=0.5)
  k.add_argument('--world', type=int, default=8,
                 help='number of GPUs the runs were sampled with')
  k.add_argument('--timeout', type=float, default=5.0)
  k.add_argument('--workers', type=int, default=16)
  k.add_argument('--bootstrap', type=int, default=10000)
  for q in (g, s, d, l, k):
    q.add_argument('--root', required=True)
    q.add_argument('--out', help='also write the numbers to this JSON file')
  a = p.parse_args()
  out = {'gsm8k': cmd_gsm8k, 'sudoku': cmd_sudoku, 'dlm': cmd_dlm,
         'lambda': cmd_lambda, 'passk': cmd_passk}[a.cmd](a)
  if a.out:
    with open(a.out, 'w') as f:
      json.dump(out, f, indent=1)


if __name__ == '__main__':
  main()
