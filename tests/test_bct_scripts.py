"""The selection rule for lambda and the Pass@K estimator of
`scripts/bct/summarize.py`.  CPU only."""
import itertools
import math

import pytest

from scripts.bct import summarize as sm


def test_lambda_grid_and_selection_temperatures():
  assert list(sm.LAMBDAS) == [0.0, 0.25, 0.5, 0.75, 1.0]
  assert list(sm.SELECTION_TEMPS) == [0.4, 0.2, 0.1]


def test_select_lambda_is_the_argmax():
  assert sm.select_lambda({0.0: 10, 0.25: 30, 0.5: 20, 0.75: 5, 1.0: 1}) == 0.25


def test_select_lambda_breaks_exact_ties_towards_larger_lambda():
  assert sm.select_lambda({0.0: 30, 0.25: 30, 0.5: 29, 0.75: 30, 1.0: 2}) == 0.75
  assert sm.select_lambda({0.0: 7, 0.25: 7, 0.5: 7, 0.75: 7, 1.0: 7}) == 1.0


def _brute(n, c, k):
  gens = [1] * c + [0] * (n - c)
  subsets = list(itertools.combinations(range(n), k))
  return sum(1 for s in subsets if any(gens[i] for i in s)) / len(subsets)


@pytest.mark.parametrize('c', range(9))
@pytest.mark.parametrize('k', sm.PASSK_KS)
def test_pass_at_k_matches_subset_enumeration(c, k):
  assert sm.pass_at_k(8, c, k) == pytest.approx(_brute(8, c, k), abs=1e-12)


@pytest.mark.parametrize('c', range(9))
def test_pass_at_k_identities(c):
  assert sm.pass_at_k(8, c, 1) == pytest.approx(c / 8, abs=1e-12)
  assert sm.pass_at_k(8, c, 8) == (1.0 if c > 0 else 0.0)
  vals = [sm.pass_at_k(8, c, k) for k in sm.PASSK_KS]
  assert all(a <= b + 1e-12 for a, b in zip(vals, vals[1:]))


def test_pass_at_k_short_branch_and_product_form():
  assert sm.pass_at_k(8, 5, 4) == 1.0
  assert sm.pass_at_k(8, 7, 2) == 1.0
  for c in range(8):
    for k in sm.PASSK_KS:
      if 8 - c < k:
        continue
      prod = math.prod(1.0 - k / i for i in range(8 - c + 1, 9))
      assert sm.pass_at_k(8, c, k) == pytest.approx(1.0 - prod, abs=1e-12)


def test_pass_at_k_rejects_bad_counts():
  for bad in ((8, 9, 1), (8, -1, 1), (8, 3, 0), (8, 3, 9)):
    with pytest.raises(ValueError):
      sm.pass_at_k(*bad)


def test_passk_generation_seeds_give_disjoint_rank_seeds():
  # Rank r of an evaluation run is seeded with seed + r, r = 0..7.
  assert list(sm.PASSK_SEEDS) == [1, 11, 21, 31, 41, 51, 61, 71]
  per_rank = [s + r for s in sm.PASSK_SEEDS for r in range(8)]
  assert len(set(per_rank)) == len(per_rank)


def test_problem_order_is_rank_interleaved():
  order = sm.problem_order(1319, 8)
  assert sorted(order) == list(range(1319))
  assert order[:3] == [0, 8, 16]
  assert order[165] == 1   # rank 0 holds ceil(1319 / 8) = 165 problems


def test_run_names():
  assert sm.run_name('bctp', 0.4, 11) == 'bctp_T0.4_seed11'
  assert sm.run_name('bctl', 0.1, 1, 0.25) == 'bctl_T0.1_l0.25_seed1'
  assert sm.run_name('bctl', 0.2, 1, 1.0) == 'bctl_T0.2_l1.0_seed1'
  assert sm.run_name('standard', 1.0, 21) == 'standard_T1.0_seed21'


def _listed(result, **kw):
  from scripts.bct import list_runs as lr
  lister = lr.Lister('CK', 'OUT', 'HELDOUT', kw.get('tft500'))
  for name in (lr.TARGETS if result == 'all' else [result]):
    getattr(lister, lr.TARGETS[name])()
  return lister.lines


@pytest.mark.parametrize('result,n', [
  ('gsm8k_main', 312), ('passk', 128), ('sudoku', 168), ('lambda_gsm8k', 30),
  ('lambda_sudoku', 15), ('dlm', 16), ('top1', 2), ('all', 623)])
def test_list_runs_counts(result, n):
  assert len(_listed(result)) == n


def test_list_runs_names_match_summarize():
  from scripts.bct import list_runs as lr
  for args in [('bctp', 0.4, 11), ('bctl', 0.1, 1, 0.25), ('bctl', 0.2, 1, 1.0),
               ('standard', 1.0, 21)]:
    assert lr.run_name(*args) == sm.run_name(*args)


def test_list_runs_cover_the_cells_summarize_reads():
  outs = {line.split('OUTPUT_DIR=OUT/')[1].split()[0] for line in _listed('all')}
  for bb, lam in (('sarch', 0.5), ('sdit', 0.25)):
    for _, model, method in sm.GSM8K_COLUMNS:
      for T in sm.TEMPS:
        for d in sm.gsm8k_dirs('', bb, model, method, T, sm.SEEDS, lam):
          assert d.lstrip('/') in outs, d
  for _, model, method in sm.SUDOKU_COLUMNS:
    for T in sm.TEMPS:
      for s in sm.SEEDS:
        lam = 1.0 if method == 'bctl' else None
        assert f'sudoku_hard/{model}/{sm.run_name(method, T, s, lam)}' in outs
