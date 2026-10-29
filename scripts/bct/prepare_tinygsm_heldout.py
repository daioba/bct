#!/usr/bin/env python
"""Build the TinyGSM held-out problems used to select lambda of BCT-L.

The candidates are 768 sequences of the TinyGSM validation split (the
`val_ratio=0.01, val_seed=42` split held out from training), drawn with a
fixed permutation.  A candidate is excluded if
  (1) its reference program does not return a number in the GSM8K grader's
      sandbox, or the grader does not accept the reference program itself;
  (2) its question matches a GSM8K test question (whitespace/case normalised);
  (3) its prompt tokens exactly match a prompt of the TinyGSM training split.
The label of a kept problem is the value returned by its reference program.

The output directory holds `records.json`, `manifest.json` and a dataset
cache in the format of the GSM8K test cache, so that
`scripts/bct/eval_gsm8k.sh` evaluates on it with EVAL_SET=tinygsm_heldout.

  python scripts/bct/prepare_tinygsm_heldout.py \
      --out-dir <empty dir> --cache-dir <data.cache_dir of TinyGSM>
"""
import argparse
import contextlib
import hashlib
import io
import json
import multiprocessing as mp
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
  os.path.abspath(__file__))))
sys.path.insert(0, REPO)

TOKENIZER = 'HuggingFaceTB/SmolLM-135M'
BLOCK = 512
TIMEOUT_S = 5.0
P1 = np.uint64(1000003)


def _yaml_separator(name):
  """Read the separator from the Hydra data config."""
  from omegaconf import OmegaConf
  return OmegaConf.load(
    os.path.join(REPO, 'configs', 'data', f'{name}.yaml')).separator


def _data_config(cache_dir):
  from omegaconf import OmegaConf
  # The same data arguments as the TFT continuation
  # (scripts/bct/train_tinygsm.sh); they determine the cache name.
  return OmegaConf.create({
    'data': {'train': 'tiny_gsm', 'valid': 'tiny_gsm',
             'tokenizer_name_or_path': TOKENIZER, 'cache_dir': cache_dir,
             'wrap': False, 'streaming': False, 'insert_train_eos': True,
             'insert_valid_eos': True, 'separator': _yaml_separator('tiny-gsm'),
             'train_on_prompt': False, 'train_on_pad': True,
             'val_ratio': 0.01, 'val_seed': 42, 'filter_too_long': True},
    'model': {'length': BLOCK},
    'loader': {'num_workers': 4}})


def candidate_indices(n_total, skip, n_candidates, perm_seed):
  """Validation indices of the candidates, in a fixed random order."""
  import torch
  g = torch.Generator().manual_seed(perm_seed)
  perm = torch.randperm(n_total, generator=g)
  perm = perm[perm >= skip]
  return perm[:n_candidates].tolist()


def reference_gold(code):
  """Run the reference program in the grader's sandbox; return (gold, reason)."""
  import sandbox_gsm8k as sb
  src = sb._extract_code(code)
  try:
    with sb._time_limit(TIMEOUT_S):
      ns = sb._safe_exec_no_timer(src)
      fn = ns.get('simple_math_problem', None)
      if fn is None:
        return None, 'no_simple_math_problem'
      out_buf = io.StringIO()
      with contextlib.redirect_stdout(out_buf):
        with contextlib.redirect_stderr(io.StringIO()):
          out = fn()
  except Exception:      # noqa: BLE001
    return None, 'exec_error'
  except BaseException:  # noqa: B036 - _Timeout
    return None, 'timeout'
  val = sb._to_number(out)
  if val is None:
    val = sb._extract_text_answer(out_buf.getvalue())
  if val is None:
    return None, 'non_numeric'
  gt = f'#### {val}'
  if not sb.evaluate_samples(code, gt, timeout_s=TIMEOUT_S):
    return None, 'self_check_failed'
  return val, None


def _norm_text(s):
  return re.sub(r'\s+', ' ', s).strip().lower()


def _hash_rows(ids, plen):
  """ids [B, L] int64, plen [B] -> (h1 [B] uint64) over ids[:plen]."""
  L = ids.shape[1]
  with np.errstate(over='ignore'):
    w = np.cumprod(np.full(L, P1, dtype=np.uint64), dtype=np.uint64)
    m = (np.arange(L)[None, :] < plen[:, None]).astype(np.uint64)
    return ((ids.astype(np.uint64) + np.uint64(1)) * w[None, :] * m).sum(
      axis=1, dtype=np.uint64)


def _train_shard_worker(args):
  """Return the rows of one arrow shard whose prompt equals a candidate's."""
  path, targets = args
  import pyarrow as pa
  with pa.memory_map(path, 'r') as src:
    try:
      table = pa.ipc.open_stream(src).read_all()
    except pa.ArrowInvalid:
      table = pa.ipc.open_file(src).read_all()
  n = table.num_rows
  ids_col = table.column('input_ids').combine_chunks()
  plen = table.column('prompt_len').to_numpy()
  flat = ids_col.flatten().to_numpy(zero_copy_only=False)
  assert flat.size == n * BLOCK, (path, flat.size, n)
  ids = flat.reshape(n, BLOCK)
  h = np.concatenate([_hash_rows(ids[s:s + 16384], plen[s:s + 16384])
                      for s in range(0, n, 16384)])
  tgt_h = {int(k.split(':')[1]) for k in targets}
  hits = []
  cand = np.nonzero(np.isin(h, np.fromiter(tgt_h, dtype=np.uint64)))[0]
  for r in cand.tolist():
    key = f'{int(plen[r])}:{int(h[r])}'
    if key in targets:
      toks = ids[r, :plen[r]].tolist()
      for vidx, ref in targets[key]:
        if toks == ref:
          hits.append((vidx, os.path.basename(path), r))
  return os.path.basename(path), n, hits


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--out-dir', required=True)
  p.add_argument('--cache-dir', default=os.environ.get('CACHE_DIR'),
                 help='data.cache_dir holding the TinyGSM cache')
  p.add_argument('--skip', type=int, default=64)
  p.add_argument('--n-candidates', type=int, default=768)
  p.add_argument('--perm-seed', type=int, default=2026)
  p.add_argument('--workers', type=int, default=16)
  p.add_argument('--skip-train-check', action='store_true')
  args = p.parse_args()

  import datasets
  import dataloader
  import utils

  if os.path.exists(args.out_dir) and os.listdir(args.out_dir):
    raise SystemExit(f'{args.out_dir} is not empty; not overwriting.')
  os.makedirs(args.out_dir, exist_ok=True)

  config = _data_config(args.cache_dir)
  tokenizer = dataloader.get_tokenizer(config)
  BOS, EOS = tokenizer.bos_token_id, tokenizer.eos_token_id
  valid = dataloader.get_dataset(config, tokenizer, mode='valid')
  valid = valid.with_format(None)
  n_valid = len(valid)
  candidates = candidate_indices(n_valid, args.skip, args.n_candidates,
                                 args.perm_seed)

  # Extract prompt and reference program.
  with open(os.path.join(REPO, 'data', 'gsm8k_test.json')) as f:
    gsm8k_test_norm = {_norm_text(r['prompt']) for r in json.load(f)}
  sep_train = config.data.separator
  sep_eval = _yaml_separator('gsm8k-test')
  sep_ids = tokenizer(sep_train, add_special_tokens=False).input_ids
  print(f'[sep] tiny-gsm={sep_train!r} gsm8k-test={sep_eval!r} ids={sep_ids} '
        f'BOS={BOS} EOS={EOS} PAD={tokenizer.pad_token_id}', flush=True)
  if sep_train != sep_eval:
    raise SystemExit('training and evaluation use different separators')
  items = []
  for vidx in candidates:
    row = valid[vidx]
    ids = list(row['input_ids'])
    pl = int(row['prompt_len'])
    mask = list(row['attention_mask'])
    # train_on_prompt=False: the loss mask starts at prompt_len.
    assert mask.index(1) == pl, (vidx, mask.index(1), pl)
    assert ids[0] == BOS and ids[pl - len(sep_ids):pl] == sep_ids, (
      vidx, pl, ids[:4], ids[max(pl - 6, 0):pl + 3], sep_ids, BOS)
    prompt_ids = ids[:pl]
    try:
      end = ids.index(EOS, pl)
    except ValueError:
      end = len(ids)
    question = tokenizer.decode(ids[1:pl - len(sep_ids)])
    code = tokenizer.decode(ids[pl:end])
    items.append({'valid_index': vidx, 'prompt_ids': prompt_ids,
                  'question': question, 'code': code})
  print(f'[heldout] {len(items)} candidates', flush=True)

  # Exclusion rules (1) and (2).
  reasons = {}
  for it in items:
    gold, why = reference_gold(it['code'])
    it['gold'] = gold
    if why is not None:
      reasons.setdefault(why, []).append(it['valid_index'])
      it['excluded'] = why
    elif _norm_text(it['question']) in gsm8k_test_norm:
      reasons.setdefault('gsm8k_test_overlap', []).append(it['valid_index'])
      it['excluded'] = 'gsm8k_test_overlap'

  # Exclusion rule (3): prompt identical to a training prompt.
  train_check = {'performed': False}
  if not args.skip_train_check:
    tok_tag = TOKENIZER.replace('/', '__')
    root = (f'{args.cache_dir}/tiny_gsm_bs{BLOCK}_answer_only_filtered'
            f'_train_on_pad_{tok_tag}')
    with open(os.path.join(root, 'train', 'state.json')) as f:
      shards = [os.path.join(root, 'train', d['filename'])
                for d in json.load(f)['_data_files']]
    targets = {}
    for it in items:
      ids = np.asarray(it['prompt_ids'], dtype=np.int64)[None, :]
      pad = np.zeros((1, BLOCK - ids.shape[1]), dtype=np.int64)
      h = int(_hash_rows(np.concatenate([ids, pad], 1),
                         np.array([ids.shape[1]]))[0])
      key = f'{ids.shape[1]}:{h}'
      targets.setdefault(key, []).append((it['valid_index'], it['prompt_ids']))
    with mp.Pool(args.workers) as pool:
      res = pool.map(_train_shard_worker, [(s, targets) for s in shards])
    n_train = sum(n for _, n, _ in res)
    hits = [h for _, _, hs in res for h in hs]
    hit_idx = sorted({h[0] for h in hits})
    train_check = {'performed': True, 'n_train_rows': n_train,
                   'n_shards': len(shards),
                   'n_heldout_with_train_duplicate': len(hit_idx),
                   'n_train_duplicate_rows': len(hits),
                   'examples': hits[:20]}
    print(f'[train check] train rows={n_train} shards={len(shards)} '
          f'held-out prompts duplicated in train: {len(hit_idx)}', flush=True)
    for it in items:
      if it['valid_index'] in hit_idx and 'excluded' not in it:
        reasons.setdefault('tft_train_prompt_duplicate', []).append(
          it['valid_index'])
        it['excluded'] = 'tft_train_prompt_duplicate'

  kept = [it for it in items if 'excluded' not in it]
  print(f'[heldout] kept {len(kept)} / {len(items)}; excluded '
        f'{ {k: len(v) for k, v in reasons.items()} }', flush=True)

  # Save the records and a cache in the format of the GSM8K test cache.
  records = [{'prompt': it['question'],
              'response_ground_truth': f"#### {it['gold']}",
              'valid_index': it['valid_index'],
              'gold': it['gold'], 'code': it['code']} for it in kept]
  rec_path = os.path.join(args.out_dir, 'records.json')
  with open(rec_path, 'w') as f:
    json.dump(records, f, indent=1)
  n_retok_diff = 0
  for it in kept:
    q = tokenizer(it['question'].strip(), add_special_tokens=False).input_ids
    if [BOS] + q + sep_ids != it['prompt_ids']:
      n_retok_diff += 1
  tok_tag = config.data.tokenizer_name_or_path.replace('/', '__')
  save_dir = f'{args.out_dir}/gsm8k_test_{tok_tag}_with_text'
  ds = datasets.Dataset.from_list([
    {'prompt': it['question'],
     'response_ground_truth': f"#### {it['gold']}",
     'input_ids': it['prompt_ids'],
     'answer': tokenizer(it['code'].strip(),
                         add_special_tokens=False).input_ids}
    for it in kept])
  ds.save_to_disk(save_dir)
  assert utils.fsspec_exists(save_dir)

  # Read it back through the path used by main._sample_gsm8k.
  cfg2 = _data_config(args.out_dir)
  cfg2.data.valid = 'gsm8k_test'
  back = dataloader.get_dataset(cfg2, tokenizer, mode='valid')
  assert len(back) == len(kept)
  for i in (0, len(kept) - 1):
    assert list(back[i]['input_ids']) == kept[i]['prompt_ids']
    assert back[i]['prompt'] == kept[i]['question']

  with open(rec_path, 'rb') as f:
    sha = hashlib.sha256(f.read()).hexdigest()
  manifest = {
    'definition': 'TinyGSM validation split (val_ratio=0.01, val_seed=42), '
                  f'{args.n_candidates} candidates (skip<{args.skip}, '
                  f'perm_seed {args.perm_seed})',
    'n_valid_total': n_valid, 'n_candidates': len(items), 'n_kept': len(kept),
    'excluded': {k: v for k, v in reasons.items()},
    'n_excluded': {k: len(v) for k, v in reasons.items()},
    'train_check': train_check,
    'gsm8k_test_overlap_rule': 'whitespace/case-normalized exact match',
    'n_retokenization_differs': n_retok_diff,
    'input_ids': 'original TinyGSM prompt tokens [BOS] question sep',
    'kept_valid_indices': [it['valid_index'] for it in kept],
    'records_sha256': sha, 'cache': save_dir,
    'gold_is_int_frac': float(np.mean([isinstance(it['gold'], int)
                                       for it in kept])),
  }
  with open(os.path.join(args.out_dir, 'manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=1)
  print(json.dumps({k: v for k, v in manifest.items()
                    if k not in ('kept_valid_indices', 'excluded')},
                   indent=1))
  print(f'[heldout] wrote {save_dir}')


if __name__ == '__main__':
  main()
