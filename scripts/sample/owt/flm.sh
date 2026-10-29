#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/checkpoints/owt/flm.ckpt}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_runs/owt/flm}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
STEPS="${STEPS:-32}"
NUM_SAMPLE_BATCHES="${NUM_SAMPLE_BATCHES:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"

cd "${REPO_ROOT}"

python -u -m main \
    mode=sample_eval \
    eval.checkpoint_path="${CKPT_PATH}" \
    eval.strict_loading=false \
    eval.compute_generative_perplexity=True \
    data=openwebtext-split \
    data.cache_dir="${CACHE_DIR}" \
    model=small-flm \
    noise=log-linear \
    algo=flm \
    sampler=flm_euler \
    sampler.steps="${STEPS}" \
    sampler.num_sample_batches="${NUM_SAMPLE_BATCHES}" \
    loader.eval_batch_size="${EVAL_BATCH_SIZE}" \
    loader.num_workers=4 \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    +wandb.offline=True \
    hydra.run.dir="${OUTPUT_DIR}"
