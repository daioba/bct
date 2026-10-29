#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/tinygsm/flm_caps}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-8}"

cd "${REPO_ROOT}"

python -u -m main \
    data=tiny-gsm \
    data.cache_dir="${CACHE_DIR}" \
    data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    data.wrap=False \
    data.train_on_prompt=False \
    data.train_on_pad=True \
    data.filter_too_long=True \
    model=small-flm \
    model.length=512 \
    model.softcap=50.0 \
    noise=log-linear \
    algo=flm \
    algo.cap_value=30.0 \
    sampler=flm_euler \
    loader.global_batch_size=512 \
    loader.batch_size=64 \
    loader.eval_batch_size=64 \
    loader.num_workers=16 \
    eval.generate_samples=False \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    trainer.val_check_interval=10_000 \
    trainer.max_steps=250_000 \
    trainer.limit_val_batches=500 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=10_000 \
    hydra.run.dir="${OUTPUT_DIR}"
