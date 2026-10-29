#!/bin/bash
#PBS -P gxx00000
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -j oe
# Run one line of a command list written by scripts/bct/list_runs.py as a job.
#
#   python scripts/bct/list_runs.py gsm8k_main --ckpt-dir "${CKPT}" --out "${OUT}" > runs.txt
#   qsub -P <group> -v LIST=$PWD/runs.txt,LINE=1 scripts/abci_example/run_list.sh
#
# One GSM8K run takes about 20 minutes on one H200 node; one Sudoku run takes
# one or two minutes.
set -euo pipefail
cd "${PBS_O_WORKDIR}"
source scripts/abci_example/env.sh
CMD="$(sed -n "${LINE}p" "${LIST}")"
echo "${CMD}"
scripts/abci_example/run.sh bash -c "${CMD}"
