#!/bin/bash
#PBS -P gxx00000
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=4:00:00
#PBS -j oe
# Example job: GSM8K evaluation on one ABCI H200 node (8 GPUs).
# Variables are passed with qsub -v; RTYPE must be repeated because -v on the
# command line replaces the one in the header.  For example, BCT-P at
# T = 0.4 on the released S-arch checkpoint:
#
#   qsub -P <your group> -v RTYPE=rt_HF,BACKBONE=sphere-arch,MODEL=released,METHOD=bctp,T=0.4,SEED=1 \
#        scripts/abci_example/eval_gsm8k.sh
#
# MODEL is released, cont or tft; the checkpoint and the output directory are
# chosen from it following the layout used by scripts/bct/summarize.py.
set -euo pipefail
cd "${PBS_O_WORKDIR}"
source scripts/abci_example/env.sh
case "${BACKBONE}" in
  sphere-arch) BB=sarch; NAME=sphere_arch_truncated_adaptive_no_renorm ;;
  sphere-dit)  BB=sdit;  NAME=sphere_dit_truncated_adaptive_no_renorm ;;
esac
case "${MODEL}" in
  released) CKPT_PATH="${CKPT}/s-flm/tinygsm/sfm/${NAME}.ckpt" ;;
  cont)     CKPT_PATH="${CKPT}/bct/tinygsm/sfm/${NAME}_cont.ckpt" ;;
  tft)      CKPT_PATH="${CKPT}/bct/tinygsm/sfm/${NAME}_tft.ckpt"; export TFT=1 ;;
esac
LAMBDA="${LAMBDA:-1.0}"
RUN="${METHOD}_T${T}"
if [ "${METHOD}" = "bctl" ]; then RUN="${RUN}_l${LAMBDA}"; fi
export CKPT_PATH BACKBONE METHOD T SEED LAMBDA
export OUTPUT_DIR="${OUT}/gsm8k/${BB}/${MODEL}/${RUN}_seed${SEED}"
scripts/abci_example/run.sh bash scripts/bct/eval_gsm8k.sh
