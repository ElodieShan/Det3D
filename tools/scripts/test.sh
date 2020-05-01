#!/bin/bash
CONFIG=$1
WORK_DIR=$2
CHECKPOINT=$3
TXTRESULT=$4
# Test
python3 -m torch.distributed.launch \
    --nproc_per_node=2 \
    ./tools/dist_test.py \
    $CONFIG \
    --work_dir=$WORK_DIR \
    --checkpoint=$CHECKPOINT \
    --txt_result=$TXTRESULT  \

