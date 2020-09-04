#!/bin/bash
CONFIG=$1
WORK_DIR=$2
CHECKPOINT=$3
TXTRESULT=$4
# Test
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch \
    --nproc_per_node=2 \
    ./tools/dist_test.py \
    $CONFIG \
    --work_dir=$WORK_DIR \
    --checkpoint=$CHECKPOINT \
    --txt_result=$TXTRESULT  \

# torch.distributed.launch --nproc_per_node=2 ./tools/train.py examples/second/configs/Nuscenes_all_vfev3_spmiddlefhd_rpn1_mghead_syncbn_freatures3.py --work_dir=$NUSC_SECOND_WORK_DIR