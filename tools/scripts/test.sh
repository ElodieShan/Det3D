#!/bin/bash
CONFIG_ROOT=/home/elodie/Det3D/examples/second/configs/
CONFIG=$CONFIG_ROOT/1012kitti_all_vfev3_spmiddlefhd_rpn1_mghead_syncbn_freatures3_16lines.py
WORK_DIR=/home/elodie/det3D_Output/KITTI_SECOND_1_20201012-144552
CHECKPOINT=$WORK_DIR/epoch_100.pth
TXTRESULT=False
# Test
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch \
    --nproc_per_node=2 \
    /home/elodie/Det3D/tools/dist_test.py \
    $CONFIG \
    --work_dir=$WORK_DIR \
    --checkpoint=$CHECKPOINT \
    --txt_result=$TXTRESULT  \

