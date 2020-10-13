#!/bin/bash
CONFIG_ROOT=/home/elodie/Det3D/examples/cbgs/configs/


# 20200915
CONFIG=$CONFIG_ROOT/0930nusc_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py
WORK_DIR=/home/elodie/det3D_Output/NUSC_CBGS_2_20200930-221140
# CHECKPOINT=$WORK_DIR/epoch_20.pth
TXTRESULT=False
# Test
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch \
    --nproc_per_node=1 \
    /home/elodie/Det3D/tools/dist_test_multiepoch.py \
    $CONFIG \
    --work_dir=$WORK_DIR \
    --epoch_end=20 \
    --dataset_type=KittiDataset \
    --anno_type=train
