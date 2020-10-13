#!/bin/bash
CONFIG_ROOT=/home/elodie/Det3D/examples/cbgs/configs/
# NUSC_SECOND_8_20200826-214223
# 20200826
# CONFIG=$CONFIG_ROOT/0826Nuscenes_all_vfev3_spmiddlefhd_rpn1_mghead_syncbn_freatures3.py
# WORK_DIR=/home/elodie/det3D_Output/NUSC_SECOND_8_20200826-214223
# CHECKPOINT=$WORK_DIR/epoch_20.pth

# 20200909 
# CONFIG=$CONFIG_ROOT/0904Nuscenes_all_vfev3_spmiddlefhd_rpn1_mghead_syncbn_freatures3.py
# WORK_DIR=/home/elodie/det3D_Output/NUSC_SECOND_9_20200909-224301
# CHECKPOINT=$WORK_DIR/epoch_20.pth

# 20200915
CONFIG=$CONFIG_ROOT/1003nusc_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py
# WORK_DIR=/home/elodie/det3D_Output/NUSC_SECOND_9_20200918-210040
WORK_DIR=/home/elodie/det3D_Output/NUSC_CBGS_2_20201003-185731
CHECKPOINT=$WORK_DIR/epoch_20.pth
TXTRESULT=False
# Test
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch \
    --nproc_per_node=2 \
    /home/elodie/Det3D/tools/dist_test.py \
    $CONFIG \
    --work_dir=$WORK_DIR \
    --checkpoint=$CHECKPOINT \
    --txt_result=$TXTRESULT  \

