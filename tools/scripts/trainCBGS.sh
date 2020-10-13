#!/bin/bash
TASK_DESC=$1
DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
OUT_DIR=/home/elodie/det3D_Output

NUSC_CBGS_WORK_DIR=$OUT_DIR/NUSC_CBGS_$TASK_DESC\_$DATE_WITH_TIME

if [ ! $TASK_DESC ] 
then
    echo "TASK_DESC must be specified."
    echo "Usage: train.sh task_description"
    exit $E_ASSERT_FAILED
fi

#Nuscened
# 0706
WORK_DIR=/home/elodie/det3D_Output/NUSC_CBGS_2_20201003-185731
RESUME_PTH=$WORK_DIR/latest.pth
# CHECKPOINT_PTH=/home/elodie/det3D_Output/NUSC_SECOND_9_20200923-175601/epoch_10.pth
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch \
    --nproc_per_node=2 ./tools/train.py \
    examples/cbgs/configs/1003nusc_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py \
    --work_dir=$WORK_DIR \
    --resume_from=$RESUME_PTH

