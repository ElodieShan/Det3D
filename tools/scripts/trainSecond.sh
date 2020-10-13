#!/bin/bash
TASK_DESC=$1
DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
OUT_DIR=/home/elodie/det3D_Output

NUSC_CBGS_WORK_DIR=$OUT_DIR/NUSC_CBGS_$TASK_DESC\_$DATE_WITH_TIME
NUSC_SECOND_WORK_DIR=$OUT_DIR/NUSC_SECOND_$TASK_DESC\_$DATE_WITH_TIME
KITTI_SECOND_WORK_DIR=$OUT_DIR/KITTI_SECOND_$TASK_DESC\_$DATE_WITH_TIME

LYFT_CBGS_WORK_DIR=$OUT_DIR/LYFT_CBGS_$TASK_DESC\_$DATE_WITH_TIME
SECOND_ALL_WORK_DIR=$OUT_DIR/SECOND_F3_all_$TASK_DESC\_$DATE_WITH_TIME
SECOND_CAR_WORK_DIR=$OUT_DIR/SECOND_car_$TASK_DESC\_$DATE_WITH_TIME
PP_WORK_DIR=$OUT_DIR/PointPillars_$TASK_DESC\_$DATE_WITH_TIME

if [ ! $TASK_DESC ] 
then
    echo "TASK_DESC must be specified."
    echo "Usage: train.sh task_description"
    exit $E_ASSERT_FAILED
fi

#Nuscened
# 0706
WORK_DIR=/home/elodie/det3D_Output/KITTI_SECOND_1_20201013-061950
RESUME_PTH=$WORK_DIR/latest.pth
# CHECKPOINT_PTH=/home/elodie/det3D_Output/NUSC_SECOND_9_20200923-175601/epoch_10.pth
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch \
    --nproc_per_node=2 ./tools/train.py \
    examples/second/configs/1012kitti_all_vfev3_spmiddlefhd_rpn1_mghead_syncbn_freatures3_64lines.py \
    --work_dir=$KITTI_SECOND_WORK_DIR
    # --work_dir=$WORK_DIR\
    # --resume_from=$RESUME_PTH
