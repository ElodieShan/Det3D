#!/bin/bash
TASK_DESC=$1
DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
OUT_DIR=/home/elodie/det3D_Output

NUSC_CBGS_WORK_DIR=$OUT_DIR/NUSC_CBGS_$TASK_DESC\_$DATE_WITH_TIME
NUSC_SECOND_WORK_DIR=$OUT_DIR/NUSC_SECOND_$TASK_DESC\_$DATE_WITH_TIME
NUSC_SECOND_WORK_DIR_static=$OUT_DIR/NUSC_SECOND_1_20200423-172311\20200423_172328
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
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 ./tools/train.py examples/second/configs/0826Nuscenes_all_vfev3_spmiddlefhd_rpn1_mghead_syncbn_freatures3.py --work_dir=$NUSC_SECOND_WORK_DIR
