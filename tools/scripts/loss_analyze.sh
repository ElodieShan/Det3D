#!/bin/bash
# plot_curve / cal_train_time
TYPE='plot_curve'
# 20200411_164006.log.json  20200826_214244.log.json  20200909_224323.log.json
# JSON_ROOT=/mnt/DockerDet3D/det3D_Output
# JSON_PATH=$JSON_ROOT/NUSC_SECOND_9_20200915-202036/20200915_202059.log.json
JSON_ROOT=/mnt/DockerDet3D/loss
JSON_PATH=$JSON_ROOT/20200922_164818.log.json
# loss / loss_direction
KEYS="loss"
# Test

python3 /mnt/DockerDet3D/Det3D/tools/analyze_logs.py $TYPE $JSON_PATH --keys $KEYS