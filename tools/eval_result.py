import argparse
import json
import os
import sys
import numpy as np
import torch
import yaml
from det3d import __version__, torchie
from det3d.datasets.kitti import kitti_common as kitti
from det3d.torchie import Config
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from det3d.datasets.nuscenes.nusc_common import (
    general_to_detection,
    cls_attr_dist,
    _second_det_to_nusc_box,
    _lidar_nusc_box_to_global,
    eval_main
)
from pathlib import Path

def lidar_box_nusc2kitti(gt_boxes):
    gt_boxes = gt_boxes.detach().cpu().numpy()
    R = np.array([[ 0, 1 ,0],
            [ -1,0,0],
            [ 0,0,1]])
    gt_boxes_theta = np.dot(R, gt_boxes[:, :3].T).T
    gt_boxes = np.hstack((gt_boxes_theta, gt_boxes[:, 3:]))

    gt_boxes[:, 6] = gt_boxes[:, 6] + np.sign(gt_boxes[:, 6]) * np.pi / 2
    return gt_boxes

def eval_det_result(res_path, config_file, output_dir, data_root, use_velo=False, only_front=False, unused_token_path=None):
    cfg = Config.fromfile(config_file)

    VERSION = "v1.0-trainval"
    VERSION_MAP = "val"
    EVAL_VERSION = "cvpr_2019"    

    name_mapping = general_to_detection
    class_names = []
    for task in cfg.tasks:
        class_names += task["class_names"]

    mapped_class_names = []
    for n in class_names:
        if n in name_mapping:
            mapped_class_names.append(name_mapping[n])
        else:
            mapped_class_names.append(n)
        
    nusc = NuScenes(version=VERSION, dataroot=str(data_root), verbose=True)

    if cfg.dataset_type == "NuScenesDataset":
        print("Classes:", mapped_class_names,"\n")

        eval_main(
                nusc,
                EVAL_VERSION,
                res_path,
                VERSION_MAP,
                output_dir,
                use_velo, #elodie
                only_front,
                unused_token_path,
        )

        with open(Path(output_dir) / "metrics_summary.json", "r") as f:
            metrics = json.load(f)

        detail = {}
        result = f"Nusc {VERSION} Evaluation\n"
        for name in mapped_class_names:
                detail[name] = {}
                for k, v in metrics["label_aps"][name].items():
                    detail[name][f"dist@{k}"] = v
                threshs = ", ".join(list(metrics["label_aps"][name].keys()))
                scores = list(metrics["label_aps"][name].values())
                mean = sum(scores) / len(scores)
                scores = ", ".join([f"{s * 100:.2f}" for s in scores])
                result += f"{name} Nusc dist AP@{threshs}\n"
                result += scores
                result += f" mean AP: {mean}"
                result += "\n"
        res_nusc = {
            "results": {"nusc": result},
            "detail": {"nusc": detail},
        }
    else:
        res_nusc = None

    if res_nusc is not None:
        res = {
            "results": {"nusc": res_nusc["results"]["nusc"],},
            "detail": {"eval.nusc": res_nusc["detail"]["nusc"],},
        }
    else:
        res = None

    for k, v in res["results"].items():
        print(f"Evaluation {k}: {v}")

if __name__ == "__main__":
    # -------- nusc_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn
    # -------- NUSC_SECOND_8_20200826-214223
    eval_type = "NUSC_SECOND_9_20200924-215737"
    unused_token_path = ""
    if eval_type == "nusc_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn":
        config_file = "/home/elodie/Det3D/examples/cbgs/configs/nusc_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py"
        output_dir = "/home/elodie/test_result"
        data_root = "/home/dataset/nuScenes_DATASET"
        use_velo = True
        only_front = False
        res_path = "/home/elodie/det3D_Output/nusc_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn/infos_val_10sweeps_withvelo.json"
    elif eval_type == "NUSC_SECOND_8_20200826-214223":
        config_file = "/home/elodie/Det3D/examples/second/configs/0826Nuscenes_all_vfev3_spmiddlefhd_rpn1_mghead_syncbn_freatures3.py"
        output_dir = "/home/elodie/test_result/NUSC_SECOND_8_20200826-214223"
        data_root = "/home/dataset/nuScenes_DATASET"
        use_velo = False
        res_path = "/home/elodie/det3D_Output/NUSC_SECOND_8_20200826-214223/infos_val_10sweeps_withvelo.json"
        only_front = True
        unused_token_path = "/home/dataset/nuScenes_DATASET/pkl/val_back_box_token.pkl"
    elif eval_type == "NUSC_SECOND_9_20200924-215737":
        config_file = "/home/elodie/Det3D/examples/second/configs/0923Nuscenes_all_vfev3_spmiddlefhd_rpn1_mghead_syncbn_freatures3.py"
        output_dir = "/home/elodie/test_result/" + eval_type
        data_root = "/home/dataset/nuScenes_DATASET"
        use_velo = False
        res_path = "/home/elodie/det3D_Output/" + eval_type +"/infos_val_10sweeps_withvelo.json"
        only_front = True
        unused_token_path = "/home/dataset/nuScenes_DATASET/pkl/val_back_box_token.pkl"
    eval_det_result(res_path, config_file, output_dir, data_root, use_velo, only_front, unused_token_path)
