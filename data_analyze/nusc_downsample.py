# Function： Nuscenes DataSet DownSample
# Editor：ElodieShan
# Date:2020-08-15

############################
#
# 说明:　用于对nusc数据集点云进行下采样，使其更贴近Tensorpro数据
#
##########################＃

import pickle
import numpy as np
from det3d.core import box_np_ops
import os
import copy
from tqdm import tqdm
import random
from det3d.datasets.kitti import kitti_common as kitti
from pointcloud_utils import *
from load_pcd import *
####
# @function:read_file()min(v),"  ",max(v)
# 1. [Nusc数据集]　读取.pcd.bin
###

def read_file(path, tries=2, num_point_feature=4):
    points = None
    try_cnt = 0
    while points is None and try_cnt < tries:
        try_cnt += 1
        try:
            points = np.fromfile(path, dtype=np.float32)
            s = points.shape[0]
            if s % 5 != 0:
                points = points[: s - (s % 5)]
            points = points.reshape(-1, 5)[:, :num_point_feature]
        except Exception:
            points = None
    return points

def load_points(info,num_point_feature=5):
    points_path = info["lidar_path"]

    if "data_type" in info:
        points = np.fromfile(points_path,dtype=np.float32).reshape([-1, 4])
    else:
        points = read_file(points_path,num_point_feature=5)

    return points

def analyze_nusc_pcd(lidar_path):
    vertical_angles = []
    points = read_file(lidar_path,num_point_feature=5)
    ring = set(points[:,4])
    vertical_angles = get_vertical_angle(points[:,2], get_distances_2d(points))
    ring_dict = {}
    for r in ring:
        ring_dict[r] = []
    for i in range(points.shape[0]):
        angle = round(vertical_angles[i],2)
        if angle not in ring_dict[points[i,4]]:
            ring_dict[points[i,4]].append(angle)
    for k,v in ring_dict.items():
        print("k:",k,"angles:",sorted(v))
    for k,v in ring_dict.items():
        # mean = mean(v[5:-5])
        angles = copy.deepcopy(sorted(v))
        angles = angles[7:-7]
        angle_mean = np.mean(angles)
        print("k:",k,"angles:",min(angles),"  ",max(angles),"  ",angle_mean)

def analyze_kitti_pcd(lidar_path):
    if lidar_path[-3:] == "pcd":
        points = get_points_from_pcd_file(lidar_path)
        points = points.astype(np.float32)
    else:
        points = np.fromfile(lidar_path,dtype=np.float32).reshape([-1, 4])

    horizontal_angles = get_horizontal_angle(points[:,0], points[:,1])
    vertical_angles = get_vertical_angle(points[:,2], get_distances_2d(points)).tolist()
    # for i in range(points.shape[0]):
        # print("horizontal_angle:",horizontal_angle[i],"\tvertical_angles:",vertical_angles[i]) 
    ring = 1
    ring_list = [ring]
    vertical_angle_ring = {}
    vertical_angle_ring[ring] = [vertical_angles[0]]
    for i in range(1,points.shape[0]):
        if horizontal_angles[i-1]<0 and horizontal_angles[i]>0:
            ring += 1
            vertical_angle_ring[ring] = []
        ring_list.append(ring)
        vertical_angle_ring[ring].append(vertical_angles[i])
    # points = np.hstack((points, vertical_angles.reshape(-1,1), np.array(ring_list).reshape(-1,1)))
    key =[33, 32, 29, 27, 25, 23, 21, 19, 16, 14, 12, 10, 8, 6, 4, 2]
    key = sorted(key)
    block = int(len(key)/4)
    print(len(key))
    for i in range(block):
        idx1 = i
        idx2 = 1*block + i
        idx3 = 2*block + i
        idx4 = 3*block + i
        try:
            print("channel:",key[idx1] ," angle mean:",round(np.mean(vertical_angle_ring[key[idx1]]),2),end='\t')
            print("channel:",key[idx2] ," angle mean:",round(np.mean(vertical_angle_ring[key[idx2]]),2),end='\t')
            print("channel:",key[idx3] ," angle mean:",round(np.mean(vertical_angle_ring[key[idx3]]),2),end='\t')
            print("channel:",key[idx4] ," angle mean:",round(np.mean(vertical_angle_ring[key[idx4]]),2),end='\t')
            print('\n')
        except:
            pass
    # vertical_angles_set = [round(angle,2) for angle in vertical_angles]
    # print(len(vertical_angles_set))
    # vertical_angles_set = sorted(set(vertical_angles_set))
    # print(vertical_angles_set)
    # print(len(vertical_angles_set))

if __name__ == "__main__":
    # nusc_path = "/home/elodie/nuScenes_DATASET_NEW/pkl/infos_train_border.pkl"
    # nusc_path = "/home/elodie/nuScenes_DATASET/pkl/infos_val_10sweeps_withvelo.pkl"
    # with open(nusc_path, "rb") as f:
    #     nusc_infos = pickle.load(f)
    # nusc_lidar_path = "/home/elodie/nuScenes_DATASET/samples/LIDAR_TOP/n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530449377.pcd.bin"
    # analyze_nusc_pcd(nusc_lidar_path)

    kitti_lidar_path = "/home/elodie/KITTI_DATASET/object/training/velodyne/000086.bin"
    analyze_kitti_pcd(kitti_lidar_path)
    # for i in range(5):
    #     points = load_points(nusc_infos[i])
    #     vertical_angles += get_vertical_angle(points[:,2], get_distances_2d(points)).tolist()
    # vertical_angles_set = [round(angle,2) for angle in vertical_angles]
    # print(len(vertical_angles_set))
    # vertical_angles_set = sorted(set(vertical_angles_set))
    # print(vertical_angles_set)
    # print(len(vertical_angles_set))