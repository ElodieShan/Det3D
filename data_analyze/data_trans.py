# Function： DataSet Transform
# Editor：ElodieShan
# Date:2020-05-20

############################
#
# 说明:　修改Nusc和Kitti数据集
# 
# 1. 去除不需要的标注类别
# 2. 创建新的数据集合 前后向点分离、镜像
# 3. 针对Kitti数据，需要顺时针旋转box
# 4. 根据create_pcd参数确定是否生成新的pcd文件
#
##########################＃

import pickle
import numpy as np
import matplotlib.pyplot as plt
from det3d.core import box_np_ops
import os
import copy
from tqdm import tqdm
import random
from det3d.datasets.kitti import kitti_common as kitti

from pointcloud_sample_utils import *

####
# @function:read_file()
#
# @brief:
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


# @brief: 查看路劲的上级文件是否存在，若不存在则逐层创建
def init_data_dir(path):
    file_root_path, file_name=os.path.split(path)
    if not os.path.exists(file_root_path):
        try:
            os.makedirs(file_root_path)
        except:
            print("Error occurs when make dirs %s"%file_root_path)

def get_point_num_in_box(points, boxes):
    point_indices = box_np_ops.points_in_rbbox(points, boxes)
    point_num = np.sum(point_indices==True,axis=0)  
    return point_num

####    
# @function:filter_boxes()
# @brief: 根据box中心点的y轴坐标是否大于0或是否露出1/3的物体或框的边界点最大y值是否大于1m来保留标注框
###
def filter_boxes(bbox3d, filter_back=True):
    alpha = 1
    if not filter_back:
        alpha = -1
    bbox3d_corner = box_np_ops.center_to_corner_box3d(bbox3d[:,:3],bbox3d[:,3:6],bbox3d[:,6])
    bbox3d_corner_max_y = [np.max(corner[:,1]*alpha)for corner in bbox3d_corner]
    box_mask = np.array([bbox3d[i][1]*alpha > 0 or \
        bbox3d_corner_max_y[i]>1 or\
        (bbox3d_corner_max_y[i]*bbox3d[i][1]*alpha<1 and bbox3d_corner_max_y[i]>0 and bbox3d_corner_max_y[i]/(bbox3d[i][1]*alpha + bbox3d_corner_max_y[i])>1/3) \
        for i in range(len(bbox3d))], dtype=np.bool_) #y轴点大于0
    return box_mask
####
# @function:create_flip_image_dataset()
#
# @brief:
# 1. [Nusc数据集]　去除标注box内的速度信息
# 2. [Nusc数据集]　去除box内包含点数小于阈值的box
# 3. [Nusc数据集]　创建新的数据集合 前后向点分离、镜像
# 4. [Nusc数据集]　直接生成新的info
# 5. [Nusc数据集]　info中增加字段"point_num_inbox"、"data_type"
# 6. [Nusc数据集]　info中增加字段{"pre_type":["area":"front/back/all","image":"true/false","sample":"downsample/none"}、"data_type"
#
# @ param nusc_infos:原先的info
###
def create_flip_image_dataset(nusc_infos, new_dataset_root, new_info_path, sample_switch=False, create_pcd=False):
    #----------------------------------旋转矩阵，旋转180度
    R = np.array([[-1, 0 ,0],
         [ 0,-1,0],
         [ 0,0,1]])
    infos_new = []
    ###--------------------------------循环遍历所有info
    for i in tqdm(range(len(nusc_infos))):
    # for i in tqdm(range(5)):

        info = nusc_infos[i]
        info["data_type"] = "nuscenes"

        front_info = copy.deepcopy(info)
        back_info = copy.deepcopy(info)
        front_info_image = copy.deepcopy(info)
        back_info_image = copy.deepcopy(info)
        
        lidar_path = info["lidar_path"]
        if "data_type" in info:
            if (info["data_type"] == "kitti"):
                    points = np.fromfile(info["lidar_path"]).reshape([-1, 4])
            elif (info["data_type"] == "nuscenes"):
                    points = read_file(info["lidar_path"],num_point_feature=5)
        else:
            points = read_file(info["lidar_path"],num_point_feature=5)
        #points[:,3] = 0 #将颜色赋值为0
        
        bbox3d_preds = info["gt_boxes"]
        bbox3d_preds_label = info["gt_names"]
        if sample_switch:
            points = downsample_nusc_v2(points)
            points = upsample_nusc(points)
        #----------------------------------去除box的6/7列(速度信息)
        if len(bbox3d_preds)>0 and len(bbox3d_preds[0]) == 9:
            bbox3d_preds = np.delete(bbox3d_preds,6,axis=1)
            bbox3d_preds = np.delete(bbox3d_preds,6,axis=1)

        #----------------------------------统计box内点数
        # point_indices = box_np_ops.points_in_rbbox(points, bbox3d_preds)
        # point_num = np.sum(point_indices==True,axis=0)
        
        #----------------------------------去除点数小于阈值的box
        # point_num_mask = np.array([num > num_threshold for num in point_num], dtype=np.bool_)
        # bbox3d_preds = bbox3d_preds[point_num_mask]
        # bbox3d_preds_label = bbox3d_preds_label[point_num_mask]
        
        #-----------------------------------前向点和box
        front_mask = np.array([point[1] > 0 for point in points], dtype=np.bool_) #y轴点大于0
        # front_box_mask = np.array([box[1] > 0 for box in bbox3d_preds], dtype=np.bool_) #y轴点大于0
        front_box_mask = filter_boxes(bbox3d_preds, filter_back=True)
        # front_point_num = point_num[front_box_mask]

        front_points = points[front_mask]
        front_bbox3d = bbox3d_preds[front_box_mask]
        front_bbox3d_label = bbox3d_preds_label[front_box_mask]

        #----------------------------------统计box内点数
        front_point_inbox_indices = box_np_ops.points_in_rbbox(front_points, front_bbox3d)
        front_point_num = np.sum(front_point_inbox_indices==True,axis=0)

        front_lidar_path = new_dataset_root + lidar_path[29:-8] +'_FRONT.pcd.bin'
        if i == 0:
            init_data_dir(front_lidar_path)
        front_points.astype(np.float32)
        if create_pcd:
            #-----------------------------------写成新的pcd.bin
            front_points.tofile(front_lidar_path)
        
        #-----------------------------------修改info
        front_info["lidar_path"] = front_lidar_path
        front_info["sweeps"] = []
        front_info["gt_boxes"] = front_bbox3d
        front_info["gt_boxes_velocity"] = np.zeros([front_bbox3d.shape[0],3])
        front_info["gt_names"] = front_bbox3d_label
        front_info["num_points_in_gt"] = front_point_num
        front_info["pre_type"] = {}
        front_info["pre_type"]["area"] = "front"
        front_info["pre_type"]["image"] = False
        front_info["pre_type"]["sample"] = sample_switch
    
        #-----------------------------------背面点和box
        points_copy = copy.deepcopy(points).astype(np.float32)
        back_points = points_copy[~front_mask]

        back_box_mask = filter_boxes(bbox3d_preds, filter_back=False)
        back_bbox3d = bbox3d_preds[back_box_mask]
        back_bbox3d_label = bbox3d_preds_label[back_box_mask]
        # back_point_num = point_num[back_box_mask]

        # back_bbox3d = bbox3d_preds[~front_box_mask]
        # back_bbox3d_label = bbox3d_preds_label[~front_box_mask]
        # back_point_num = point_num[~front_box_mask]

        #-----------------------------------将背面点的点和box旋转180度
        back_points_theta = np.hstack((back_points[:,:2],np.zeros((back_points.shape[0],1))))
        back_points_theta = np.dot(R,back_points_theta.T).T
        back_points = np.hstack((back_points_theta[:,:2],back_points[:,2:]))

        #-----------------------------------将box的中心点旋转180
        back_bbox3d_theta = np.dot(R,back_bbox3d[:,:3].T).T
        back_bbox3d = np.hstack((back_bbox3d_theta,back_bbox3d[:,3:]))

        #----------------------------------统计box内点数
        back_point_inbox_indices = box_np_ops.points_in_rbbox(back_points, back_bbox3d)
        back_point_num = np.sum(back_point_inbox_indices==True,axis=0)

        back_lidar_path = new_dataset_root + lidar_path[29:-8] +'_BACK.pcd.bin'
        back_points_copy1 = copy.deepcopy(back_points).astype(np.float32)
        if create_pcd:
            #-----------------------------------写成新的pcd.bin
            back_points_copy1.tofile(back_lidar_path)
        
        #-----------------------------------修改info
        back_info["lidar_path"] = back_lidar_path
        back_info["sweeps"] = []
        back_info["gt_boxes"] = back_bbox3d
        back_info["gt_boxes_velocity"] = np.zeros([back_bbox3d.shape[0],3])
        back_info["gt_names"] = back_bbox3d_label
        back_info["pre_type"] = {}
        back_info["num_points_in_gt"] = back_point_num
        back_info["pre_type"]["area"] = "back"
        back_info["pre_type"]["image"] = False
        back_info["pre_type"]["sample"] = sample_switch

        #镜像点
        #------------------------------------前后面点的镜像点
        front_points_copy = copy.deepcopy(front_points).astype(np.float32)
        front_points_image = np.hstack((-front_points_copy[:,:1],front_points_copy[:,1:]))
        front_bbox3d_image = copy.deepcopy(front_bbox3d)
        front_bbox3d_image[:,0] = -front_bbox3d_image[:,0]
        front_bbox3d_image[:,-1] = -front_bbox3d_image[:,-1]
        front_bbox3d_label_image = copy.deepcopy(front_bbox3d_label)
        front_point_num_image = copy.deepcopy(front_point_num)

        front_lidar_path_image = new_dataset_root + lidar_path[29:-8] +'_FRONT_IMAGE.pcd.bin'
        front_points_image.astype(np.float32)
        if create_pcd:
            #------------------------------------写成新的pcd.bin
            front_points_image.tofile(front_lidar_path_image)
        
        #------------------------------------修改info
        front_info_image["lidar_path"] = front_lidar_path_image
        front_info_image["sweeps"] = []
        front_info_image["gt_boxes"] = front_bbox3d_image
        front_info_image["gt_boxes_velocity"] = np.zeros([front_bbox3d_image.shape[0],3])
        front_info_image["gt_names"] = front_bbox3d_label_image   
        front_info_image["num_points_in_gt"] = front_point_num_image   
        front_info_image["pre_type"] = {}
        front_info_image["pre_type"]["area"] = "front"
        front_info_image["pre_type"]["image"] = True
        front_info_image["pre_type"]["sample"] = sample_switch
        #------------------------------------前后面点的镜像点
        back_points_copy = copy.deepcopy(back_points).astype(np.float32)
        back_points_image = np.hstack((-back_points_copy[:,:1],back_points_copy[:,1:]))
        back_bbox3d_image = copy.deepcopy(back_bbox3d)
        back_bbox3d_image[:,0] = -back_bbox3d_image[:,0]
        back_bbox3d_image[:,-1] = -back_bbox3d_image[:,-1]
        back_bbox3d_label_image = copy.deepcopy(back_bbox3d_label)
        back_point_num_image = copy.deepcopy(back_point_num)
        
        back_lidar_path_image = new_dataset_root + lidar_path[29:-8] +'_BACK_IMAGE.pcd.bin'
        back_points_image.astype(np.float32)
        if create_pcd:
            #------------------------------------写成新的pcd.bin
            back_points_image.tofile(back_lidar_path_image)
        
        #------------------------------------修改info
        back_info_image["lidar_path"] = back_lidar_path_image
        back_info_image["sweeps"] = []
        back_info_image["gt_boxes"] = back_bbox3d_image
        back_info_image["gt_boxes_velocity"] = np.zeros([back_bbox3d_image.shape[0],3])
        back_info_image["gt_names"] = back_bbox3d_label_image
        back_info_image["num_points_in_gt"] = back_point_num_image   
        back_info_image["pre_type"] = {}
        back_info_image["pre_type"]["area"] = "back"
        back_info_image["pre_type"]["image"] = True
        back_info_image["pre_type"]["sample"] = sample_switch

        infos_new.append(front_info)
        infos_new.append(back_info)
        infos_new.append(front_info_image)
        infos_new.append(back_info_image)
        #------------------------------------打印进程，之后可以写成进度条
        if i%1000==0:
            print(i)
        
    #----------------------------------------打包info
    init_data_dir(new_info_path)
    with open(new_info_path, "wb") as f:
        pickle.dump(infos_new, f)
    print("finish!")

def convert_detection_to_kitti_anno(info, boxes, label, score):
    print("info[calib]:\n", info["calib"])
    print("info[annos]:\n", info["annos"])
  
    calib = info["calib"]
    rect = calib["R0_rect"]
    Trv2c = calib["Tr_velo_to_cam"]
    P2 = calib["P2"]

    # anno = info["annos"]
    anno = kitti.get_start_result_anno()
    final_box_preds = boxes
    label_preds = label
    scores = score
    if final_box_preds.shape[0] != 0:
        final_box_preds[:, -1] = box_np_ops.limit_period(
            final_box_preds[:, -1], offset=0.5, period=np.pi * 2,
            )
        final_box_preds[:, 2] -= final_box_preds[:, 5] / 2

        # aim: x, y, z, w, l, h, r -> -y, -z, x, h, w, l, r
        # (x, y, z, w, l, h r) in lidar -> (x', y', z', l, h, w, r) in camera
        box3d_camera = box_np_ops.box_lidar_to_camera(
                final_box_preds, rect, Trv2c
        )

        camera_box_origin = [0.5, 1.0, 0.5]
        box_corners = box_np_ops.center_to_corner_box3d(
                    box3d_camera[:, :3],
                    box3d_camera[:, 3:6],
                    box3d_camera[:, 6],
                    camera_box_origin,
                    axis=1,
        )
        box_corners_in_image = box_np_ops.project_to_image(box_corners, P2)
        # box_corners_in_image: [N, 8, 2]
        minxy = np.min(box_corners_in_image, axis=1)
        maxxy = np.max(box_corners_in_image, axis=1)
        bbox = np.concatenate([minxy, maxxy], axis=1)

        for j in range(box3d_camera.shape[0]):
            image_shape = info["image"]["image_shape"]
            # if bbox[j, 0] > image_shape[1] or bbox[j, 1] > image_shape[0]:
            #     continue
            # if bbox[j, 2] < 0 or bbox[j, 3] < 0:
            #     continue
            bbox[j, 2:] = np.minimum(bbox[j, 2:], image_shape[::-1])
            bbox[j, :2] = np.maximum(bbox[j, :2], [0, 0])
            anno["bbox"].append(bbox[j])

            anno["alpha"].append(
                        -np.arctan2(-final_box_preds[j, 1], final_box_preds[j, 0])
                        + box3d_camera[j, 6]
            )
            # anno["dimensions"].append(box3d_camera[j, [4, 5, 3]])
            anno["dimensions"].append(box3d_camera[j, 3:6])
            anno["location"].append(box3d_camera[j, :3])
            anno["rotation_y"].append(box3d_camera[j, 6])
            # anno["name"].append(class_names[int(label_preds[j])])
            anno["truncated"].append(0.0)
            anno["occluded"].append(0)

    anno["bbox"] = np.array(anno["bbox"])
    anno["alpha"] = -np.arctan2(-final_box_preds[:, 1], final_box_preds[:, 0]) + final_box_preds[:, 6]
    # anno["alpha"] = np.array(anno["alpha"])
    anno["dimensions"] = np.array(anno["dimensions"])      
    anno["location"] = np.array(anno["location"])      
    anno["rotation_y"] = np.array(anno["rotation_y"])      
    anno["truncated"] = np.array(anno["truncated"])      
    anno["occluded"] = np.array(anno["occluded"])      
    anno["rotation_y"] = np.array(anno["rotation_y"])        
    anno["name"] = label_preds
    anno["score"] = scores
    print("new anno:\n", anno)
    return anno

def sample_gt_nusc(nusc_db_root, nusc_db_infos, new_nusc_db_root, new_db_info_path, point_feature_num=4, sample_switch=False, create_pcd=False):
    new_db_infos = {}
    for class_name, infos in nusc_db_infos.items():
        new_db_infos[class_name] = []
        # for i in tqdm(range(5)): 
        for i in tqdm(range(len(infos))):
            points_path = nusc_db_root + infos[i]['path']
            try:
                points = np.fromfile(points_path,dtype=np.float32).reshape([-1, point_feature_num])
            except:
                print("read pointcloud from path failed. please check point_feature_num")
            points = downsample_nusc_v2(points)
            points = upsample_nusc(points)

            #----------------------------------更新box内点数
            infos[i]['num_points_in_gt'] = points.shape[0]
            points.astype(np.float32)
            if create_pcd:
                #-----------------------------------写成新的pcd.bin

                file_name = str(infos[i]['path']).split('/')[1]
                new_lidar_path = new_nusc_db_root + "gt_database_downsample/" + file_name
                infos[i]['path'] = "gt_database_downsample/" + file_name
                if i == 0:
                    init_data_dir(new_lidar_path)
                points.tofile(new_lidar_path)
            new_db_infos[class_name].append(infos[i])
    with open(new_db_info_path, "wb") as f:
        pickle.dump(new_db_infos, f)
        
####
# @function:create_dataset_kitti2nusc_format()
#
# @brief: 将kitti数据集转换为nusc格式，包括坐标系转换、info格式（保留kitti原info内容）
# 1. [Kitti数据集]　将原info中的信息合成gt_box
# 2. [Kitti数据集]　去除box内包含点数小于阈值的box
# 3. [Kitti数据集]　创建新的数据集合，由于标注仅在前向180度包含，因此仅对数据集镜像
# 4. [Kitti数据集]　直接生成新的info，info对齐nusc的info格式
# 5. [Kitti数据集]　info中增加字段"data_type"="kitti","num_points_in_gt"=[n,n,n,n,n.....]
# 6. [Kitti数据集]　将van类别合并至truck
#***pcd.bin中不将rgb赋值为０
#***不去除box内包含点数小于阈值的box，改为增加字段num_points_in_gt，在训练时剔除
#
# @ param nusc_infos:原先的info
###
def create_dataset_kitti2nusc_format(kitti_infos, new_info_path, point_feature_num=4, sample_switch=False, image_switch=False, create_pcd=False):
    kitti_infos_new = []
    #----------------------------------旋转矩阵，旋转90度
    R = np.array([[ 0, -1 ,0],
        [ 1,0,0],
        [ 0,0,1]])

    velodyne_root_path = "/home/elodie/KITTI_DATASET/object/"
    # for i in tqdm(range(5)):
    for i in tqdm(range(len(kitti_infos))):
        kitti_info = kitti_infos[i]
        kitti_info_image = copy.deepcopy(kitti_info) #镜像

        ref_lidar_path = "/home/elodie/KITTI_DATASET_NEW/object/" + kitti_info["point_cloud"]["velodyne_path"] #绝对路径
        ref_lidar_path_image = "/home/elodie/KITTI_DATASET_NEW/object/" + kitti_info["point_cloud"]["velodyne_path"][:-4] +'_IMAGE.bin' #镜像绝对路径

        ref_cam_path = "/home/elodie/KITTI_DATASET_NEW/object/" + kitti_info["image"]["image_path"] #绝对路径
        
        cam_intrinsic_pre = kitti_info["calib"]["R0_rect"]
        cam_intrinsic_pre = np.delete(cam_intrinsic_pre, 3, axis=0)
        cam_intrinsic_pre = np.delete(cam_intrinsic_pre, 3, axis=1)
        ref_cam_intrinsic = cam_intrinsic_pre

        token = kitti_info["image"]["image_idx"]
        ref_from_car = kitti_info["calib"]["Tr_velo_to_cam"]
        car_from_global = kitti_info["calib"]["P0"]

        #-------------------------------读取点云数据
        points = np.fromfile(velodyne_root_path+kitti_info["point_cloud"]["velodyne_path"], dtype=np.float32, count=-1).reshape([-1, point_feature_num])
        if sample_switch:
            points = downsample_kitti(points)
            point_feature_num += 1
        #-------------------------------顺时针旋转90度并保存为bin
        points_xy = np.hstack((points[:,:2],np.ones((points.shape[0],1))))
        points_theta = np.dot(R,points_xy.T).T
        points_theta = np.hstack((points_theta[:,:2],points[:,2:]))
        if create_pcd:
            if i == 0:
                init_data_dir(ref_lidar_path)
            points_theta.astype(np.float32).tofile(ref_lidar_path)

        if image_switch:
            #-------------------------------对已旋转90度后的点做镜像点，并保存为bin
            points_copy = copy.deepcopy(points_theta).astype(np.float32)
            points_image = np.hstack((-points_copy[:,:1],points_copy[:,1:]))
            if create_pcd:
                points_image.tofile(ref_lidar_path_image)

        if "annos" in kitti_info:
            annos = kitti_info["annos"]
            # we need other objects to avoid collision when sample
            # annos = kitti.remove_dontcare(annos) #elodie
            locs = annos["location"]
            dims = annos["dimensions"]
            rots = annos["rotation_y"]
            gt_names = annos["name"]
            gt_boxes = np.concatenate(
                [locs, dims, rots[..., np.newaxis]], axis=1
            ).astype(np.float32)
            calib = kitti_info["calib"]
            
            #box_camera_to_lidar [xyz_lidar, l, h, w, r] - > [xyz_lidar, w, l, h, r]
            gt_boxes = box_np_ops.box_camera_to_lidar(
                gt_boxes, calib["R0_rect"], calib["Tr_velo_to_cam"]
            )

            # only center format is allowed. so we need to convert
            # kitti [0.5, 0.5, 0] center to [0.5, 0.5, 0.5]
            box_np_ops.change_box3d_center_(
                gt_boxes, [0.5, 0.5, 0], [0.5, 0.5, 0.5]
            )
            #------------------------------调整坐标系,将gt_boxes 顺时针旋转90°
            gt_boxes_theta = np.dot(R, gt_boxes[:, :3].T).T
            gt_boxes = np.hstack((gt_boxes_theta, gt_boxes[:, 3:]))

            gt_boxes[:, 6] = gt_boxes[:, 6] - np.sign(gt_boxes[:, 6]) * np.pi / 2
            gt_boxes_velocity = np.zeros([gt_boxes.shape[0],3])

            gt_names_pre = annos["name"]

            #------------------------------标注类别转为小写，Cyclist->bicycle,与Nusc对齐
            for i in range(len(gt_names_pre)):
                if  gt_names_pre[i] == 'Cyclist':
                    gt_names_pre[i] = 'bicycle'
                if  gt_names_pre[i] == 'Truck':
                    # kitti的truck分两类，长小于8m的归于truck，长大于8m的归于bus。
                    box_length = gt_boxes[i][4]
                    if box_length<=8:
                        gt_names_pre[i] = 'truck'
                    else:
                        gt_names_pre[i] = "bus"
                if  gt_names_pre[i] == 'Van':
                    gt_names_pre[i] = 'truck'                    
                gt_names_pre[i] = gt_names_pre[i].lower()
            
            #-------------------------------统计标注box内的点数
            point_indices = box_np_ops.points_in_rbbox(points_theta, gt_boxes)
            point_num = np.sum(point_indices==True,axis=0)  

            #-------------------------------去除点数小于阈值的box
            # point_num_mask = np.array([num > num_threshold for num in point_num], dtype=np.bool_)
            # gt_boxes = gt_boxes[point_num_mask]
            # gt_names = gt_names_pre[point_num_mask]

            #-------------------------------生成镜像后的box和lable标注
            gt_boxes_image = copy.deepcopy(gt_boxes)
            gt_boxes_image[:,0] = -gt_boxes_image[:,0]
            gt_boxes_image[:,-1] = -gt_boxes_image[:,-1]
            gt_names_image = copy.deepcopy(gt_names)

            # new_anno = convert_detection_to_kitti_anno(kitti_info, gt_boxes, gt_names_pre, annos["score"])

        info = {
            "lidar_path": ref_lidar_path,
            "cam_front_path": ref_cam_path,
            "cam_intrinsic": ref_cam_intrinsic,
            "token": token,
            "sweeps": [],
            "ref_from_car": ref_from_car,
            "car_from_global": car_from_global,
            "timestamp": 0.0,
            "gt_boxes": gt_boxes,
            "gt_boxes_velocity": gt_boxes_velocity,
            "gt_names": gt_names_pre,
            "gt_boxes_token": np.zeros(gt_boxes.shape[0]),
            "num_points_in_gt":point_num,
            "data_type": "kitti",
            "point_feature_num": point_feature_num,
            "image":kitti_info["image"],
            "point_cloud":kitti_info["point_cloud"],
            "calib":kitti_info["calib"],
            # "annos":new_anno
            "annos":kitti_info["annos"]
        }
        info["pre_type"]={
            "area":"all",
            "image":False,
            "sample":sample_switch
        }
        kitti_infos_new.append(info)

        if image_switch:
            kitti_info_image = {
                "lidar_path": ref_lidar_path_image,
                "cam_front_path": ref_cam_path,
                "cam_intrinsic": ref_cam_intrinsic,
                "token": token,
                "sweeps": [],
                "ref_from_car": ref_from_car,
                "car_from_global": car_from_global,
                "timestamp": 0.0,
                "gt_boxes": gt_boxes_image,
                "gt_boxes_velocity": gt_boxes_velocity,
                "gt_names": gt_names_image,
                "gt_boxes_token": np.zeros(gt_boxes.shape[0]),
                "num_points_in_gt":point_num,
                "data_type": "kitti",
                "point_feature_num": point_feature_num,
                "image":kitti_info["image"],
                "point_cloud":kitti_info["point_cloud"],
                "calib":kitti_info["calib"],
                "annos":kitti_info["annos"],
            }
            kitti_info_image["pre_type"]={
                "area":"all",
                "image":True,
                "sample":sample_switch
            }
            kitti_infos_new.append(kitti_info_image)
        
    init_data_dir(new_info_path)
    with open(new_info_path, "wb") as f:
        pickle.dump(kitti_infos_new, f)

def change_gt_database_kitti2nusc_format(kitti_db_infos, new_info_path, point_feature_num=4, sample_switch=False, create_pcd=False):
    kitti_db_infos_new = {}
    #----------------------------------旋转矩阵，旋转90度
    R = np.array([[ 0, -1 ,0],
        [ 1,0,0],
        [ 0,0,1]])

    velodyne_root_path = "/home/elodie/KITTI_DATASET/object/"
    new_root_path = "/home/elodie/KITTI_DATASET_NEW/object/"
    
    kitti_class_list = kitti_db_infos.keys()
    for kitti_class in kitti_class_list:
        kitti_infos = kitti_db_infos[kitti_class]
        kitti_infos_new =[]

        #------------------------------标注类别转为小写，Cyclist->bicycle,与Nusc对齐  
        kitti_class_new = ""
        if  kitti_class == 'Cyclist':
            kitti_class_new = 'bicycle'
        elif  kitti_class == 'Van':
            kitti_class_new = 'truck'                    
        else:
            kitti_class_new = kitti_class.lower()
        
        
        # for i in tqdm(range(5)):
        for i in tqdm(range(len(kitti_infos))):
            kitti_info = kitti_infos[i]

            db_lidar_path = velodyne_root_path + kitti_info["path"] #绝对路径

            if sample_switch: #修改gt文件夹名称为gt_database_downsample
                file_name = str(kitti_info["path"]).split('/')[-1]
                db_lidar_path_new = new_root_path + "gt_database_downsample/" + file_name
                kitti_info["path"] = "gt_database_downsample/" + file_name
            else:
                db_lidar_path_new = new_root_path + kitti_info["path"]

            #-------------------------------读取点云数据，逆时针旋转90度并保存为bin
            points = np.fromfile(db_lidar_path, dtype=np.float32, count=-1).reshape([-1, point_feature_num])
            if sample_switch:
                points = downsample_kitti(points)
                point_feature_num += 1
            points_xy = np.hstack((points[:,:2],np.ones((points.shape[0],1))))
            points_theta = np.dot(R,points_xy.T).T
            points_theta = np.hstack((points_theta[:,:2],points[:,2:]))
            if create_pcd:
                if i==0:
                    init_data_dir(db_lidar_path_new)
                points_theta.astype(np.float32).tofile(db_lidar_path_new)
            
            gt_boxes = np.array([copy.deepcopy(kitti_info["box3d_lidar"])])
            #------------------------------调整坐标系,将gt_boxes 逆时针旋转90°
            gt_boxes_theta = np.dot(R, gt_boxes[:, :3].T).T
            gt_boxes = np.hstack((gt_boxes_theta, gt_boxes[:, 3:]))

            gt_boxes[:, 6] = gt_boxes[:, 6] - np.sign(gt_boxes[:, 6]) * np.pi / 2

            kitti_info["box3d_lidar"] = gt_boxes[0]
            kitti_info["name"] = kitti_class_new
            if kitti_class.lower() == 'truck':
                # kitti的truck分两类，长小于8m的归于truck，长大于8m的归于bus。
                box_length = kitti_info["box3d_lidar"][4]
                if box_length<=8:
                    kitti_infos_new.append(kitti_info)
                else:
                    kitti_info["name"] = "bus"
                    if "bus" in kitti_db_infos_new:
                        kitti_db_infos_new["bus"].append(kitti_info)
                    else:
                        kitti_db_infos_new["bus"] = [kitti_info]
            else:        
                kitti_infos_new.append(kitti_info)

        kitti_db_infos_new[kitti_class_new] = kitti_infos_new

    init_data_dir(new_info_path)
    with open(new_info_path, "wb") as f:
        pickle.dump(kitti_db_infos_new, f)

####
# @function:mix_kitti_nusc()
#
# @brief:
# 1. 混合kitti和nusc的训练info，并对kitti进行倍增
#
# @ param kitti_infos:已转换格式的kitti info
# @ param nusc_infos:已转换格式的nusc_info
# ***如果没有标注信息，则将此帧去除，不计入info
###
def mix_kitti_nusc(kitti_infos, nusc_infos, muti_seed, new_info_path, point_feature_num=4):
    infos_new = kitti_infos
    for i in range(muti_seed-1):
        infos_new = infos_new + kitti_infos
    infos_new = infos_new + nusc_infos
    random.shuffle(infos_new)
    print("kitti info num:",len(kitti_infos))
    print("nusc info num:",len(nusc_infos))
    print("muti_seed:",muti_seed)
    print("new info num:",len(infos_new))
    # ----------------------------------根据点和类别对box进行筛选，直接删除
    class_list = ['car', 'truck', 'bus', 'trailer','pedestrian','motorcycle', 'bicycle','traffic_cone']    #保留的类别
    point_num_thred = 4
    infos_new_filtered = []
    for i in tqdm(range(len(infos_new))):
        info = infos_new[i]
        # print(info.keys())
        gt_names = info["gt_names"] 
        gt_box = info['gt_boxes']
        if 'num_points_in_gt' in info:
            gt_point_num = info['num_points_in_gt']
        else:
            points = np.fromfile(info["lidar_path"],dtype=np.float32).reshape([-1, point_feature_num])
            gt_point_num = get_point_num_in_box(points, gt_box)

        # box保留条件:1.box内点数大于阈值 2.traffic_cone类别 3.box中心点y值大于30
        box_mask = np.array([gt_names[i] in class_list and \
            (gt_point_num[i] >= point_num_thred or \
            (gt_names[i]=='traffic_cone' and gt_point_num[i] > 1)  or\
            (gt_box[i][1]>30 and gt_point_num[i] > 1))
            for i in range(len(gt_names))], dtype=np.bool_)

        info['gt_names'] = gt_names[box_mask]
        info['num_points_in_gt'] = gt_point_num[box_mask]
        info['gt_boxes'] = info['gt_boxes'][box_mask]
        info['gt_boxes_velocity'] = info['gt_boxes_velocity'][box_mask]
        info['point_feature_num'] = point_feature_num
        infos_new_filtered.append(info)
    with open(new_info_path, "wb") as f:
        pickle.dump(infos_new_filtered, f)

def mix_gt_kitti_nusc(kitti_infos, nusc_infos, new_info_path, kitti_root=None, nusc_root=None, front_nusc=True, add_data_type=True, absolutely_path=True):

    # --------------------------------将Nusc的Ground Truth DataBase info中的背面框，中心点y值取相反数转换到正面
    if front_nusc:
        for class_name, infos in nusc_infos.items():
            for i in tqdm(range(len(infos))):
                if infos[i]["box3d_lidar"][1]<0:
                    infos[i]["box3d_lidar"][1] = -infos[i]["box3d_lidar"][1]
                if infos[i]["box3d_lidar"].shape[0] == 9:#elodie
                    infos[i]["box3d_lidar"] = np.delete(infos[i]["box3d_lidar"],6)#elodie
                    infos[i]["box3d_lidar"] = np.delete(infos[i]["box3d_lidar"],6)#elodie 
    if add_data_type:
        for class_name, infos in nusc_infos.items():
            for i in tqdm(range(len(infos))):
                infos[i]["data_type"] = "nuscenes"
                if absolutely_path:
                    infos[i]["path"] = nusc_root + infos[i]["path"]
                    infos[i]['point_feature_num'] = 5
        for class_name, infos in kitti_infos.items():
            for i in tqdm(range(len(infos))):
                infos[i]["data_type"] = "kitti"
                if absolutely_path:
                    infos[i]["path"] = kitti_root + infos[i]["path"]
                    infos[i]['point_feature_num'] = 5
                
    new_infos = nusc_infos

    nucs_classes = new_infos.keys()

    for class_name in kitti_infos.keys():
        if class_name in nucs_classes:
            new_infos[class_name] += kitti_infos[class_name]

    init_data_dir(new_info_path)
    with open(new_info_path, "wb") as f:
        pickle.dump(new_infos, f)

if __name__ == '__main__':
    #-----------------------NUSC
    # nusc_path = "/home/elodie/nuScenes_DATASET/pkl/infos_train_10sweeps_withvelo.pkl"
    # new_info_path = "/home/elodie/nuScenes_DATASET_NEW/pkl/infos_train_sample_20200825.pkl"
    # new_dataset_root = "/home/elodie/nuScenes_DATASET_NEW"
    # with open(nusc_path, "rb") as f:
    #     nusc_infos = pickle.load(f)
    # print("len nusc_infos",len(nusc_infos))
    # create_flip_image_dataset(nusc_infos, new_dataset_root, new_info_path, sample_switch=True, create_pcd=True)

    #-----------------------NUSC Val Back Sample
    # nusc_val_path = "/home/elodie/nuScenes_DATASET/pkl/infos_val_10sweeps_withvelo.pkl"
    # back_token_path = "/home/dataset/nuScenes_DATASET/pkl/val_back_box_token.pkl"
    # with open(nusc_val_path, "rb") as f:
    #     nusc_infos = pickle.load(f)
    # back_box_sample =[] 
    # for info in nusc_infos:
    #     gt_boxes = info["gt_boxes"]
    #     gt_boxes_token = info["gt_boxes_token"]
    #     back_box_mask = filter_boxes(gt_boxes, filter_back=True)
    #     back_box_sample += gt_boxes_token[~back_box_mask].tolist()
    # with open(back_token_path,"wb") as f:
    #     pickle.dump(back_box_sample, f)


    #-----------------------NUSC GT Sample
    # nusc_db_root = "/home/elodie/nuScenes_DATASET/"
    # nusc_db_infos_path = "/home/elodie/nuScenes_DATASET/pkl/dbinfos_train_1sweeps_withvelo.pkl"
    # new_nusc_db_root = "/home/elodie/nuScenes_DATASET_NEW/"
    # new_db_info_path = "/home/elodie/nuScenes_DATASET_NEW/pkl/dbinfos_train_sample_20200825.pkl"
    # with open(nusc_db_infos_path, "rb") as f:
    #     nusc_db_infos = pickle.load(f)
    # sample_gt_nusc(nusc_db_root, nusc_db_infos, new_nusc_db_root, new_db_info_path, point_feature_num=5, sample_switch=True, create_pcd=True)
    #-----------------------KITTI Train
    # kitti_path = "/home/elodie/KITTI_DATASET/object/kitti_infos_train.pkl"
    # new_info_path = "/home/elodie/KITTI_DATASET_NEW/object/kitti_infos_train_sample_20200825.pkl"
    # with open(kitti_path, "rb") as f:
    #     kitti_infos = pickle.load(f)
    # print("len kitti_infos",len(kitti_infos))
    # create_dataset_kitti2nusc_format(kitti_infos, new_info_path,
    #         point_feature_num=4, sample_switch=True, image_switch=True, create_pcd=False)

    # info_path = "/home/elodie/KITTI_DATASET_NEW/object/kitti_infos_train_sample_20200825_back.pkl"
    # new_info_path = "/home/elodie/KITTI_DATASET_NEW/object/kitti_infos_train_sample_20200825.pkl"

    # with open(info_path, "rb") as f:
    #     kitti_infos = pickle.load(f)
    # for i in tqdm(range(len(kitti_infos))):
    #     info = kitti_infos[i]
    #     for j in range(len(info["gt_names"])):
    #         if info["gt_names"][j] == 'truck':
    #             box_length = info["gt_boxes"][j][4]
    #             if box_length>8:
    #                 info["gt_names"][j] = "bus"
    #                 print("box:",info["gt_boxes"][j])
    #                 print("name:",info["gt_names"][j])
    # with open(new_info_path, "wb") as f:
    #     pickle.dump(kitti_infos, f)
    #-----------------------KITTI Val
    # kitti_path = "/home/elodie/KITTI_DATASET/object/kitti_infos_val.pkl"
    # new_info_path = "/home/elodie/KITTI_DATASET_NEW/object/kitti_infos_val_kitti.pkl"
    # with open(kitti_path, "rb") as f:
    #     kitti_infos = pickle.load(f)
    # print("len kitti_infos",len(kitti_infos))
    # create_dataset_kitti2nusc_format(kitti_infos, new_info_path,
    #     point_feature_num=4, sample_switch=True, image_switch=False, create_pcd=False)

    # with open(new_info_path, "rb") as f:
    #     kitti_infos = pickle.load(f)
    # kitti_infos_new = []
    # for i in tqdm(range(len(kitti_infos))):
    #     info = kitti_infos[i]
        
    #     nonezero_points_inbox_mask = np.where(info["num_points_in_gt"]>0)
    #     if np.array(nonezero_points_inbox_mask).shape[1]<info["num_points_in_gt"].shape[0]:
    #         print(np.array(nonezero_points_inbox_mask).shape[0])
    #         print(info["num_points_in_gt"].shape[0])
    #         print("index:",i)
    #         print("Find Zeros")
    #         print(info["num_points_in_gt"])
    #         # print("annos:",info["annos"])
    #         for k,v in info["annos"].items():
    #             info["annos"][k] = info["annos"][k][nonezero_points_inbox_mask]
    #         info["gt_boxes"] = info["gt_boxes"][nonezero_points_inbox_mask]
    #         info["gt_names"] = info["gt_names"][nonezero_points_inbox_mask]
    #         info["num_points_in_gt"] = info["num_points_in_gt"][nonezero_points_inbox_mask]
    #     kitti_infos_new.append(info)
    # with open(new_info_path, "wb") as f:
    #     pickle.dump(kitti_infos, f)
    #-----------------------KITTI Ground Truth DataBase
    # kitti_db_infos_path = "/home/elodie/KITTI_DATASET/object/dbinfos_train.pkl"
    # new_info_path =  "/home/elodie/KITTI_DATASET_NEW/object/dbinfos_train_sample_20200825.pkl"
    # with open(kitti_db_infos_path, "rb") as f:
    #     kitti_db_infos = pickle.load(f)
    # change_gt_database_kitti2nusc_format(kitti_db_infos, new_info_path, point_feature_num=5, sample_switch=True, create_pcd=False)

    #----------------------Mix Ground Truth DataBase Info of Kitti & Nusc
    # kitti_root = "/home/elodie/KITTI_DATASET_NEW/object/"
    # kitti_db_infos_path = "/home/elodie/KITTI_DATASET_NEW/object/kitti_dbinfos_train_sample_20200825.pkl"
    # nusc_root = "/home/elodie/nuScenes_DATASET_NEW/"
    # nusc_db_info_path =  "/home/elodie/nuScenes_DATASET_NEW/pkl/dbinfos_train_sample_20200825.pkl"
    # new_info_path = "/home/elodie/DATASET_INFO/pkl/dbinfos_mix_sample.pkl"
    # with open(kitti_db_infos_path, "rb") as f:
    #     kitti_db_infos = pickle.load(f)

    # with open(nusc_db_info_path, "rb") as f:
    #     nusc_db_infos = pickle.load(f)
    # mix_gt_kitti_nusc(kitti_db_infos, nusc_db_infos, new_info_path, kitti_root=kitti_root, nusc_root=nusc_root, front_nusc=True, add_data_type=True, absolutely_path=True)

    #----------------------Mix Info of Kitti & Nusc
    # kitti_info_path = "/home/elodie/KITTI_DATASET_NEW/object/kitti_infos_train_sample_20200825.pkl"
    # nusc_info_path =  "/home/elodie/nuScenes_DATASET_NEW/pkl/infos_train_sample_20200825.pkl"
    # new_info_path = "/home/elodie/DATASET_INFO/infos_train_sample.pkl"
    # muti_seed = 3
    # with open(kitti_info_path, "rb") as f:
    #     kitti_infos = pickle.load(f)

    # with open(nusc_info_path, "rb") as f:
    #     nusc_infos = pickle.load(f)

    # mix_kitti_nusc(kitti_infos, nusc_infos, muti_seed, new_info_path, point_feature_num=5)

    #-------------------------db DATASET 将所有背面的box去除。
    # db_info_path = "/home/elodie/nuScenes_DATASET/pkl/dbinfos_train_10sweeps_withoutvelo.pkl"
    # new_db_info_path = "/home/elodie/nuScenes_DATASET/pkl/dbinfos_train_10sweeps_withoutvelo_forward.pkl"
    # # class_list = ['car', 'truck', 'bus', 'trailer','pedestrian','motorcycle', 'bicycle','traffic_cone']    #保留的类别

    # with open(db_info_path, "rb") as f:
    #     nusc_db_info = pickle.load(f)

    # gt_boxes = {}
    # gt_infos_filtered = {}
    # for class_name in class_list:
    #     filtered_info = [gt_info for gt_info in nusc_db_info[class_name] 
    #             if gt_info["box3d_lidar"][1]>0 and  gt_info["box3d_lidar"][1]<50 and  gt_info["box3d_lidar"][0]>-50 and  gt_info["box3d_lidar"][0]<-50  and  gt_info["box3d_lidar"]>5 ]
    #     gt_infos_filtered[class_name] = np.array(filtered_info,dtype=np.bool_)
    
    # with open(new_db_info_path, "wb") as f:
    #     pickle.dump(gt_infos_filtered, f)

    #------------------------Test
    # new_info_path = "/home/elodie/nuScenes_DATASET/pkl/infos_train_10sweeps_withvelo.pkl"
    # ori_info_path = "/home/elodie/nuScenes_DATASET_NEW/infos_train.pkl"
    # with open(new_info_path, "rb") as f:
    #      new_info = pickle.load(f)
    # with open(ori_info_path, "rb") as f:
    #      ori_info = pickle.load(f)

    # for i in range(len(new_info)):
    #     new = new_info[i]["lidar_path"].split('/')[-1]
    #     ori = ori_info[i*4]["lidar_path"].split('/')[-1].replace('_FRONT','')
    #     print(i)
    #     if new!=ori:
    #         print(new)
    #         print(ori)
    # print("finish")

    #-------------------------Test 
    # info_path = "/home/elodie/DATASET_INFO/infos_train.pkl"
    # with open(info_path, "rb") as f:
    #     infos = pickle.load(f)
    # for i in tqdm(range(len(infos))):
    #     info = infos[i]
    #     gt_boxes = info['gt_boxes']
    #     for box in gt_boxes:
    #         if box.shape[0]!=7:
    #             print("Error box shape:",box.shape[0])
    #             print(info["lidar_path"])
