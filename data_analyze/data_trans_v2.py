# Function： DataSet Transform
# Editor：ElodieShan
# Date:2020-05-20

############################
#
# 说明:　修改Nusc和Kitti数据集INFO + 修改kitti数据集数据
# 
# 1. 去除不需要的标注类别 KITTI/NUSC
# 2. 在info中增加预处理标签 KITTI/NUSC
# 3. 转换点云数据坐标系，统一至NUSC坐标系下 KITTI（data+gt）
# 4. 增加数据ring属性 KITTI（data+gt）
#
#############


import pickle
import numpy as np
import matplotlib.pyplot as plt
from det3d.core import box_np_ops
import os
import copy
from tqdm import tqdm
import random
from det3d.datasets.kitti import kitti_common as kitti
from det3d.datasets.pipelines.loading import read_file
from det3d.core.sampler import preprocess as prep
from nuscenes.nuscenes import NuScenes

from pointcloud_sample_utils import *

ALL_CLASS_LIST = ['car','truck','van','bus','trailer','tram','misc','constructed vehicle','pedestrian','person_sitting','bicycle','motorcycle','traffic cone','barrier','dontcare']

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

def double_check_gt_names(gt_names):
    #-------------------------------二次校验，以防止类别没有写完整
    gt_names2 = []
    flag = True
    for i in range(len(gt_names)):
        if gt_names[i] not in ALL_CLASS_LIST:
            flag = False
            print("Double check gt_names, find error class name [%s]"%gt_names[i])
            for j in range(len(ALL_CLASS_LIST)):
                if ALL_CLASS_LIST[j][:len(gt_names[i])] == gt_names[i]:

                    gt_names[i] = copy.deepcopy(ALL_CLASS_LIST[j])
                    gt_names2.append(ALL_CLASS_LIST[j])
                    print("successfuly correct class name as [%d, %s, %s]"%(i, gt_names2[-1],ALL_CLASS_LIST[j]))
                    break
        else:
            gt_names2.append(gt_names[i])
    return np.array(gt_names2), flag
####
# @function:create_nusc_info()
#
# @brief:
# 1. [Nusc数据集]　去除标注box内的速度信息 ndim=9--->7 [x,y,z,w,l,h,vx,vy,theta] --> [x,y,z,w,l,h,theta]
# 4. [Nusc数据集]  sweeps=[]、gt_boxes_velocity=[[0,0]]
# 5. [Nusc数据集]　info中增加字段"point_num_inbox"、"data_type"
# 6. [Nusc数据集]　info中增加字段{"preprocess_type":["sample":"True/False","x_flip":"True/False","y_flip":"True/False"}
#
# @ param nusc_infos:原先的info
###
def create_nusc_info(nusc_info_path, new_info_path, absolutely_path=None, sample_switch=True, flip_x_switch=True, flip_y_switch=False):
    print("Start creating nusc info...")

    VERSION = "v1.0-trainval"
    data_root = "/home/dataset/nuScenes_DATASET"
    nusc = NuScenes(version=VERSION, dataroot=str(data_root), verbose=True)
    
    print("Successfully load nusc info in %s"%nusc_info_path)
    with open(nusc_info_path, "rb") as f:
        nusc_infos = pickle.load(f)
    print("len nusc_infos",len(nusc_infos))
    
    infos_new = []
    ###--------------------------------循环遍历所有info
    for i in tqdm(range(len(nusc_infos))):
    # for i in tqdm(range(5)):
        info = nusc_infos[i]
        gt_boxes = info["gt_boxes"]
        #----------------------------------去除box的6/7列(速度信息)
        if len(gt_boxes)>0 and len(gt_boxes[0]) == 9:
            gt_boxes = np.delete(gt_boxes,6,axis=1)
            gt_boxes = np.delete(gt_boxes,6,axis=1)

        #----------------------------------box内点数
        point_num = np.array([nusc.get('sample_annotation', anno_token)["num_lidar_pts"] \
                            for anno_token in info["gt_boxes_token"]])

        #-----------------------------------修改info
        info["gt_boxes"] = gt_boxes
        info["sweeps"] = []
        info["gt_boxes_velocity"] = np.zeros([gt_boxes.shape[0],3])
        info["num_points_in_gt"] = point_num
        info["point_feature_num"] = 5
        # info["preprocess_type"] = {
        #     "sample":True,
        #     "x_flip":False,
        #     "y_flip":False
        # }
        if sample_switch:
            info["preprocess_type"]["sample"] = True
        if flip_x_switch:
            info["preprocess_type"]["x_flip"] = False
        if flip_y_switch:
            info["preprocess_type"]["y_flip"] = False

        info["data_type"] = "nuscenes"

        if absolutely_path is not None:
            info["lidar_path"] = absolutely_path + info["lidar_path"].split('/')[-1]
            info["cam_front_path"] = absolutely_path.replace('LIDAR_TOP/','') + '/'.join(info["cam_front_path"].split('/')[-2:])
        infos_new.append(info)

        if flip_x_switch:
            info2 = copy.deepcopy(info)
            info2["preprocess_type"]["x_flip"] = True
            infos_new.append(info2)

        if flip_y_switch:
            info3 = copy.deepcopy(info2)
            info3["preprocess_type"]["y_flip"] = True
            infos_new.append(info3)
            info4 = copy.deepcopy(info3)
            info4["preprocess_type"]["x_flip"] = False
            infos_new.append(info4)
        #------------------------------------打印进程，之后可以写成进度条
        if i%1000==0:
            print(i)
        
    #----------------------------------------打包info
    init_data_dir(new_info_path)
    with open(new_info_path, "wb") as f:
        pickle.dump(infos_new, f)
    print("Successfully dump new nusc info in %s"%new_info_path)

    #---------------------------------------打印第一个数据，查看是否正确
    print("\nnew_nusc_infos[0]:")
    with open(new_info_path, "rb") as f:
        new_nusc_infos = pickle.load(f)
    print(new_nusc_infos[0])

def create_nusc_gt_info(nusc_gt_info_path, new_gt_info_path,  point_feature_num=4, absolutely_path=None, flip_to_front_switch=True, del_velo_switch=True, create_pcd=True):
    print("Start creating nusc gt info...")
    with open(nusc_gt_info_path,"rb") as f:
        nusc_gt_infos = pickle.load(f)
    ori_gt_root = "/home/dataset/nuScenes_DATASET/gt_database_feature5_withoutvelo/"
    print("Successfully load nusc gt info...")
    new_nusc_gt_infos = {}
    for class_name, infos in nusc_gt_infos.items():
        new_nusc_gt_infos[class_name] = []
        # for i in tqdm(range(5)):
        for i in tqdm(range(len(infos))):
            info = copy.deepcopy(infos[i])
            if del_velo_switch and info["box3d_lidar"].shape[0] == 9:#elodie
                info["box3d_lidar"] = np.delete(info["box3d_lidar"],6)#elodie
                info["box3d_lidar"] = np.delete(info["box3d_lidar"],6)#elodie 

            if absolutely_path is not None:
                info["path"] = absolutely_path + info["path"].split('/')[-1]

            info["data_type"] = "nuscenes"
            info["point_feature_num"] = point_feature_num

            if flip_to_front_switch and info["box3d_lidar"][1]<0:
                ori_path = ori_gt_root + info["path"].split('/')[-1] 
                boxes = np.array([info["box3d_lidar"]])
                points = np.fromfile(ori_path, dtype=np.float32).reshape([-1, info["point_feature_num"]])
                boxes, points = prep.flip_XY(boxes, points,\
                        x_filp_switch=True, y_filp_switch=False)
                info["box3d_lidar"] = boxes[0]
                if create_pcd:
                    if i == 0:
                        init_data_dir(info["path"])
                    points.astype(np.float32).tofile(info["path"])
            else:
                if create_pcd:
                    if i == 0:
                        init_data_dir(info["path"])
                    ori_path = ori_gt_root + info["path"].split('/')[-1] 
                    points = np.fromfile(ori_path, dtype=np.float32).reshape([-1, info["point_feature_num"]])
                    points.astype(np.float32).tofile(info["path"])

            new_nusc_gt_infos[class_name].append(info)

    init_data_dir(new_gt_info_path)
    with open(new_gt_info_path, "wb") as f:
        pickle.dump(new_nusc_gt_infos, f)    
    print("Successfully dump new nusc gt info in %s"%new_gt_info_path)

    print("\ngt_new_nusc_infos[car][0]:")
    with open(new_gt_info_path, "rb") as f:
        gt_new_nusc_infos = pickle.load(f)
    print(gt_new_nusc_infos['car'][0])

####
# @function:create_dataset_kitti2nusc_format()
#
# @brief: 将kitti数据集转换为nusc格式，包括坐标系转换、info格式（保留kitti原info内容）
# 1. [Kitti数据集]　将kitti的anno格式对齐至nusc anno格式，同时保留原kitti anno信息
# 2. [Kitti数据集]　将kitti坐标系对齐至nusc坐标系,并生成新的pcd数据
# 3. [Kitti数据集]　点云数据与Ground truth datasee中都增加一个ring通道号属性 [x,y,z,intensity]  ----> [x,y,z,intensity,ring]
# 4. [Kitti数据集]　info中增加字段{"preprocess_type":["sample":"True","x_flip":"True/False","y_flip":"False"}
# 5. [Kitti数据集]　info中增加字段"data_type"="kitti","num_points_in_gt"=[n,n,n,n,n.....]
# 6. [Kitti数据集]　类别对齐 [kitti] Cyclist -> bicycle、Van -> truck,其他class小写即可
# 7*.[Kitti数据集]　类别对齐 [kitti] Truck box length<8m -> truck、Truck box length>8m -> bus


def create_dataset_kitti2nusc_format(
    kitti_infos_path, 
    new_info_path, 
    absolutely_root_path=None, 
    add_ring_switch=True, 
    flip_y_switch=False, 
    create_pcd=False, 
    truck_to_bus_switch=False,
    van_to_truck_switch=False):

    print("Start creating kitti info...")
    with open(kitti_infos_path, "rb") as f:
        kitti_infos = pickle.load(f)
    print("Successfully load %d kitti info..."%len(kitti_infos))
    kitti_infos_new = []
    #----------------------------------旋转矩阵，旋转90度
    R = np.array([[ 0, -1 ,0],
        [ 1,0,0],
        [ 0,0,1]])

    assert absolutely_root_path is not None, "pealse input absolutely_root_path.."

    ori_velodyne_root_path = "/home/dataset/KITTI_DATASET/object/"
    # for i in tqdm(range(5)):
    for i in tqdm(range(len(kitti_infos))):
        kitti_info = kitti_infos[i]
        ref_lidar_path = absolutely_root_path + kitti_info["point_cloud"]["velodyne_path"] #绝对路径

        ref_cam_path = absolutely_root_path + kitti_info["image"]["image_path"] #绝对路径
        
        cam_intrinsic_pre = kitti_info["calib"]["R0_rect"]
        cam_intrinsic_pre = np.delete(cam_intrinsic_pre, 3, axis=0)
        cam_intrinsic_pre = np.delete(cam_intrinsic_pre, 3, axis=1)
        ref_cam_intrinsic = cam_intrinsic_pre

        token = kitti_info["image"]["image_idx"]
        ref_from_car = kitti_info["calib"]["Tr_velo_to_cam"]
        car_from_global = kitti_info["calib"]["P0"]

        #-------------------------------读取点云数据
        point_feature_num = kitti_info["point_cloud"]["num_features"]
        points = np.fromfile(ori_velodyne_root_path + kitti_info["point_cloud"]["velodyne_path"], dtype=np.float32, count=-1).reshape([-1, point_feature_num])
        if add_ring_switch:
            points = add_ring_feature_kitti(points)
            point_feature_num += 1

        if create_pcd:
            if i == 0:
                init_data_dir(ref_lidar_path)
            if os.path.exists(ref_lidar_path) is not True:
                #-------------------------------顺时针旋转90度并保存为bin
                points_xy = np.hstack((points[:,:2],np.ones((points.shape[0],1))))
                points_theta = np.dot(R,points_xy.T).T
                points_theta = np.hstack((points_theta[:,:2],points[:,2:]))
                points_theta.astype(np.float32).tofile(ref_lidar_path)

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

            #------------------------------标注类别转为小写，Cyclist->bicycle,与Nusc对齐
            gt_names_pre = annos["name"]
            gt_names = []
            for i in range(len(gt_names_pre)):
                name_pre = gt_names_pre[i]
                name = ""
                if  name_pre == 'Cyclist':
                    name = 'bicycle'
                elif  name_pre == 'Truck':
                    if truck_to_bus_switch:
                        # kitti的truck分两类，长小于8m的归于truck，长大于8m的归于bus。
                        box_length = gt_boxes[i][4]
                        if box_length<=8:
                            name = 'truck'
                        else:
                            name = "bus"
                    else:
                        name = 'truck'     
                elif  name_pre == 'Van':
                    if van_to_truck_switch:
                        name = 'truck'
                    else:
                        name = 'van'
                else:              
                    name = name_pre.lower()
                gt_names.append(name)
            gt_names = np.array(gt_names)

            #-------------------------------二次校验，以防止类别没有写完整
            check_time = 0
            flag = False
            while flag is not True:     
                check_time += 1
                gt_names, flag = double_check_gt_names(gt_names)
                if check_time > 1:
                    print("check_time:", check_time)

        kitti_info["annos"]['name'] = gt_names #对kitti val产生影响

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
            "gt_names": gt_names,
            "gt_boxes_token": np.zeros(gt_boxes.shape[0]),

            "image":kitti_info["image"],
            "point_cloud":kitti_info["point_cloud"],
            "calib":kitti_info["calib"],
            "annos":kitti_info["annos"],
            "num_points_in_gt":kitti_info["annos"]["num_points_in_gt"],
            "point_feature_num": point_feature_num,
            "data_type": "kitti",
        }
        info["preprocess_type"] = {
            "sample":True,
            "x_flip":False,
        }
        kitti_infos_new.append(info)
        if flip_y_switch:
            info["preprocess_type"]["y_flip"] = False

            info2 = copy.deepcopy(info)
            info["preprocess_type"]["y_flip"] = True
            kitti_infos_new.append(info2)

    init_data_dir(new_info_path)
    with open(new_info_path, "wb") as f:
        pickle.dump(kitti_infos_new, f)
    print("Successfully dump new kitti info in %s"%new_info_path)

    #---------------------------------------打印第一个数据，查看是否正确
    print("\nkitti_infos_new[0]:")
    with open(new_info_path, "rb") as f:
        kitti_infos_new = pickle.load(f)

    print("\nkitti_infos_new[0]:\n",kitti_infos_new[0])

    points = np.fromfile(kitti_infos_new[0]["lidar_path"], dtype=np.float32, count=-1).reshape([-1, kitti_infos_new[0]["point_feature_num"]])
    print("\nkitti_infos_new[0]- points:\n",points[:5])

def change_gt_database_kitti2nusc_format(
    kitti_db_infos_path, 
    new_gt_info_path, 
    point_feature_num=4, 
    absolutely_root_path=None, 
    add_ring_switch=True, 
    create_pcd=False, 
    truck_to_bus_switch=False,
    van_to_truck_switch=False,
    ):

    print("Start creating kitti ground truth info...")
    with open(kitti_db_infos_path, "rb") as f:
        kitti_db_infos = pickle.load(f)
    print("Successfully load kitti gt info...")
    
    kitti_db_infos_new = {}
    #----------------------------------旋转矩阵，旋转90度
    R = np.array([[ 0, -1 ,0],
        [ 1,0,0],
        [ 0,0,1]])

    velodyne_root_path = "/home/dataset/KITTI_DATASET/object/"

    assert absolutely_root_path is not None, "pealse input kitti ground truth absolutely_root_path.."
    new_root_path = absolutely_root_path
    
    kitti_class_list = kitti_db_infos.keys()
    for kitti_class in kitti_class_list:
        kitti_infos = kitti_db_infos[kitti_class]
        kitti_infos_new =[]

        #------------------------------标注类别转为小写，Cyclist->bicycle,与Nusc对齐  
        kitti_class_new = ""
        if  kitti_class == 'Cyclist':
            kitti_class_new = 'bicycle'
        elif  kitti_class == 'Van':
            if van_to_truck_switch: # v3_1 elodie
                kitti_class_new = 'truck'
            else:
                kitti_class_new = 'van'          
        else:
            kitti_class_new = kitti_class.lower()
        
        # for i in tqdm(range(5)):
        for i in tqdm(range(len(kitti_infos))):
            kitti_info = kitti_infos[i]

            db_lidar_path = velodyne_root_path + kitti_info["path"] #绝对路径
            #修改gt文件路径为绝对路径
            file_name = str(kitti_info["path"]).split('/')[-1]
            db_lidar_path_new = new_root_path + file_name
            kitti_info["path"] = db_lidar_path_new
            if create_pcd:
                #-------------------------------读取点云数据，逆时针旋转90度并保存为bin
                points = np.fromfile(db_lidar_path, dtype=np.float32, count=-1).reshape([-1, point_feature_num])
                if add_ring_switch:
                    points = add_ring_feature_kitti(points)
                    point_feature_num += 1
                points_xy = np.hstack((points[:,:2],np.ones((points.shape[0],1))))
                points_theta = np.dot(R,points_xy.T).T
                points_theta = np.hstack((points_theta[:,:2],points[:,2:]))
                if i==0:
                    init_data_dir(db_lidar_path_new)
                points_theta.astype(np.float32).tofile(db_lidar_path_new)
            
            #------------------------------调整坐标系,将gt_boxes 逆时针旋转90°
            gt_boxes = np.array([copy.deepcopy(kitti_info["box3d_lidar"])])
            gt_boxes_theta = np.dot(R, gt_boxes[:, :3].T).T
            gt_boxes = np.hstack((gt_boxes_theta, gt_boxes[:, 3:]))
            gt_boxes[:, 6] = gt_boxes[:, 6] - np.sign(gt_boxes[:, 6]) * np.pi / 2

            kitti_info["box3d_lidar"] = gt_boxes[0]
            kitti_info["data_type"] = "kitti"
            kitti_info["point_feature_num"] = point_feature_num
            kitti_info["name"] = kitti_class_new

            if truck_to_bus_switch and kitti_class.lower() == 'truck':
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

    init_data_dir(new_gt_info_path)
    with open(new_gt_info_path, "wb") as f:
        pickle.dump(kitti_db_infos_new, f)
    print("Successfully dump new nusc gt info in %s"%new_gt_info_path)

    print("\ngt_new_nusc_infos[car][0]:")
    with open(new_gt_info_path, "rb") as f:
        gt_new_kitti_infos = pickle.load(f)
    print(gt_new_kitti_infos['car'][0])

    points = np.fromfile(gt_new_kitti_infos['car'][0]["path"], dtype=np.float32, count=-1).reshape([-1, gt_new_kitti_infos['car'][0]["point_feature_num"]])
    print("\ngt_new_kitti_infos['car'][0]- points:\n",points[:5])

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
def mix_kitti_nusc(kitti_infos_path, nusc_infos_path, muti_seed, new_info_path):
    with open(kitti_infos_path, "rb") as f:
        kitti_infos = pickle.load(f)
    print("Successfully load %d kitti info..."%len(kitti_infos))

    with open(nusc_infos_path, "rb") as f:
        nusc_infos = pickle.load(f)
    print("Successfully load %d nusc info..."%len(nusc_infos))

    infos_new = kitti_infos
    for i in range(muti_seed-1):
        infos_new = infos_new + kitti_infos
    infos_new = infos_new + nusc_infos
    random.shuffle(infos_new)

    print("muti_seed:",muti_seed)
    print("new info num:",len(infos_new))
    # ----------------------------------根据点和类别对box进行筛选，直接删除
    class_list = ['car', 'truck', 'bus', 'trailer','pedestrian','motorcycle', 'bicycle','traffic_cone']    #保留的类别
    point_num_thred = 4
    infos_new_filtered = []
    for i in tqdm(range(len(infos_new))):
        info = infos_new[i]
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

        infos_new_filtered.append(info)
    with open(new_info_path, "wb") as f:
        pickle.dump(infos_new_filtered, f)
    print("Successfully dump mix info in %s"%new_info_path)

def mix_gt_kitti_nusc(kitti_infos_path, nusc_infos_path, new_info_path):     
    with open(kitti_infos_path, "rb") as f:
        kitti_infos = pickle.load(f)
    print("Successfully load kitti gt info...")    

    with open(nusc_infos_path, "rb") as f:
        nusc_infos = pickle.load(f)
    print("Successfully load kitti gt info...")    

    new_infos = nusc_infos

    nucs_classes = new_infos.keys()

    for class_name in kitti_infos.keys():
        if class_name in nucs_classes:
            new_infos[class_name] += kitti_infos[class_name]

    init_data_dir(new_info_path)
    with open(new_info_path, "wb") as f:
        pickle.dump(new_infos, f)
    print("Successfully dump mix gt info in %s"%new_info_path)

if __name__ == '__main__':
    Process_List = [
        # "CreateNuscInfo",
        # "CreateNuscGTInfo",
        # "CreateKittiInfo",
        # "CreateKittiValInfo",
        "CreateKittiGTInfo",
        # "CreateMixInfo",
        # "CreateMixGTInfo",
    ]

    #-----------------------NUSC ANNO INFO
    if "CreateNuscInfo" in Process_List:
        nusc_path = "/home/elodie/nuScenes_DATASET/pkl/infos_train_10sweeps_withvelo.pkl"
        new_info_path = "/home/elodie/nuScenes_DATASET/pkl_new/infos_train_extrav3.pkl"
        absolutely_lidar_path = "/home/dataset/nuScenes_DATASET/samples/LIDAR_TOP/"
        create_nusc_info(nusc_path, new_info_path, absolutely_path=absolutely_lidar_path,\
        sample_switch=True, flip_x_switch=True, flip_y_switch=False)

    #-----------------------NUSC GT INFO
    if "CreateNuscGTInfo" in Process_List:
        nusc_gt_info_path = "/home/dataset/nuScenes_DATASET/pkl/dbinfos_train_feature5_withoutvelo.pkl"
        new_gt_info_path = "/home/dataset/nuScenes_DATASET/pkl_new/dbinfos_train_front_feature5_withoutvelo.pkl"
        absolutely_data_path = "/home/dataset/nuScenes_DATASET/gt_database_front_feature5_withoutvelo/"
        gt_point_feature_num = 5
        create_nusc_gt_info(nusc_gt_info_path, new_gt_info_path, \
            point_feature_num=gt_point_feature_num, absolutely_path=absolutely_data_path, \
            flip_to_front_switch=True, del_velo_switch=True)
    #-----------------------KITTI ANNO INFO to NUSC Anno Format
    if "CreateKittiInfo" in Process_List:
        kitti_absolutely_root_path = "/home/dataset/KITTI_DATASET_NEW/object/"
        kitti_info_path = "/home/dataset/KITTI_DATASET/object/kitti_infos_train.pkl"
        kitti_info_path_new = kitti_absolutely_root_path + "pkl/kitti_infos_train_formatnusc_feature5_v3_1.pkl"
        
        # create_dataset_kitti2nusc_format(kitti_info_path, kitti_info_path_new, \
        #     absolutely_root_path=kitti_absolutely_root_path, \
        #     add_ring_switch=True, flip_y_switch=False, create_pcd=False, \
        #     truck_to_bus_switch=False, van_to_truck_switch=False)
        # with open("/home/dataset/KITTI_DATASET_NEW/object/pkl/kitti_infos_train_formatnusc_feature5_v3_back.pkl", "rb") as f:
        #     infos = pickle.load(f)
        # for i in range(len(infos)):
        #     infos[i]["annos"]['name'] = infos[i]["gt_names"]
        # with open("/home/dataset/KITTI_DATASET_NEW/object/pkl/kitti_infos_train_formatnusc_feature5_v3.pkl", "wb") as f:
        #     pickle.dump(infos, f)
        with open("/home/dataset/KITTI_DATASET_NEW/object/pkl/kitti_infos_train_formatnusc_feature5_v3_1.pkl", "rb") as f:
            infos = pickle.load(f)
        for i in range(len(infos)):
            for j in range(len(infos[i]["gt_names"])):
                if infos[i]["gt_names"][j] == 'van':
                    infos[i]["gt_names"][j] = 'car'
            infos[i]["annos"]['name'] = infos[i]["gt_names"]
        with open("/home/dataset/KITTI_DATASET_NEW/object/pkl/kitti_infos_train_formatnusc_feature5_v3_2.pkl", "wb") as f:
            pickle.dump(infos, f)
    #-----------------------KITTI ANNO Val INFO to NUSC Anno Format
    if "CreateKittiValInfo" in Process_List:
        kitti_val_info_path = "/home/dataset/KITTI_DATASET/object/kitti_infos_val.pkl"
        kitti_val_info_path_new = "/home/dataset/KITTI_DATASET_NEW/object/pkl/kitti_infos_val_formatnusc_feature5_v3_1.pkl"
        kitti_absolutely_root_path = "/home/dataset/KITTI_DATASET_NEW/object/"
        # create_dataset_kitti2nusc_format(kitti_val_info_path, kitti_val_info_path_new, \
        #             absolutely_root_path=kitti_absolutely_root_path, add_ring_switch=True, create_pcd=False, \
        #             flip_y_switch=False, truck_to_bus_switch=False, van_to_truck_switch=False)
        with open("/home/dataset/KITTI_DATASET_NEW/object/pkl/kitti_infos_val_formatnusc_feature5_v3_1.pkl", "rb") as f:
            infos = pickle.load(f)
        for i in range(len(infos)):
            for j in range(len(infos[i]["gt_names"])):
                if infos[i]["gt_names"][j] == 'van':
                    infos[i]["gt_names"][j] = 'car'
            infos[i]["annos"]['name'] = infos[i]["gt_names"]
        with open("/home/dataset/KITTI_DATASET_NEW/object/pkl/kitti_infos_val_formatnusc_feature5_v3_2.pkl", "wb") as f:
            pickle.dump(infos, f)
    #-----------------------KITTI Ground Truth Info
    if "CreateKittiGTInfo" in Process_List:
        kitti_gt_info_path = "/home/dataset/KITTI_DATASET/object/dbinfos_train_feature5.pkl"
        kitti_gt_info_path_new = "/home/dataset/KITTI_DATASET_NEW/object/pkl/dbinfos_train_axisnusc_feature5_v3_1.pkl"
        kitti_gt_absolutely_root_path = "/home/dataset/KITTI_DATASET_NEW/object/gt_database_axisnusc_feature5/"
        change_gt_database_kitti2nusc_format(kitti_gt_info_path, kitti_gt_info_path_new, \
            point_feature_num=5, absolutely_root_path=kitti_gt_absolutely_root_path, \
            add_ring_switch=False, create_pcd=False, truck_to_bus_switch=False)

    #----------------------Mix KITTI&NUSC ANNO
    if "CreateMixInfo" in Process_List:
        kitti_info_path = "/home/dataset/KITTI_DATASET_NEW/object/pkl/kitti_infos_train_formatnusc_feature5_v3.pkl"
        nusc_info_path = "/home/elodie/nuScenes_DATASET/pkl_new/infos_train_extrav3.pkl"
        new_info_path = "/home/elodie/nuScenes_DATASET/pkl_mix/infos_train_v3.pkl"
        muti_seed = 3
        mix_kitti_nusc(kitti_info_path, nusc_info_path, muti_seed, new_info_path)

        with open(new_info_path,"rb") as f:
            infos = pickle.load(f)
        infos = infos[:500]
        with open("/home/elodie/nuScenes_DATASET/pkl_new/infos_extrav3_temp.pkl", "wb") as f:
            pickle.dump(infos, f)
        print("Successfully dump mix gt info in %s"%new_info_path)
    #---------------------Mix KITTI & NUSC GT 
    if "CreateMixGTInfo" in Process_List:
        kitti_gt_info_path = "/home/dataset/KITTI_DATASET_NEW/object/pkl/dbinfos_train_axisnusc_feature5.pkl"
        nusc_gt_info_path = "/home/dataset/nuScenes_DATASET/pkl_v2/dbinfos_train_front_feature5_withoutvelo.pkl"
        new_gt_info_path = "/home/elodie/nuScenes_DATASET/pkl_mix/dbinfos_train_v2.pkl"
        mix_gt_kitti_nusc(kitti_gt_info_path, nusc_gt_info_path, new_gt_info_path)

