# Function： DataSet boxes analyse
# Editor：ElodieShan
# Date:2020-05-01

############################
#
# 说明:　分析nusc和kitti的数据中标注数据的box信息
#
##########################＃

import pickle
import numpy as np
import matplotlib.pyplot as plt
from det3d.core import box_np_ops
import os

# @brief 加载info数据
def load_info(info_path):
    with open(info_path, "rb") as f:
        infos = pickle.load(f)
    return infos

# @brief 绘制直方图
def get_hist_pic_single(data, num_bins, x_lable, class_name, data_type, max_distance):
    hist,bins = np.histogram(data,num_bins) #return:频数，分箱的边界

    hist_max = max(hist)
    hist_max_index = hist.tolist().index(hist_max)
    avg = np.average(data)
    x = [(bins[i+1]+bins[i])/2 for i in range(len(bins)-1)]    
    msg = "max = " + str(hist_max) + " in " + str(round(x[hist_max_index],2)) + \
    " ∈ [" + str(round(bins[hist_max_index],2)) + ":" + str(round(bins[hist_max_index+1],2)) + "],  average = " + str(round(avg,2)) 

    plt.bar(x,hist,width=10/sum(hist!=0))
    plt.xlabel(x_lable)
    plt.ylabel('num')
    plt.ylim([0,hist_max + int(hist_max/5)])
    plt.title("[" + data_type + "] " + class_name + ": " +  x_lable + " in 0 - " + str(max_distance) + "m" + '\n' + msg)

    savefig_dir = "./result/"+ data_type + "/" + class_name + "/" + str(max_distance) + "m/"
    if not os.path.exists(savefig_dir):
        os.makedirs(savefig_dir)
    
    plt.savefig( savefig_dir + data_type + "_" + class_name + "_" +  x_lable + "_" + str(max_distance) + "m.jpg")
    plt.show()
    plt.cla()

    return round(x[hist_max_index],2), round(avg,2)

# @brief 绘制直方图
def get_hist_pic(ax, data, num_bins, x_label, class_name, data_type, max_distance):
    hist,bins = np.histogram(data,num_bins) #return:频数，分箱的边界

    hist_max = max(hist)
    hist_max_index = hist.tolist().index(hist_max)
    avg = np.average(data)
    x = [(bins[i+1]+bins[i])/2 for i in range(len(bins)-1)]    
    msg = "max = " + str(hist_max) + " in " + str(round(x[hist_max_index],2)) + \
    " ∈ [" + str(round(bins[hist_max_index],2)) + ":" + str(round(bins[hist_max_index+1],2)) + "],  average = " + str(round(avg,2)) 

    ax.bar(x,hist,width=5/sum(hist!=0))
    ax.set_title("[" + data_type + "] " + class_name + ": " +  x_label + " in 0 - " + \
        str(max_distance) + "m" + '\n' + msg, fontsize = 20)
    if x_label in ['width','height','length']:
        ax.set_xlabel(x_label + '/m', fontsize = 20 )
    elif x_label in ['length_width-ratio','point_num']:
        ax.set_xlabel(x_label, fontsize = 20)
    else:
        ax.set_xlabel('distance/m', fontsize = 20)
    ax.tick_params(labelsize=15)
    ax.set_ylabel('box num', fontsize = 20)
    ax.set_ylim([0,hist_max + int(hist_max/5)])

    return round(x[hist_max_index],2), round(avg,2)
### @function: get_gt_box_info()
#
# @brief 获取所有box的信息。需要输入db info，否则不适用
#
# @param info:需要输入db info，否则不适用
# @param class_list:待分析类别
# @param max_distance:待分析box的最远距离
#
# @return 
# gt_boxes 类别DICT:　｛class_name1:[info],class_name2:[info]……}
# gt_boxes 类别DICT:　｛class_name1:[boxes],class_name2:[boxes]……｝
###
def get_gt_info(info, class_list, max_distance):
    gt_boxes = {}
    gt_infos = {}
    for class_name in class_list:
        filtered_info = [gt_info for gt_info in info[class_name] 
                if get_distance_2d(gt_info["box3d_lidar"][0],gt_info["box3d_lidar"][1]) < max_distance]
        filtered_box = [box_info["box3d_lidar"] for box_info in filtered_info]
        # filtered_box = []
        # for box_info in filtered_info:
        #     if box_info["box3d_lidar"][4] >8 and  box_info["box3d_lidar"] [4]<10:
        #         filtered_box.append(box_info["box3d_lidar"])
        gt_infos[class_name] = np.array(filtered_info)
        gt_boxes[class_name] = np.array(filtered_box)
    return gt_infos, gt_boxes

def get_distance_2d(x,y):
    return np.sqrt(np.sum([x**2,y**2]))

### @function: get_size_distribution()
#
# @brief 获取所有box的长、宽、高、长宽比信息。
#
# @param gt_boxes:需要输入gt_boxes dict，否则不适用
# @param data_type: 数据集名称　Nuscenes or KITTI
# @param num_bins:分频数，直方图时使用
###

def get_size_distribution(gt_boxes, data_type, max_distance, num_bins):
    x_label = "length_width_ratio"
    for class_name, boxes in gt_boxes.items():
        #使用subplots 画图
        f, ax = plt.subplots(2,2,figsize=(14, 12))
        # f.tight_layout()#调整整体空白
        plt.subplots_adjust(wspace=0.3, hspace = 0.4)#调整子图间距

        gt_box_lw_ratio = boxes[:,4]/boxes[:,3]
        # -------------KITTI - box dimensions: lhw(camera) format
        width_freq_max, width_avg = get_hist_pic(ax[0][0], boxes[:,3], num_bins, 'width', class_name, data_type, max_distance)
        length_freq_max, length_avg = get_hist_pic(ax[0][1], boxes[:,4], num_bins, 'length', class_name, data_type, max_distance)
        height_freq_max, height_avg = get_hist_pic(ax[1][0], boxes[:,5], num_bins, 'height', class_name, data_type, max_distance)
        l_w_freq_max, l_w_avg = get_hist_pic(ax[1][1], gt_box_lw_ratio, num_bins, 'length_width-ratio', class_name, data_type, max_distance)
        savefig_dir = "./result/"+ data_type + "/" + class_name + "/" + str(max_distance) + "m/"
        if not os.path.exists(savefig_dir):
            os.makedirs(savefig_dir)
    
        plt.savefig( savefig_dir + data_type + "_" + class_name + "_" + str(max_distance) + "m.jpg")
        plt.show()
        plt.cla()

        print('MAX - {} : [{:.2f}, {:.2f}, {:.2f}] {:.2f}/{:.2f}={:.2f}  l_w_freq:{:.2f}'.format(class_name, width_freq_max, length_freq_max, 
            height_freq_max, length_freq_max, width_freq_max, round(length_freq_max/width_freq_max,2), l_w_freq_max))
        print('AVG - {} : [{:.2f}, {:.2f}, {:.2f}] {:.2f}/{:.2f}={:.2f}  l_w_freq:{:.2f}'.format(class_name, width_avg, length_avg, \
            height_avg, length_avg, width_avg, round(length_avg/width_avg,2), l_w_avg))
        print()

def get_points_num_distribution(gt_infos, gt_boxes, data_type, max_point_num, num_bins): 
    #start analyza
    x_label = "point_num"
    for class_name, class_info in gt_infos.items():
        num_points_in_gt = [int(info['num_points_in_gt']) for info in class_info]

        mask = np.array([num_points<max_point_num
                                 for num_points in num_points_in_gt], dtype=np.bool_)

        num_points_in_gt = np.array(num_points_in_gt)[mask]

        f, ax = plt.subplots(2,3,figsize=(25, 12))
        plt.subplots_adjust(left=0.1, wspace=1, hspace = 0.4)#调整子图间距
        point_num_freq_max, point_num_avg = get_hist_pic(ax[0][0], num_points_in_gt, num_bins, 'point_num', class_name, data_type, max_distance)

        for n in range(5):
            # 不同点数的box的距离分布
            mask_temp = np.array([num_points == (n+1)
                                 for num_points in num_points_in_gt], dtype=np.bool_)
            box = np.array(gt_boxes[class_name])[mask][mask_temp]
            distance = np.sqrt(np.sum([box[:,0]**2,\
                box[:,1]**2], axis=0))
            ax_row = int((n+1)/3)
            ax_col = (n+1)%3
            x_label_new = "box of "+str(n+1)+" points distribution"
            point_num_freq_max, point_num_avg = get_hist_pic(ax[ax_row][ax_col], distance, num_bins, x_label_new, class_name, data_type, max_distance)

        savefig_dir = "./result/"+ data_type + "/" + class_name + "/" + str(max_distance) + "m/"
        if not os.path.exists(savefig_dir):
            os.makedirs(savefig_dir)
    
        plt.savefig( savefig_dir + data_type + "_" + class_name + "_point_num.jpg")
        plt.show()
        plt.cla()

        # plt.cla()

if __name__ == '__main__':
    db_info_path = "/home/elodie/KITTI_DATASET/object/dbinfos_train.pkl"
    data_type = "KITTI"
    class_list = ['Car', 'Truck', 'Van', 'Tram', 'Misc', 'Pedestrian', 'Person_sitting', 'Cyclist']    #待分析的类别
    # db_info_path = "/home/elodie/nuScenes_DATASET/pkl/dbinfos_train_10sweeps_withvelo.pkl"
    # data_type = "NUSC"
    # class_list = ['car', 'truck', 'bus', 'trailer','pedestrian','motorcycle', 'bicycle','traffic_cone','barrier','construction_vehicle']    #待分析的类别
    # class_list = ['Truck']
    max_distance = 100
    num_bins = 100
    max_point_num = 10
    analyze_mode = "size"

    info = load_info(db_info_path)
    gt_infos, gt_boxes = get_gt_info(info, class_list, max_distance)

    if analyze_mode == "size":
        get_size_distribution(gt_boxes, data_type, max_distance, num_bins)
    elif analyze_mode == "point_num":
        get_points_num_distribution(gt_infos, gt_boxes, data_type, max_point_num, 20)