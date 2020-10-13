import numpy as np
from det3d.datasets.utils.pointcloud_utils import *

def downsample_kitti(points, verticle_switch=True, horizontal_switch=True):
    assert points.shape[1] == 5, "Kitti points must have ring feature for downsample."
    if verticle_switch:
        ring_remained = [33, 32, 29, 27, 25, 23, 21, 19, 16, 14, 12, 10, 8, 6, 4, 2]
        points = points[np.in1d(points[:,-1],ring_remained)] # faster
    if horizontal_switch:
        distances = np.array(get_distances_2d(points)).T
        distances = np.fabs(distances[1:] - distances[:-1])
        np.append(distances,0)
        half_mask = np.arange(0,points.shape[0]-1,2)
        mask = np.ones(points.shape[0], dtype=bool)
        mask[half_mask[np.any((distances[half_mask] < 0.1,distances[half_mask-1] < 0.1),axis=0)]] = False
        points = points[mask]
    return points

def downsample_nusc_v2(points):
    if points.shape[1]!=5:
        print("points attribution do not contains ring num..")
        return points
    points_mask = np.all([points[:,4]>16 ,points[:,4]<26],axis=0)
    points = points[points_mask]
    return points

def upsample_nusc_v1(points):
    if points.shape[0]>0:
        distances = np.array(get_distances_2d(points)).T
        points_upsample = (points[:-1] + points[1:])/2
        mask = np.all((points[:-1,4]!=25, abs(distances[1:] - distances[:-1])<0.1, distances[1:]>0.1 , distances[:-1]>0.1),axis=0)
        points_upsample = points_upsample[mask]
        if points_upsample.shape[0]>0:
            points_upsample[:,4] = 0
            points = np.vstack((points, points_upsample))
    return points[:,:5]
