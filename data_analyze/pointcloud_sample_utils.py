import numpy as np
from pointcloud_utils import *

def add_ring_feature_kitti(points):
    if points.shape[1] == 4: # add feature ring
        horizontal_angles = get_horizontal_angle(points[:,0],points[:,1])
        ring = 1
        ring_list = [ring]
        for i in range(1,points.shape[0]):
            if horizontal_angles[i-1]<0 and horizontal_angles[i]>0:
                ring += 1
            ring_list.append(ring)
        points = np.hstack((points, np.array(ring_list).reshape(-1,1)))
    return points

def downsample_kitti(points):
    points = add_ring_feature_kitti(points)
    if points.shape[1] == 5:
        ring_remained = [33, 32, 29, 27, 25, 23, 21, 19, 16, 14, 12, 10, 8, 6, 4, 2]
        points_mask = np.array([point[-1] in ring_remained for point in points], dtype=np.bool_)
        points = points[points_mask]
    return points

def downsample_nusc_v2(points):
    if points.shape[1]!=5:
        print("points attribution do not contains ring num..")
        return points
    points_mask = np.all([points[:,4]>16 ,points[:,4]<26],axis=0)
    points = points[points_mask]
    return points

def upsample_nusc(points):
    if points.shape[0]>0:
        # points = np.hstack((points, get_vertical_angle(points[:,2], get_distances_2d(points))))
        points = np.hstack((points, np.array([get_distances_2d(points)]).T))
        points_upsample = np.array([(points[i-1] + points[i])/2 for i in range(points.shape[0]) if points[i,4]!=25 and abs(points[i,5]-points[i-1,5])<0.1 ])
        if points_upsample.shape[0]>0:
            points_upsample[:,4] = 0
            points = np.vstack((points, points_upsample))
    return points[:,:5]