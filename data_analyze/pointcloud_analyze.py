import numpy as np

""" Function: get_distances_3d
@breif 多点根据x、y计算水平距离，
@param x、y输入类型是np.array
"""
def get_distances_2d(points):
    return np.sqrt(np.sum([points[:,0]**2,points[:,1]**2],axis=0))

def get_vertical_angle(z, distance2d):
    return np.arctan2(z,distance2d)/np.pi*180

if __name__ == '__main__':
    points_path_root = "/home/elodie/nuScenes_DATASET_NEW/samples/LIDAR_TOP/"
    points_file ="n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530449377_FRONT.pcd.bin"
    points_path = points_path_root + points_file
    points = np.fromfile(points_path,dtype=np.float32).reshape([-1, 4])

    vertical_angles = get_vertical_angle(points[:,2], get_distances_2d(points))
    vertical_angles_set = [round(angle,1) for angle in vertical_angles]
    print(len(vertical_angles_set))
    vertical_angles_set = sorted(set(vertical_angles_set))
    print(vertical_angles_set)
    print(len(vertical_angles_set))
