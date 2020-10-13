import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.image import imread
from det3d.core import box_np_ops
import os
colors = {
    'Car': 'b',
    'Tram': 'r',
    'Cyclist': 'g',
    'Van': 'c',
    'Truck': 'm',
    'Pedestrian': 'y',
    'Sitter': 'k'
}

tsp_axes_limits = [
    [-40, 40], # X axis range
    [0, 60], # Y axis range
    [-2, 10]   # Z axis range
]

velo_axes_limis = [
    [0, 60], # X axis range
    [-40, 40], # Y axis range
    [-2, 10]   # Z axis range
]

axes_str = ['X', 'Y', 'Z']

# @brief: 查看路劲的上级文件是否存在，若不存在则逐层创建
def init_data_dir(path):
    file_root_path, file_name=os.path.split(path)
    if not os.path.exists(file_root_path):
        try:
            os.makedirs(file_root_path)
        except:
            print("Error occurs when make dirs %s"%file_root_path)

def draw_box(pyplot_axis, vertices, label=None, score=None, axes=[0, 1, 2], color='black'):
    """
    Draws a bounding 3D box in a pyplot axis.
    
    Parameters
    ----------
    pyplot_axis : Pyplot axis to draw in.
    vertices    : Array 8 box vertices containing x, y, z coordinates.
    axes        : Axes to use. Defaults to `[0, 1, 2]`, e.g. x, y and z axes.
    color       : Drawing color. Defaults to `black`.
    """
    vertices = np.transpose(vertices)[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
        [0, 4], [1, 5], [2, 6], [3, 7]  # Connections between upper and lower planes
    ]
    if color == "green":
        line_width = 2
    else:
        line_width = 1
        
    for connection in connections:
        pyplot_axis.plot(*vertices[:, connection], c=color, lw=line_width)

    if score is not None:
        label_scores = label + " " + str(score)
    else:
        label_scores = label

    if label_scores is not None:
        if len(axes) > 2:
            pyplot_axis.text(min(vertices[0]), max(vertices[1]),max(vertices[2]), label_scores)
        else:
            pyplot_axis.text(min(vertices[0]), max(vertices[1]), label_scores)


def get_points_inbox(rbbox_corners, points):
    surfaces = box_np_ops.corner_to_surfaces_3d(rbbox_corners)
    indices = box_np_ops.points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    points_inbox_indices = indices.any(axis=1)
    return points_inbox_indices

"""
Convenient method for drawing various point cloud projections as a part of frame statistics.
"""
def draw_point_cloud(ax, data, boxes, labels=None, scores=None, title=None, gt_boxes=None, gt_labels=None, axes=[0, 1, 2], xlim3d=None, ylim3d=None, zlim3d=None, point_size=0.1, view_points_in_box=True, coordinates="tensorpro"):
    if coordinates == "velodyne" or "kitti":
        axes_limits = velo_axes_limis
        color_axe = 0
    else:
        axes_limits = tsp_axes_limits
        color_axe = 1

    if view_points_in_box:
        points_inbox_indices = get_points_inbox(boxes, data)
        ax.scatter(*np.transpose(data[points_inbox_indices][:, axes]), s=point_size+0.05, c='r')
        ax.scatter(*np.transpose(data[~points_inbox_indices][:, axes]), s=point_size, c=data[~points_inbox_indices][:, color_axe], cmap='terrain')
    else:
        ax.scatter(*np.transpose(data[:, axes]), s=point_size, c=data[:, color_axe])

    ax.set_title(title)
    ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
    ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
    if len(axes) > 2:
        ax.set_xlim3d(*axes_limits[axes[0]])
        ax.set_ylim3d(*axes_limits[axes[1]])
        ax.set_zlim3d(*axes_limits[axes[2]])
        ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
    else:
        ax.set_xlim(*axes_limits[axes[0]])
        ax.set_ylim(*axes_limits[axes[1]])
    # User specified limits
    if xlim3d!=None:
        ax.set_xlim3d(xlim3d)
    if ylim3d!=None:
        ax.set_ylim3d(ylim3d)
    if zlim3d!=None:
        ax.set_zlim3d(zlim3d)

    if gt_boxes is not None:
        for i in range(gt_boxes.shape[0]):
            draw_box(ax, gt_boxes[i], axes=axes, color='green')
        for i in range(boxes.shape[0]):
            draw_box(ax, boxes[i], axes=axes, color='blue')
    else:
        for i in range(boxes.shape[0]):
            if scores is not None:
                draw_box(ax, boxes[i], labels[i], scores[i], axes=axes, color='blue')
            else:
                if labels is not None:
                    draw_box(ax, boxes[i], labels[i], axes=axes, color='blue')
                else:
                    draw_box(ax, boxes[i], axes=axes, color='blue')

def display_pred_and_gt(save_path, points, boxes_dt=None, boxes_gt=None, labels_dt=None, labels_gt=None, scores_dt=None, points_size=0.2, view_3d=False, coordinates="kitti"):
    # data = data[np.where(data[:,1]>0)]
    # Init axes.
    _, ax = plt.subplots(1, 1, figsize=(9, 9))
    draw_point_cloud(
            ax, points, boxes_dt, labels_dt, scores_dt,
            title = 'Point Cloud & Result Compare, XY projection (Z = 0)',
            axes=[0, 1],
            gt_boxes=boxes_gt, gt_labels=labels_gt, 
            point_size = points_size,
            coordinates = coordinates,
            view_points_in_box=False
        )
    plt.savefig(save_path)
    plt.close('all')