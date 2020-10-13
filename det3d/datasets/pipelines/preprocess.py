import numpy as np

from det3d import torchie
from det3d.datasets.kitti import kitti_common as kitti
from det3d.core.evaluation.bbox_overlaps import bbox_overlaps
from det3d.core.bbox import box_np_ops
from det3d.core.sampler import preprocess as prep
from det3d.builder import (
    build_dbsampler,
    build_anchor_generator,
    build_similarity_metric,
    build_box_coder,
)
from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.core.anchor.target_assigner import TargetAssigner

from ..registry import PIPELINES

from det3d.datasets.utils.pointcloud_sample_utils import *

import time #elodie time
def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]


@PIPELINES.register_module
class Preprocess(object):
    def __init__(self, cfg=None, view_sampled_points=False, pc_range=None, **kwargs):
        self.view_sampled_points = view_sampled_points #elodie - change for sampled points visualization
        self.remove_environment = cfg.remove_environment
        self.shuffle_points = cfg.shuffle_points
        self.remove_unknown = cfg.remove_unknown_examples
        self.pc_range = np.array(pc_range, dtype=np.float32)

        self.downsample_to_16lines = cfg.get("downsample_to_16lines", True)
        self.noise_per_object = cfg.get("noise_per_object", True)
        self.min_points_in_gt = cfg.get("min_points_in_gt", -1)
        self.add_rgb_to_points = cfg.get("add_rgb_to_points", False)
        self.reference_detections = cfg.get("reference_detections", None)
        self.remove_outside_points = cfg.get("remove_outside_points", False)
        self.random_crop = cfg.get("random_crop", False)
        self.global_rotation_noise_kitti = cfg.get("global_rotation_noise_kitti", [0,0])

        self.mode = cfg.mode
        if self.mode == "train":
            self.gt_rotation_noise = cfg.gt_rot_noise
            self.gt_loc_noise_std = cfg.gt_loc_noise
            self.global_rotation_noise = cfg.global_rot_noise
            self.global_scaling_noise = cfg.global_scale_noise
            self.global_random_rot_range = cfg.global_rot_per_obj_range
            self.global_translate_noise_std = cfg.global_trans_noise
            self.gt_points_drop = (cfg.gt_drop_percentage,)
            self.remove_points_after_sample = cfg.remove_points_after_sample
            self.class_names = cfg.class_names
            self.db_sampler = build_dbsampler(cfg.db_sampler)
            self.npoints = cfg.get("npoints", -1)
            self.random_select = cfg.get("random_select", False)

        self.symmetry_intensity = cfg.get("symmetry_intensity", False)

    def __call__(self, res, info):
        # t_start = time.time() #elodie time
        # print("t_start:",t_start) 

        res["mode"] = self.mode

        if res["type"] in ["KittiDataset", "LyftDataset"]:
            points = res["lidar"]["points"]
        elif res["type"] == "NuScenesDataset":
            if res["lidar"]["nsweeps"]>1:
                points = res["lidar"]["combined"]
            else:
                points = res["lidar"]["points"]
        else: # elodie 
            points = res["lidar"]["points"]

        if self.mode == "train":
            anno_dict = res["lidar"]["annotations"]

            gt_dict = {
                "gt_boxes": anno_dict["boxes"],
                "gt_names": np.array(anno_dict["names"]).reshape(-1),
            }
            if "num_points_in_gt" in anno_dict: #elodie
                gt_dict["num_points_in_gt"] = anno_dict["num_points_in_gt"].reshape(-1)

            if "difficulty" not in anno_dict:
                difficulty = np.zeros([anno_dict["boxes"].shape[0]], dtype=np.int32)
                gt_dict["difficulty"] = difficulty
            else:
                gt_dict["difficulty"] = anno_dict["difficulty"]

        if "calib" in res:
            calib = res["calib"]
        else:
            calib = None

        if self.add_rgb_to_points: #False
            assert calib is not None and "image" in res
            image_path = res["image"]["image_path"]
            image = (
                imgio.imread(str(pathlib.Path(root_path) / image_path)).astype(
                    np.float32
                )
                / 255
            )
            points_rgb = box_np_ops.add_rgb_to_points(
                points, image, calib["rect"], calib["Trv2c"], calib["P2"]
            )
            points = np.concatenate([points, points_rgb], axis=1)
            num_point_features += 3

        if self.reference_detections is not None: #False
            assert calib is not None and "image" in res
            C, R, T = box_np_ops.projection_matrix_to_CRT_kitti(P2)
            frustums = box_np_ops.get_frustum_v2(reference_detections, C)
            frustums -= T
            frustums = np.einsum("ij, akj->aki", np.linalg.inv(R), frustums)
            frustums = box_np_ops.camera_to_lidar(frustums, rect, Trv2c)
            surfaces = box_np_ops.corner_to_surfaces_3d_jit(frustums)
            masks = points_in_convex_polygon_3d_jit(points, surfaces)
            points = points[masks.any(-1)]

        if self.remove_outside_points: #False 
            assert calib is not None
            image_shape = res["image"]["image_shape"]
            points = box_np_ops.remove_outside_points(
                points, calib["rect"], calib["Trv2c"], calib["P2"], image_shape
            )
        if self.remove_environment is True and self.mode == "train": #False
            selected = kitti.keep_arrays_by_name(gt_names, target_assigner.classes)
            _dict_select(gt_dict, selected)
            masks = box_np_ops.points_in_rbbox(points, gt_dict["gt_boxes"])
            points = points[masks.any(-1)]

        if self.mode == "train":
            selected = kitti.drop_arrays_by_name(
                gt_dict["gt_names"], ["DontCare", "dontcare", "ignore"]
            ) #elodie change - add dontcare 20200803

            _dict_select(gt_dict, selected)
            if self.remove_unknown:  #False 
                remove_mask = gt_dict["difficulty"] == -1
                """
                gt_boxes_remove = gt_boxes[remove_mask]
                gt_boxes_remove[:, 3:6] += 0.25
                points = prep.remove_points_in_boxes(points, gt_boxes_remove)
                """
                keep_mask = np.logical_not(remove_mask)
                _dict_select(gt_dict, keep_mask)
            gt_dict.pop("difficulty")
            
            # t_min_points_in_gt_start = time.time() #elodie time 
            # print("start to min_points_in_gt:",t_min_points_in_gt_start - t_start)

            # ------------------------------MIN POINTS BOX FILTER1
            if self.min_points_in_gt > 0:
                if "num_points_in_gt" in gt_dict: #elodie
                    point_counts = gt_dict["num_points_in_gt"]
                    gt_dict.pop("num_points_in_gt")
                else:
                    # points_count_rbbox takes 10ms with 10 sweeps nuscenes data
                    point_counts = box_np_ops.points_count_rbbox(
                        points, gt_dict["gt_boxes"]
                    )
                mask = point_counts >= self.min_points_in_gt
                _dict_select(gt_dict, mask)
            else:
                gt_dict.pop("num_points_in_gt")

            # t_flip_start = time.time() #elodie time 
            # print("min_points_in_gt duration time:",t_flip_start - t_min_points_in_gt_start)

            # ------------------------------X/Y AXIS FLIP
            if "preprocess_type" in info: # elodie -change for data aug of data 20200905
                if "x_flip" in info["preprocess_type"]:
                    if "y_flip" in info["preprocess_type"]:
                        gt_dict["gt_boxes"], points = prep.flip_XY(gt_dict["gt_boxes"], points,\
                            x_filp_switch=info["preprocess_type"]["x_flip"], y_filp_switch=info["preprocess_type"]["y_flip"])
                    else:
                        gt_dict["gt_boxes"], points = prep.flip_XY(gt_dict["gt_boxes"], points,\
                            x_filp_switch=info["preprocess_type"]["x_flip"], random_y_filp=True)   
            else:
                if res["type"] in ["NuScenesDataset"]:
                    # double flip gives 3 map improvement for pointppillars on nuScenes 
                    gt_dict["gt_boxes"], points = prep.random_flip_both(gt_dict["gt_boxes"], points)
                else:
                    gt_dict["gt_boxes"], points = prep.random_flip(gt_dict["gt_boxes"], points)

            # t_gtsample_start = time.time() #elodie time 
            # print("flip duration time:",t_gtsample_start - t_flip_start)

            # ------------------------------GT SMPLER 
            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )

            if self.db_sampler:
                sampled_dict = self.db_sampler.sample_all(
                    res["metadata"]["image_prefix"],
                    gt_dict["gt_boxes"],
                    gt_dict["gt_names"],
                    res["metadata"]["num_point_features"],
                    self.random_crop,
                    gt_group_ids=None,
                    calib=calib,
                )

                if sampled_dict is not None:
                    sampled_gt_names = sampled_dict["gt_names"]
                    sampled_gt_boxes = sampled_dict["gt_boxes"]
                    sampled_points = sampled_dict["points"]
                    if self.view_sampled_points: #elodie - change for sampled points visualization
                        sampled_points[:,3] = 500
                    sampled_gt_masks = sampled_dict["gt_masks"]
                    gt_dict["gt_names"] = np.concatenate(
                        [gt_dict["gt_names"], sampled_gt_names], axis=0
                    )
                    gt_dict["gt_boxes"] = np.concatenate(
                        [gt_dict["gt_boxes"], sampled_gt_boxes]
                    )
                    gt_boxes_mask = np.concatenate(
                        [gt_boxes_mask, sampled_gt_masks], axis=0
                    )

                    if self.remove_points_after_sample: #滤除与sampled标注框位置想重的点
                        masks = box_np_ops.points_in_rbbox(points, sampled_gt_boxes)
                        points = points[np.logical_not(masks.any(-1))]

                    points = np.concatenate([sampled_points, points], axis=0)

            # t_noiseobject_start = time.time() #elodie time 
            # print("gt sample duration time:",t_noiseobject_start - t_gtsample_start)
            # ------------------------------NOISE PER OBJECT 
            if self.noise_per_object:
                prep.noise_per_object_v3_(
                    gt_dict["gt_boxes"],
                    points,
                    gt_boxes_mask,
                    rotation_perturb=self.gt_rotation_noise,
                    center_noise_std=self.gt_loc_noise_std,
                    global_random_rot_range=self.global_random_rot_range,
                    group_ids=None,
                    num_try=100,
                )

            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes

            # t_rotscal_start = time.time() #elodie time 
            # print("noiseobject duration time:",t_rotscal_start - t_noiseobject_start)

            # ------------------------------GLOBAL ROTATION & SCALING
            if "data_type" in info and info["data_type"] == "kitti": #elodie
                gt_dict["gt_boxes"], points = prep.global_rotation(
                    gt_dict["gt_boxes"], points, rotation=self.global_rotation_noise_kitti
                )
            else:
                gt_dict["gt_boxes"], points = prep.global_rotation(
                    gt_dict["gt_boxes"], points, rotation=self.global_rotation_noise
                )

            gt_dict["gt_boxes"], points = prep.global_scaling_v2(
                gt_dict["gt_boxes"], points, *self.global_scaling_noise
            )

            # t_frontfilter_start = time.time() #elodie time 
            # print("rot&scal duration time:",t_frontfilter_start - t_rotscal_start)
            if self.pc_range[1] == 0:
                points = points[points[:,1]>0]

            # t_downsample_start = time.time() #elodie time 
            # print("front range filter duration time:",t_downsample_start - t_frontfilter_start)

        # ------------------------------DOWNSAMPLE/UPSAMPLE POINTCLOUD
        if self.downsample_to_16lines and "preprocess_type" in info and info["preprocess_type"]["sample"]: # elodie -change for data aug of data 20200905
            # print("shape before:",points.shape[0])
            if info["data_type"] == "kitti":
                points = downsample_kitti(points, verticle_switch=True, horizontal_switch=True)
            if info["data_type"] == "nuscenes":
                points = downsample_nusc_v2(points)
                points = upsample_nusc_v1(points)
            # print("shape after:",points.shape[0])

        # t_filter_outrange_start = time.time() #elodie time 
        # print("downsample duration time:",t_filter_outrange_start - t_downsample_start)

        # ------------------------------FILTER POINT & BOX OUTSIDE RANGE
        points = prep.filter_points_outside_range(points, self.pc_range)

        if self.mode == "train" :
            assert self.pc_range is not None, 'Preprocess: pc_range is None. \nRemove gtbox outside training range.'    

            bv_range = self.pc_range[[0, 1, 3, 4]]

            if bv_range[1] == 0: # elodie - only use front points
                mask = prep.filter_gt_box_on_edge(gt_dict["gt_boxes"], bv_range)
            else:
                mask = prep.filter_gt_box_outside_range(gt_dict["gt_boxes"], bv_range)
            _dict_select(gt_dict, mask)

            # ------------------------------MIN POINTS BOX FILTER2
            if bv_range[1] == 0:
                point_counts = box_np_ops.points_count_rbbox(
                        points, gt_dict["gt_boxes"]
                    )

                mask = point_counts >= self.min_points_in_gt
                _dict_select(gt_dict, mask)
        # t_shuffle_start = time.time() #elodie time 
        # print("filter_outrange duration time:",t_shuffle_start - t_filter_outrange_start)

        # ------------------------------SHUFFLE POINTS
        if self.shuffle_points: # shuffle is a little slow.
            np.random.shuffle(points)
        # t_random_select_start = time.time() #elodie time 
        # print("shuffle duration time:",t_random_select_start - t_shuffle_start)

        # ------------------------------RANDOM SELECT
        if self.mode == "train" and self.random_select:
            if self.npoints < points.shape[0]:
                pts_depth = points[:, 2]
                pts_near_flag = pts_depth < 40.0
                far_idxs_choice = np.where(pts_near_flag == 0)[0]
                near_idxs = np.where(pts_near_flag == 1)[0]
                near_idxs_choice = np.random.choice(
                    near_idxs, self.npoints - len(far_idxs_choice), replace=False
                )

                choice = (
                    np.concatenate((near_idxs_choice, far_idxs_choice), axis=0)
                    if len(far_idxs_choice) > 0
                    else near_idxs_choice
                )
                np.random.shuffle(choice)
            else:
                choice = np.arange(0, len(points), dtype=np.int32)
                if self.npoints > len(points):
                    extra_choice = np.random.choice(
                        choice, self.npoints - len(points), replace=False
                    )
                    choice = np.concatenate((choice, extra_choice), axis=0)
                np.random.shuffle(choice)

            points = points[choice]

        if self.symmetry_intensity:
            points[:, -1] -= 0.5  # translate intensity to [-0.5, 0.5]

        # t_end = time.time() #elodie time 
        # print("random_select duration time:",t_end - t_random_select_start)
        # print("data preprocess time:",t_end - t_start,"\n===\n")
  

        res["lidar"]["points"] = points

        if self.mode == "train":
            res["lidar"]["annotations"] = gt_dict

        return res, info


@PIPELINES.register_module
class Voxelization(object):
    def __init__(self, **kwargs):
        cfg = kwargs.get("cfg", None)
        self.range = cfg.range
        self.voxel_size = cfg.voxel_size
        self.max_points_in_voxel = cfg.max_points_in_voxel
        self.max_voxel_num = cfg.max_voxel_num

        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num,
        )

    def __call__(self, res, info):
        voxel_size = self.voxel_generator.voxel_size
        pc_range = self.voxel_generator.point_cloud_range
        grid_size = self.voxel_generator.grid_size

        if res["mode"] == "train":
            gt_dict = res["lidar"]["annotations"]
            #  Remove gtbox outside training range. Part is remove to Preprocess.
            # bv_range = pc_range[[0, 1, 3, 4]]
            # if bv_range[1] == 0: # elodie - only use front points
            #     mask = prep.filter_gt_box_on_edge(gt_dict["gt_boxes"], bv_range)
            # else:
            #     mask = prep.filter_gt_box_outside_range(gt_dict["gt_boxes"], bv_range)
            # _dict_select(gt_dict, mask)

            # res["lidar"]["annotations"] = gt_dict


        voxels, coordinates, num_points = self.voxel_generator.generate(
            res["lidar"]["points"]
        )

        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
        
        res["lidar"]["voxels"] = dict(
            voxels=voxels,
            coordinates=coordinates,
            num_points=num_points,
            num_voxels=num_voxels,
            shape=grid_size,
        )

        return res, info


@PIPELINES.register_module
class AssignTarget(object):
    def __init__(self, **kwargs):
        assigner_cfg = kwargs["cfg"]
        target_assigner_config = assigner_cfg.target_assigner
        tasks = target_assigner_config.tasks
        box_coder_cfg = assigner_cfg.box_coder

        anchor_cfg = target_assigner_config.anchor_generators
        anchor_generators = []
        for a_cfg in anchor_cfg:
            anchor_generator = build_anchor_generator(a_cfg)
            anchor_generators.append(anchor_generator)
        similarity_calc = build_similarity_metric(
            target_assigner_config.region_similarity_calculator
        )
        positive_fraction = target_assigner_config.sample_positive_fraction
        if positive_fraction < 0:
            positive_fraction = None
        target_assigners = []
        flag = 0

        box_coder = build_box_coder(box_coder_cfg)

        for task in tasks:
            target_assigner = TargetAssigner(
                box_coder=box_coder,
                anchor_generators=anchor_generators[flag : flag + task.num_class],
                region_similarity_calculator=similarity_calc,
                positive_fraction=positive_fraction,
                sample_size=target_assigner_config.sample_size,
            )
            flag += task.num_class
            target_assigners.append(target_assigner)

        self.target_assigners = target_assigners
        self.out_size_factor = assigner_cfg.out_size_factor
        self.anchor_area_threshold = target_assigner_config.pos_area_threshold

    def __call__(self, res, info):

        class_names_by_task = [t.classes for t in self.target_assigners]

        # Calculate output featuremap size
        grid_size = res["lidar"]["voxels"]["shape"]
        feature_map_size = grid_size[:2] // self.out_size_factor
        feature_map_size = [*feature_map_size, 1][::-1]

        anchors_by_task = [
            t.generate_anchors(feature_map_size) for t in self.target_assigners
        ]
        anchor_dicts_by_task = [
            t.generate_anchors_dict(feature_map_size) for t in self.target_assigners
        ]
        reshaped_anchors_by_task = [
            t["anchors"].reshape([-1, t["anchors"].shape[-1]]) for t in anchors_by_task
        ]
        
#         print("anchors_by_task:",np.array(anchors_by_task).shape)
#         print("reshaped_anchors_by_task:",np.array(reshaped_anchors_by_task).shape)
        
        matched_by_task = [t["matched_thresholds"] for t in anchors_by_task]
        unmatched_by_task = [t["unmatched_thresholds"] for t in anchors_by_task]

        bv_anchors_by_task = [
            box_np_ops.rbbox2d_to_near_bbox(anchors[:, [0, 1, 3, 4, -1]])
            for anchors in reshaped_anchors_by_task
        ]

        anchor_caches_by_task = dict(
            anchors=reshaped_anchors_by_task,
            anchors_bv=bv_anchors_by_task,
            matched_thresholds=matched_by_task,
            unmatched_thresholds=unmatched_by_task,
            anchors_dict=anchor_dicts_by_task,
        )

        if res["mode"] == "train":
            gt_dict = res["lidar"]["annotations"]

            task_masks = []
            flag = 0
            for class_name in class_names_by_task:
                task_masks.append(
                    [
                        np.where(
                            gt_dict["gt_classes"] == class_name.index(i) + 1 + flag
                        )
                        for i in class_name
                    ]
                )
                flag += len(class_name)

            task_boxes = []
            task_classes = []
            task_names = []
            flag2 = 0
            for idx, mask in enumerate(task_masks):
                task_box = []
                task_class = []
                task_name = []
                for m in mask:
                    task_box.append(gt_dict["gt_boxes"][m])
                    task_class.append(gt_dict["gt_classes"][m] - flag2)
                    task_name.append(gt_dict["gt_names"][m])
                task_boxes.append(np.concatenate(task_box, axis=0))
                task_classes.append(np.concatenate(task_class))
                task_names.append(np.concatenate(task_name))
                flag2 += len(mask)

            for task_box in task_boxes:
                # limit rad to [-pi, pi]
                task_box[:, -1] = box_np_ops.limit_period(
                    task_box[:, -1], offset=0.5, period=np.pi * 2
                )

            # print(gt_dict.keys())
            gt_dict["gt_classes"] = task_classes
            gt_dict["gt_names"] = task_names
            gt_dict["gt_boxes"] = task_boxes

            res["lidar"]["annotations"] = gt_dict

        anchorss = anchor_caches_by_task["anchors"]
        anchors_bvs = anchor_caches_by_task["anchors_bv"]
        anchors_dicts = anchor_caches_by_task["anchors_dict"]

        example = {}
        example["anchors"] = anchorss

        if self.anchor_area_threshold >= 0:
            example["anchors_mask"] = []
            for idx, anchors_bv in enumerate(anchors_bvs):
                anchors_mask = None
                # slow with high resolution. recommend disable this forever.
                coors = coordinates
                dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask(
                    coors, tuple(grid_size[::-1][1:])
                )
                dense_voxel_map = dense_voxel_map.cumsum(0)
                dense_voxel_map = dense_voxel_map.cumsum(1)
                anchors_area = box_np_ops.fused_get_anchors_area(
                    dense_voxel_map, anchors_bv, voxel_size, pc_range, grid_size
                )
                anchors_mask = anchors_area > anchor_area_threshold
                example["anchors_mask"].append(anchors_mask)

        if res["mode"] == "train":
            targets_dicts = []
            for idx, target_assigner in enumerate(self.target_assigners):
                if "anchors_mask" in example:
                    anchors_mask = example["anchors_mask"][idx]
                else:
                    anchors_mask = None
                targets_dict = target_assigner.assign_v2(
                    anchors_dicts[idx],
                    gt_dict["gt_boxes"][idx],
                    anchors_mask,
                    gt_classes=gt_dict["gt_classes"][idx],
                    gt_names=gt_dict["gt_names"][idx],
                )
                targets_dicts.append(targets_dict)

            example.update(
                {
                    "labels": [
                        targets_dict["labels"] for targets_dict in targets_dicts
                    ],
                    "reg_targets": [
                        targets_dict["bbox_targets"] for targets_dict in targets_dicts
                    ],
                    "reg_weights": [
                        targets_dict["bbox_outside_weights"]
                        for targets_dict in targets_dicts
                    ],
                }
            )

        res["lidar"]["targets"] = example

        return res, info
