import itertools
import logging

from det3d.builder import build_box_coder
from det3d.utils.config_tool import get_downsample_factor

## 1012
## 1. DataSet Sample to 16 lines
norm_cfg = None

# dataset settings
nsweeps = 1
test_anno = None
# db_info_path = "/home/elodie/nuScenes_DATASET/pkl_mix/dbinfos_train_v2.pkl"
# train_anno = "/home/elodie/nuScenes_DATASET/pkl_mix/infos_train_v3.pkl"

# dataset_type = "NuScenesDataset"
# data_root = "/home/dataset/nuScenes_DATASET"
# val_anno = "/home/dataset/nuScenes_DATASET/pkl/infos_val_10sweeps_withvelo.pkl"

dataset_type = "KittiDataset"
data_root = "/home/dataset/KITTI_DATASET_NEW/object"
val_anno = "/home/dataset/KITTI_DATASET_NEW/object/pkl/kitti_infos_val_formatnusc_feature5.pkl"
train_anno = "/home/dataset/KITTI_DATASET_NEW/object/pkl/kitti_infos_train_formatnusc_feature5.pkl"
db_info_path = "/home/dataset/KITTI_DATASET_NEW/object/pkl/dbinfos_train_axisnusc_feature5.pkl"

tasks = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=1, class_names=["truck"]),
    dict(num_class=1, class_names=["bicycle"]),
    dict(num_class=1, class_names=["pedestrian"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    type="iou",
    anchor_generators=[
        dict(
            type="anchor_generator_range",
            sizes=[1.96, 4.62, 1.73],
            anchor_ranges=[-51.2, 0, -0.95, 51.2, 51.2, -0.95],
            rotations=[0, 1.57],
            matched_threshold=0.6,
            unmatched_threshold=0.45,
            class_name="car",
        ),

        dict(
            type="anchor_generator_range",
            # sizes=[2.51, 6.93, 2.84], #Ori
            sizes=[2.10,5.80,2.02],
            anchor_ranges=[-51.2, 0, -0.40, 51.2, 51.2, -0.40],
            rotations=[0, 1.57],
            matched_threshold=0.55,
            unmatched_threshold=0.4,
            class_name="truck",
        ),
        dict(
            type="anchor_generator_range",
            #sizes=[0.60, 1.70, 1.28], #Ori
            sizes=[0.70,1.59,1.11],
            anchor_ranges=[-51.2, 0, -1.18, 51.2, 51.2, -1.18],
            rotations=[0, 1.57],
            matched_threshold=0.5,
            unmatched_threshold=0.35,
            class_name="bicycle",
        ),
        dict(
            type="anchor_generator_range",
            sizes=[0.61,0.63,1.77],
            anchor_ranges=[-51.2, 0, -0.935, 51.2, 51.2, -0.935],
            rotations=[0, 1.57],
            matched_threshold=0.6,
            unmatched_threshold=0.4,
            class_name="pedestrian",
        ),
    ],
    sample_positive_fraction=-1,
    sample_size=512,
    region_similarity_calculator=dict(type="nearest_iou_similarity",),
    pos_area_threshold=-1,
    tasks=tasks,
)

box_coder = dict(
    type="ground_box3d_coder", n_dim=7, linear_dim=False, encode_angle_vector=False,
)

# model settings
model = dict(
    type="VoxelNet",
    pretrained=None,
    reader=dict(
        type="VoxelFeatureExtractorV3",
        num_input_features=3,
        norm_cfg=norm_cfg,
    ),
    backbone=dict(
        type="SpMiddleFHD", num_input_features=3, ds_factor=8, norm_cfg=norm_cfg,
    ),
    neck=dict(
        type="RPN",
        layer_nums=[5,],
        ds_layer_strides=[1,],
        ds_num_filters=[128,],
        us_layer_strides=[1,],
        us_num_filters=[128,],
        num_input_features=128,
        norm_cfg=norm_cfg,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="MultiGroupHead",
        mode="3d",
        in_channels=sum([128,]),
        norm_cfg=norm_cfg,
        tasks=tasks,
        weights=[1,],
        box_coder=build_box_coder(box_coder),
        encode_background_as_zeros=True,
        loss_norm=dict(
            type="NormByNumPositives", pos_cls_weight=1.0, neg_cls_weight=1.0,
        ),
        loss_cls=dict(type="SigmoidFocalLoss", alpha=0.25, gamma=2.0, loss_weight=1.0,),
        use_sigmoid_score=True,
        loss_bbox=dict(
            type="WeightedSmoothL1Loss",
            sigma=3.0,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            codewise=True,
            loss_weight=2.0,
        ),
        encode_rad_error_by_sin=True,
        loss_aux=dict(
            type="WeightedSoftmaxClassificationLoss",
            name="direction_classifier",
            loss_weight=0.2,
        ),
        direction_offset=0.785,
    ),
)

assigner = dict(
    box_coder=box_coder,
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    debug=False,
)

train_cfg = dict(assigner=assigner)


test_cfg = dict(
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=83,
        nms_iou_threshold=0.2,
    ),
    score_threshold=0.1,
    post_center_limit_range=[-61.2, 0, -10.0, 61.2, 61.2, 10.0],
    max_per_img=200,
)


db_sampler = dict(
    type="GT-AUG",
    enable=False,
    db_info_path=db_info_path,
    sample_groups=[
        dict(car=2),
        dict(truck=2),
        dict(bicycle=3),
        dict(pedestrian=3),
    ],
    db_prep_steps=[
        dict(
            filter_by_min_num_points=dict(
                car=5,
                truck=5,
                bicycle=5,
                pedestrian=5,
            )
        ),
        dict(filter_by_difficulty=[-1],),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
)

train_preprocessor = dict(
    mode="train",
    noise_per_object=False, #notice elodie
    shuffle_points=False, #notice elodie - voxel size
    min_points_in_gt=3, # notice elodie
    gt_loc_noise=[0.0, 0.0, 0.0],
    gt_rot_noise=[0.0, 0.0],
    global_rot_noise=[-0.3925, 0.3925],
    global_rotation_noise_kitti = [-0.1745,0.1745],
    global_scale_noise=[0.95, 1.05],
    global_rot_per_obj_range=[0, 0],
    global_trans_noise=[0.2, 0.2, 0.2],
    remove_points_after_sample=True,
    gt_drop_percentage=0.0,
    gt_drop_max_keep_points=15,
    remove_unknown_examples=False,
    remove_environment=False,
    db_sampler=db_sampler,
    class_names=class_names,
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
    remove_environment=False,
    remove_unknown_examples=False,
)

voxel_generator = dict(
    range=[-51.2, 0, -3.0, 51.2, 51.2, 1.0],
    voxel_size=[0.1, 0.1, 0.1],
    max_points_in_voxel=10,
    max_voxel_num=60000,
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor, pc_range=voxel_generator["range"]),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignTarget", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor, pc_range=voxel_generator["range"]),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignTarget", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        ann_file=val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        ann_file=test_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)

# optimizer
optimizer = dict(
    TYPE="adam",
    VALUE=dict(amsgrad=0.0, wd=0.01),
    FIXED_WD=True,
    MOVING_AVERAGE=False,
)
"""training hooks """
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy in training hooks
lr_config = dict(
    type="one_cycle", lr_max=0.003, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)

# yapf:enable
# runtime settings
total_epochs = 100
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = "/home/elodie/det3D_Output/SECOND_NUSC"
load_from = None
resume_from = None
workflow = [("train", 5), ("val", 1)]
