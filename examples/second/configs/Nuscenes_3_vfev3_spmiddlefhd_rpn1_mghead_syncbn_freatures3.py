import itertools
import logging

from det3d.builder import build_box_coder
from det3d.utils.config_tool import get_downsample_factor

norm_cfg = None

tasks = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=1, class_names=["pedestrian"]),
    dict(num_class=1, class_names=["bicycle"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    type="iou",
    anchor_generators=[
        dict(
            type="anchor_generator_range",
            #sizes=[1.6, 3.9, 1.56],
            #anchor_ranges=[0, -40.0, -1.0, 70.4, 40.0, -1.0],
            sizes=[1.97, 4.63, 1.74],
            anchor_ranges=[-50.4, 0, -0.95, 50.4, 50.4, -0.95],
            rotations=[0, 1.57],
            matched_threshold=0.6,
            unmatched_threshold=0.45,
            class_name="car",
        ),
        dict(
            type="anchor_generator_range",
            #sizes=[0.6, 0.8, 1.73],
            #anchor_ranges=[0, -40.0, -0.6, 70.4, 40.0, -0.6],
            sizes=[0.67, 0.73, 1.77],
            anchor_ranges=[-50.4, 0, -0.935, 50.4, 50.4, -0.935],
            rotations=[0, 1.57],
            matched_threshold=0.6,
            unmatched_threshold=0.4,
            class_name="pedestrian",
        ),
        dict(
            type="anchor_generator_range",
#             sizes=[0.6, 1.76, 1.73],
#             anchor_ranges=[0, -40.0, -0.6, 70.4, 40.0, -0.6],
            sizes=[0.60, 1.70, 1.28],
            anchor_ranges=[-50.4,0, -1.18, 50.4, 50.4, -1.18],
            rotations=[0, 1.57],
            matched_threshold=0.5,
            unmatched_threshold=0.35,
            class_name="bicycle",
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
    pretrained="/home/elodie/det3D_Output/NUSC_SECOND_3_20200428-124425/epoch_5.pth",
    reader=dict(
        type="VoxelFeatureExtractorV3",
        num_input_features=3,
        norm_cfg=norm_cfg,
    ),
    backbone=dict(
        #type="SpMiddleFHD", num_input_features=4, ds_factor=8, norm_cfg=norm_cfg,
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
        # type='RPNHead',
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
        direction_offset=0.0,
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
        nms_post_max_size=100,
        nms_iou_threshold=0.01,
    ),
    score_threshold=0.3,
    post_center_limit_range=[-61.2, 0, -10.0, 61.2, 61.2, 10.0],
    max_per_img=100,
)

# dataset settings
n_sweeps = 1
# dataset_type = "NuScenesDataset"
# data_root = "/home/elodie/nuScenes_DATASET"
dataset_type = "KittiDataset"
data_root = "/home/elodie/KITTI_DATASET/object"

db_sampler = dict(
    type="GT-AUG",
    enable=False,
    #db_info_path = "/home/elodie/nuScenes_DATASET/kitti_dbinfos_train.pkl", #(Without velo)
    db_info_path="/home/elodie/nuScenes_DATASET/pkl/dbinfos_train_10sweeps_withoutvelo.pkl",
    #db_info_path="/data/Datasets/KITTI/Kitti/object/dbinfos_train.pkl",
    #db_info_path="/home/elodie/KITTI_DATASET/object/dbinfos_train.pkl",
    sample_groups=[dict(car=2), dict(pedestrian=6), dict(bicycle=2),],
    db_prep_steps=[
        dict(filter_by_min_num_points=dict(car=5, pedestrian=5, bicycle=5)),
        dict(filter_by_difficulty=[-1],),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
)

train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    gt_loc_noise=[0.0, 0.0, 0.0],
    gt_rot_noise=[0.0, 0.0],
    global_rot_noise=[-0.3925, 0.3925],
    global_scale_noise=[0.95, 1.05],
    global_rot_per_obj_range=[0, 0],
    global_trans_noise=[0.2, 0.2, 0.2],
    remove_points_after_sample=False,
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

# voxel_generator = dict(
#     range=[0, -40.0, -3.0, 70.4, 40.0, 1.0],
#     voxel_size=[0.05, 0.05, 0.1],
#     max_points_in_voxel=5,
#     max_voxel_num=40000,
# )
voxel_generator = dict(
    range=[-50.4, 0, -5.0, 50.4, 50.4, 3.0],
    voxel_size=[0.1, 0.1, 0.2],
    max_points_in_voxel=10,
    max_voxel_num=60000,
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignTarget", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
    # dict(type='PointCloudCollect', keys=['points', 'voxels', 'annotations', 'calib']),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignTarget", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

#train_anno = "/home/elodie/nuScenes_DATASET/infos_train.pkl"
#val_anno = "/home/elodie/nuScenes_DATASET/infos_val.pkl"
# train_anno = "/home/elodie/nuScenes_DATASET/pkl/infos_train_10sweeps_withvelo.pkl"
# val_anno = "/home/elodie/nuScenes_DATASET/pkl/infos_val_10sweeps_withvelo.pkl"
val_anno = "/home/elodie/KITTI_DATASET_NEW/object/kitti_infos_val_kitti.pkl"

train_anno = "/home/elodie/DATASET_INFO/infos_train.pkl"
#val_anno = "/home/elodie/DATASET_INFO/infos_val.pkl"

test_anno = None

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        n_sweeps=n_sweeps,
        class_names=class_names,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        ann_file=val_anno,
        n_sweeps=n_sweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        ann_file=test_anno,
        n_sweeps=n_sweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)

# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)

"""training hooks """
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy in training hooks
lr_config = dict(
    type="one_cycle", lr_max=0.001, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)

# yapf:enable
# runtime settings
total_epochs = 20
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = "/home/elodie/det3D_Output/MegDet3D_Outputs/SECOND_NUSC"
load_from = None
resume_from = None
workflow = [("train", 1), ("val", 1)]
# workflow = [('train', 1)]
