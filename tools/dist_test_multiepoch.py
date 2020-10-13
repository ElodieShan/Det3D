import argparse
import json
import os
import sys

import apex
import numpy as np
import torch
import yaml
from det3d import __version__, torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.datasets.kitti import kitti_common as kitti
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.trainer import get_dist_info, load_checkpoint
from det3d.torchie.trainer.utils import all_gather, synchronize
from torch.nn.parallel import DistributedDataParallel
import time
import logging
# @brief: 查看路劲的上级文件是否存在，若不存在则逐层创建
def init_data_dir(path):
    file_root_path, file_name=os.path.split(path)
    if not os.path.exists(file_root_path):
        try:
            os.makedirs(file_root_path)
        except:
            print("Error occurs when make dirs %s"%file_root_path)

def get_logger(log_level=logging.INFO, log_path=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", help="the dir to save logs and models")
    parser.add_argument("--dataset_type", help="dataset_type to test")
    parser.add_argument("--anno_type", help="dataset_type to test")

    parser.add_argument(
        "--epoch_start",
        type=int,
        default=1,
        help="epochs to test",
    )
    parser.add_argument(
        "--epoch_end",
        type=int,
        default=20,
        help="epochs to test",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args

def lidar_box_nusc2kitti(gt_boxes):
    gt_boxes = gt_boxes.detach().cpu().numpy()
    R = np.array([[ 0, 1 ,0],
            [ -1,0,0],
            [ 0,0,1]])
    gt_boxes_theta = np.dot(R, gt_boxes[:, :3].T).T
    gt_boxes = np.hstack((gt_boxes_theta, gt_boxes[:, 3:]))

    gt_boxes[:, 6] = gt_boxes[:, 6] + np.sign(gt_boxes[:, 6]) * np.pi / 2
    return gt_boxes

def main():

    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)

    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank
    # update cfg
    assert args.dataset_type in ["KittiDataset","NuScenesDataset"], "args.dataset_type is neither KittiDataset nor NuScenesDataset!"
    assert args.anno_type in ["val","train"], "args.anno_type is neither val nor train!"

    if args.dataset_type == "KittiDataset":
        cfg.dataset_type = cfg.kitti_dataset_type
        cfg.data_root = cfg.kitti_data_root

        if args.anno_type == "val":
            cfg.val_anno = cfg.kitti_val_anno
        elif args.anno_type == "train":
            cfg.val_anno = cfg.kitti_train_anno
        for i in range(len(cfg.test_pipeline)):
            if cfg.test_pipeline[i]["type"] == "LoadPointCloudFromFile":
                cfg.test_pipeline[i]["dataset"] = cfg.dataset_type
                break
        cfg.data.val.type = cfg.dataset_type
        cfg.data.val.root_path = cfg.data_root 
        cfg.data.val.info_path = cfg.val_anno
        cfg.data.val.ann_file = cfg.val_anno
        cfg.data.val.pipeline = cfg.test_pipeline


    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        cfg.gpus = torch.distributed.get_world_size()
    else:
        cfg.gpus = args.gpus

    # init logger before other steps
    log_path = cfg.work_dir + "/multi_epochs_val/" + cfg.dataset_type + "/" +  cfg.dataset_type + "_" + args.anno_type + "_epoch_" + str(args.epoch_start) + "_" + str(args.epoch_end) + ".log"
    init_data_dir(log_path)
    logger = get_logger(cfg.log_level, log_path=log_path)
    logger.info("Distributed testing: {}".format(distributed))
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    logger.info("Dataset Type: {}".format(cfg.dataset_type))
    logger.info("Dataset Root: {}".format(cfg.data_root))
    logger.info("Val Dataset Anno Path: {}".format(cfg.val_anno))

    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        batch_size=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    for epoch in range(args.epoch_start,args.epoch_end+1): 
        logger.info("\n\n----------- Epoch {} ----------\n".format(str(epoch)))
        checkpoint_path = args.work_dir + "/epoch_" + str(epoch) + ".pth"
        output_dir = cfg.work_dir + "/multi_epochs_val/" + cfg.dataset_type + "/epoch_" + str(epoch)
        init_data_dir(output_dir+"/")
        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

        checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")

        # put model on gpus
        if distributed:
            model = apex.parallel.convert_syncbn_model(model)
            model = DistributedDataParallel(
                model.cuda(cfg.local_rank),
                device_ids=[cfg.local_rank],
                output_device=cfg.local_rank,
                # broadcast_buffers=False,
                find_unused_parameters=True,
            )
        else:
            model = model.cuda()

        model.eval()
        mode = "val"

        logger.info(f"work dir: {args.work_dir}")

        if cfg.local_rank == 0:
            prog_bar = torchie.ProgressBar(len(data_loader.dataset) // cfg.gpus)

        detections = {}
        cpu_device = torch.device("cpu")

        for i, data_batch in enumerate(data_loader):
            with torch.no_grad():
                outputs = batch_processor(
                    model, data_batch, train_mode=False, local_rank=args.local_rank,
                )
            for output in outputs:
                token = output["metadata"]["token"]
                for k, v in output.items():
                    if k not in [
                        "metadata",
                    ]:
                        output[k] = v.to(cpu_device)
                detections.update(
                    {token: output,}
                )
                
                if args.local_rank == 0:
                    prog_bar.update()

        synchronize() 

        all_predictions = all_gather(detections)
        if args.local_rank != 0:
            return

        predictions = {}
        for p in all_predictions:
            if cfg.dataset_type == "KittiDataset": # change for new kitti dataset elodie
                for k,v in p.items():
                    new_box3d_lidar = lidar_box_nusc2kitti(v['box3d_lidar'])
                    p[k]['box3d_lidar']=new_box3d_lidar
            if cfg.dataset_type == "NuScenesDataset":
                if cfg.box_coder["n_dim"] == 7:
                    for k,v in p.items():
                        p[k]['box3d_lidar'] = np.insert(p[k]['box3d_lidar'],6,np.zeros(p[k]['box3d_lidar'].shape[0]),1)
                        p[k]['box3d_lidar'] = np.insert(p[k]['box3d_lidar'],6,np.zeros(p[k]['box3d_lidar'].shape[0]),1)
                        # p[k]['box3d_lidar']=new_box3d_lidar
            predictions.update(p)
        if cfg.dataset_type == "NuScenesDataset":
            result_dict, dt_annos = dataset.evaluation(predictions, output_dir=output_dir, use_velo=False, only_front=True)
        else:
            result_dict, dt_annos = dataset.evaluation(predictions, output_dir=output_dir)
        for k, v in result_dict["results"].items():
            logger.info(f"Evaluation {k}: {v}")
        time.sleep(5)

if __name__ == "__main__":
    main()
