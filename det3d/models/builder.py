from det3d.utils import build_from_cfg
from torch import nn

from .registry import (
    BACKBONES,
    DETECTORS,
    HEADS,
    LOSSES,
    NECKS,
    READERS,
    ROI_EXTRACTORS,
    SHARED_HEADS,
)


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        #build_from_cfg()返回值是一个带形参的类，返回时也就完成了实例化的过程。
        #所以modules就是一个class类的列表
        return nn.Sequential(*modules)
    #nn.Sequential 一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行
    #同时以神经网络模块为元素的有序字典也可以作为传入参数
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_reader(cfg):
    return build(cfg, READERS)


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_neck(cfg):
    return build(cfg, NECKS)


def build_roi_extractor(cfg):
    return build(cfg, ROI_EXTRACTORS)


def build_shared_head(cfg):
    return build(cfg, SHARED_HEADS)


def build_head(cfg):
    return build(cfg, HEADS)


def build_loss(cfg):
    return build(cfg, LOSSES)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    return build(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
