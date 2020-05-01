import torch.nn as nn

from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector


@DETECTORS.register_module
class SingleStageDetector(BaseDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck=None,
        bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(SingleStageDetector, self).__init__()
        #创建模型，对各个组件（比如backbone、neck、bbox_head等字典数据，构建成module类）分别创建module类模型
        self.reader = builder.build_reader(reader)
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, data):
        input_features = self.reader(data)
        x = self.backbone(input_features)# 经过backbone的前向计算，提取特征
        if self.with_neck:#如果有neck特征处理的话，将提取处的特征，进行对应的特征处理
            x = self.neck(x)
        return x

    def forward_dummy(self, example):
        x = self.extract_feat(example) 
        outs = self.bbox_head(x) #在bbox_head中进行特征处理，输出特征
        return outs

    """
    def simple_test(self, example, example_meta, rescale=False):
        x = self.extract_feat(example)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (example_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]
    """

    def aug_test(self, example, rescale=False):
        raise NotImplementedError

    def forward(self, example, return_loss=True, **kwargs):
        pass

    def predict(self, example, preds_dicts):
        pass
