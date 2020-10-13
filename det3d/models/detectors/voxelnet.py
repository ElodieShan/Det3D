from ..registry import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module
class VoxelNet(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(VoxelNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )

    def extract_feat(self, data):
#         print("Voxelnet: \ndata[features]:",data["features"],"\ndata[num_voxels]:", data["num_voxels"])
        input_features = self.reader(data["features"], data["num_voxels"])
        # print("data[coors]:",data["coors"],"\nbatch_size:", data["batch_size"],"\ninput_shape:", data["input_shape"])

        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        # print("xshape:",x.shape)
        if self.with_neck: #如果有neck特征处理的话，将提取处的特征，进行对应的特征处理
            x = self.neck(x)
        return x

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        preds = self.bbox_head(x)

        if return_loss:
            return self.bbox_head.loss(example, preds)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)
