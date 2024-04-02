import torch
import torch.nn as nn
from torch.nn import functional as F
from common.nets.mano_head import mano_regHead
from common.nets.Backbone import resnet_backbone
from common.nets.PMR_module import Coarse2Decent, Decent2Fine
from common.nets.SGD_module import SGD


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class Model(nn.Module):
    def __init__(self, backbone, c2d, d2f, sgd):
        super(Model, self).__init__()
        self.backbone = backbone
        self.coarse2decent = c2d
        self.decent2fine = d2f
        self.sgd = sgd

    def forward(self, inputs):
        feats = self.backbone(inputs['img'])
        feat, regression_f, mano_results, out_hm, struct_f, preds_joints_img = self.coarse2decent(feats)
        seg_info = self.sgd(feats, struct_f)
        pred_mano_results, gt_mano_results = self.decent2fine(struct_f, out_hm, seg_info, regression_f, None)
        out = {}
        out['joints_coord_cam'] = pred_mano_results['joints3d']
        out['mesh_coord_cam'] = pred_mano_results['verts3d']
        return out


def get_model():
    mano_reg = mano_regHead()
    backbone = resnet_backbone(pretrained=True)
    coarse2decent = Coarse2Decent(mano_reg=mano_reg)
    decent2fine = Decent2Fine(mano_reg=mano_reg)
    sgd_module = SGD()
    model = Model(backbone, coarse2decent, decent2fine, sgd_module)
    return model


if __name__ == '__main__':
    model = get_model()
    y = model({'img': torch.ones([1, 3, 256, 256])})
