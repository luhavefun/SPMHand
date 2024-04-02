import torch.nn as nn
import torch.nn.functional as F
from common.nets.net_utils import regression_block, seg_emb
from common.nets.hand_head import hand_Encoder, hand_regHead





class Coarse2Decent(nn.Module):
    def __init__(self, mano_reg=None):
        super(Coarse2Decent, self).__init__()
        self.in_planes = 64

        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.pool = nn.AvgPool2d(2, stride=2)

        self.coarse_regression_block = regression_block(1)
        self.medium_regression_block = regression_block(2)
        self.decent_regression_block = regression_block(3)

        self.resize1 = nn.Conv2d(256 + 128, 256, kernel_size=1, stride=1, padding=0)
        self.resize2 = nn.Conv2d(256 + 128, 256, kernel_size=1, stride=1, padding=0)
        self.resize3 = nn.Conv2d(256 + 128, 256, kernel_size=1, stride=1, padding=0)

        self.mano_branch = mano_reg
        self.hand_regHead = hand_regHead()

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        c5, c4, c3, c2 = x

        p5 = self.toplayer(c5)

        # coarse
        P5_m, P5_list = self.coarse_regression_block(p5, None)
        pred_mano_results1, _ = self.mano_branch(P5_m)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p4 = self.smooth1(p4)

        P4_m, P4_list = self.medium_regression_block(p4, P5_list)
        pred_mano_results2, _ = self.mano_branch(P4_m)
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p3 = self.smooth2(p3)

        P3_m, P3_list = self.decent_regression_block(p3, P4_list)
        pred_mano_results3, _ = self.mano_branch(P3_m)
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p2 = self.smooth3(p2)

        p2 = self.pool(p2)
        mano_results = [pred_mano_results1, pred_mano_results2, pred_mano_results3]
        out_hm, struct_f, preds_joints_img = self.hand_regHead(p2)
        return p2, P3_list, mano_results, out_hm, struct_f, preds_joints_img


class Decent2Fine(nn.Module):
    def __init__(self, mano_reg=None):
        super(Decent2Fine, self).__init__()
        self.seg_emb = seg_emb()
        self.hand_Encoder = hand_Encoder()
        self.mano_regHead = mano_reg


    def forward(self, struct_f, out_hm, seg_info, r_f, gt_mano_params):
        seg_f = self.seg_emb(seg_info[0], seg_info[1])
        mano_encoding = self.hand_Encoder(struct_f, out_hm, seg_f, r_f)
        pred_mano_results, gt_mano_results = self.mano_regHead(mano_encoding, gt_mano_params)
        return pred_mano_results, gt_mano_results


