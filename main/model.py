import torch
import torch.nn as nn
from torch.nn import functional as F
from common.nets.mano_head import mano_regHead
from common.nets.Backbone import resnet_backbone
from common.nets.PMR_module import Coarse2Decent, Decent2Fine
from common.nets.SGD_module import SGD
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, TexturesVertex, PerspectiveCameras, MeshRenderer
from pytorch3d.structures import Meshes
from config import cfg


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

        self.seg_loss = BCEDiceLoss().cuda()
        self.raster_setting = RasterizationSettings(image_size=(256, 256), blur_radius=0.0, faces_per_pixel=1, )
        self.R = torch.Tensor([[[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]])
        self.T = torch.Tensor([[0., 0., 0.]])
        self.faces = self.decent2fine.mano_regHead.mano_layer.th_faces

    def get_pred_mask(self, pred_mano_results, meta_info):
        K = meta_info['K']
        N = K.shape[0]
        focal_length = torch.zeros([N, 2])
        focal_length[:, 0] = K[:, 0, 0] / 127.5
        focal_length[:, 1] = K[:, 1, 1] / 127.5
        prince_point = torch.zeros([N, 2])
        prince_point[:, 0] = (127.5 - K[:, 0, 2]) / 127.5
        prince_point[:, 1] = (127.5 - K[:, 1, 2]) / 127.5
        camera = PerspectiveCameras(device='cuda', R=self.R.repeat(N, 1, 1), T=self.T.repeat(N, 1),
                                    focal_length=focal_length, principal_point=prince_point)
        raster = MeshRasterizer(cameras=camera, raster_settings=self.raster_setting)
        mesh = Meshes(verts=pred_mano_results['verts3d'] - pred_mano_results['joints3d'][:, 0:1, :] + meta_info[
            'root_joint_cam_rot'], faces=self.faces[None, :, :].repeat(N, 1, 1).cuda())
        depth_map = raster(mesh).zbuf
        pred_mask = (torch.sign(depth_map) + 1) / 2
        return pred_mask

    def mano_loss(self, loss, pred_mano, gt_mano, c2f_level, weight):
        loss['mano_verts_{}'.format(c2f_level)] = cfg.lambda_mano_verts * F.mse_loss(pred_mano['verts3d'],
                                                                gt_mano['verts3d']) * weight * cfg.loss_scale
        loss['mano_joints_{}'.format(c2f_level)] = cfg.lambda_mano_joints * F.mse_loss(pred_mano['joints3d'],
                                                                  gt_mano['joints3d']) * weight  * cfg.loss_scale
        loss['mano_pose_{}'.format(c2f_level)] = cfg.lambda_mano_pose * F.mse_loss(pred_mano['mano_pose'],
                                                              gt_mano['mano_pose']) * weight  * cfg.loss_scale
        loss['mano_shape_{}'.format(c2f_level)] = cfg.lambda_mano_shape * F.mse_loss(pred_mano['mano_shape'],
                                                                gt_mano['mano_shape']) * weight  * cfg.loss_scale

    def forward(self, inputs, targets, meta_info, mode):
        feats = self.backbone(inputs['img'])
        feat, regression_f, mano_results, out_hm, struct_f, preds_joints_img = self.coarse2decent(feats)
        seg_info, seg_f = self.sgd(feats, struct_f)

        if mode == 'train':
            gt_mano_params = torch.cat([targets['mano_pose'], targets['mano_shape']], dim=1)
        else:
            gt_mano_params = None
        pred_mano_results, gt_mano_results = self.decent2fine(struct_f, out_hm, seg_info, regression_f, gt_mano_params)
        if mode == 'train':
            loss = {}
            pred_mask = self.get_pred_mask(pred_mano_results, meta_info) * cfg.lambda_seg * cfg.loss_scale
            loss['render'] = F.mse_loss(targets['mask_whole_patch_256'][:, 0, :, :], pred_mask[:, :, :, 0]) * cfg.lambda_proj * cfg.loss_scale
            loss['seg'] = self.seg_loss(seg_f[1], targets['mask']) * cfg.lambda_seg * cfg.loss_scale
            loss['seg_whole'] = self.seg_loss(seg_f[0], targets['mask_whole']) * cfg.lambda_seg * cfg.loss_scale
            self.mano_loss(loss, mano_results[0], gt_mano_results, 0, cfg.lambda_coarse2decent)
            self.mano_loss(loss, mano_results[1], gt_mano_results, 1, cfg.lambda_coarse2decent)
            self.mano_loss(loss, mano_results[2], gt_mano_results, 2, cfg.lambda_coarse2decent)
            self.mano_loss(loss, pred_mano_results, gt_mano_results, 3, cfg.lambda_fine)
            return loss

        else:
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
    y = model({'img': torch.ones([1, 3, 256, 256])}, None, None, 'test')
