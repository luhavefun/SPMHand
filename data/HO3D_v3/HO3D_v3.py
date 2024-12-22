import os
import os.path as osp
import numpy as np
import torch
import cv2
import random
import json
import math
import copy
from pycocotools.coco import COCO
from main.config import cfg
from common.utils.preprocessing import load_img, get_bbox, process_bbox, generate_patch_image, augmentation
from common.utils.transforms import world2cam, cam2pixel, pixel2cam, rigid_align, transform_joint_to_other_db
# from utils.vis import vis_keypoints, vis_mesh, save_obj, vis_keypoints_with_skeleton
from common.utils.mano import MANO

mano = MANO()
import torchvision.transforms as transforms


def get_affine_trans_no_rot(center, scale, res):
    affinet = np.zeros((3, 3))
    affinet[0, 0] = float(res[0]) / scale
    affinet[1, 1] = float(res[1]) / scale
    affinet[0, 2] = res[1] * (-float(center[0]) / scale + .5)
    affinet[1, 2] = res[0] * (-float(center[1]) / scale + .5)
    affinet[2, 2] = 1
    return affinet


def get_affine_transform(center, scale, res, rot=0, K=None):
    rot_mat = np.zeros((3, 3))
    sn, cs = np.sin(rot), np.cos(rot)
    rot_mat[0, :2] = [cs, -sn]
    rot_mat[1, :2] = [sn, cs]
    rot_mat[2, 2] = 1
    origin_rot_center = rot_mat.dot(center.tolist() + [1, ])[:2]
    post_rot_trans = get_affine_trans_no_rot(origin_rot_center, scale, res)
    total_trans = post_rot_trans.dot(rot_mat)
    if K is not None:
        t_mat = np.eye(3)
        t_mat[0, 2] = -K[0, 2]
        t_mat[1, 2] = -K[1, 2]
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        transformed_center = t_inv.dot(rot_mat).dot(t_mat).dot(center.tolist() + [1, ])
        affinetrans_post_rot = get_affine_trans_no_rot(transformed_center[:2], scale, res)
        return total_trans.astype(np.float32), affinetrans_post_rot.astype(np.float32), rot_mat.astype(np.float32)
    else:
        return total_trans.astype(np.float32), rot_mat.astype(np.float32)


class HO3D_v3(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform

        self.data_split = data_split if data_split == 'train' else 'evaluation'
        self.root_dir = osp.join('..', 'data', 'HO3D_v3', 'data')
        self.annot_path = osp.join(self.root_dir, 'annotations')

        self.root_joint_idx = 0

        self.datalist = self.load_data()
        if self.data_split != 'train':
            self.eval_result = [[], []]  # [pred_joints_list, pred_verts_list]
        self.joints_name = (
        'Wrist', 'Index_1', 'Index_2', 'Index_3', 'Middle_1', 'Middle_2', 'Middle_3', 'Pinky_1', 'Pinky_2', 'Pinky_3',
        'Ring_1', 'Ring_2', 'Ring_3', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_4', 'Middle_4', 'Ring_4',
        'Pinly_4')

    def load_data(self):
        db = COCO(osp.join(self.annot_path, "HO3D_v3_{}_data.json".format(self.data_split)))

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]

            img_path = osp.join(self.root_dir, self.data_split, img['file_name'])
            img_shape = (img['height'], img['width'])
            if self.data_split == 'train':
                joints_coord_cam = np.array(ann['joints_coord_cam'], dtype=np.float32)  # meter
                cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann['cam_param'].items()}
                joints_coord_img = cam2pixel(joints_coord_cam, cam_param['focal'], cam_param['princpt'])
                bbox = get_bbox(joints_coord_img[:, :2], np.ones_like(joints_coord_img[:, 0]), expansion_factor=1.5)
                bbox = process_bbox(bbox, img['width'], img['height'], expansion_factor=1.0)
                if bbox is None:
                    continue

                mano_pose = np.array(ann['mano_param']['pose'], dtype=np.float32)
                mano_shape = np.array(ann['mano_param']['shape'], dtype=np.float32)

                data = {"img_path": img_path, "img_shape": img_shape, "joints_coord_cam": joints_coord_cam,
                        "joints_coord_img": joints_coord_img,
                        "bbox": bbox, "cam_param": cam_param, "mano_pose": mano_pose, "mano_shape": mano_shape,
                        "file_name": img['file_name']}
            else:
                root_joint_cam = np.array(ann['root_joint_cam'], dtype=np.float32)
                cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann['cam_param'].items()}
                bbox = np.array(ann['bbox'], dtype=np.float32)
                bbox = process_bbox(bbox, img['width'], img['height'], expansion_factor=1.5)

                data = {"img_path": img_path, "img_shape": img_shape, "root_joint_cam": root_joint_cam,
                        "bbox": bbox, "cam_param": cam_param}

            datalist.append(data)

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):

        # idx = 0

        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox, cam_para = data['img_path'], data['img_shape'], data['bbox'], data['cam_param']

        K = np.zeros([3, 3]).astype(np.float32)
        K[0, 0] = cam_para['focal'][0]
        K[1, 1] = cam_para['focal'][1]
        K[0, 2] = cam_para['princpt'][0]
        K[1, 2] = cam_para['princpt'][1]
        K[2, 2] = 1.

        'seg part'
        if self.data_split == 'train':
            seq, rgb, id = data['file_name'].split('/')
            mask = cv2.imread(os.path.join(self.seg_root, seq, 'seg', id[:5] + 'png'))
            mask = mask[:, :, 0:1]
            mask = cv2.resize(mask, (640, 480), cv2.INTER_LINEAR)
            mask_whole = cv2.imread(os.path.join(self.seg_whole_root, seq, id[:5] + 'jpg'), cv2.IMREAD_GRAYSCALE)
        else:
            mask = None
            mask_whole = None

        # img

        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, scale, mask, mask_whole, [bb_c_x, bb_c_y,
                                                                        bb_width], mask_whole_patch_256 = augmentation(
            img, bbox, self.data_split, do_flip=False, mask=mask, mask_whole=mask_whole)
        _, post_rot_trans, _ = get_affine_transform(np.asarray([bb_c_x, bb_c_y]), bb_width, [256, 256],
                                                    rot=-rot / 180 * np.pi, K=K)
        K = post_rot_trans.dot(K)
        # img_s = img.copy()
        img = self.transform(img.astype(np.float32)) / 255.
        if self.data_split == 'train':
            mask = self.transform(mask.astype(np.float32)) / 255.
            mask_whole = self.transform(mask_whole.astype(np.float32)) / 255.
            mask_whole_patch_256 = self.transform(mask_whole_patch_256.astype(np.float32)) / 255.

        if self.data_split == 'train':
            ## 2D joint coordinate
            joints_img = data['joints_coord_img']
            joints_img_xy1 = np.concatenate((joints_img[:, :2], np.ones_like(joints_img[:, :1])), 1)
            joints_img = np.dot(img2bb_trans, joints_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
            # normalize to [0,1]
            joints_img[:, 0] /= cfg.input_img_shape[1]
            joints_img[:, 1] /= cfg.input_img_shape[0]

            ## 3D joint camera coordinate
            joints_coord_cam = data['joints_coord_cam']
            root_joint_cam = copy.deepcopy(joints_coord_cam[self.root_joint_idx])
            root_joint_cam_rot = root_joint_cam.copy()[None, :]
            joints_coord_cam -= joints_coord_cam[self.root_joint_idx, None, :]  # root-relative
            # 3D data rotation augmentation
            rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                                    [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                                    [0, 0, 1]], dtype=np.float32)
            joints_coord_cam = np.dot(rot_aug_mat, joints_coord_cam.transpose(1, 0)).transpose(1, 0)

            root_joint_cam_rot = np.dot(rot_aug_mat, root_joint_cam_rot.transpose(1, 0)).transpose(1, 0)

            ## mano parameter
            mano_pose, mano_shape = data['mano_pose'], data['mano_shape']
            # 3D data rotation augmentation
            mano_pose = mano_pose.reshape(-1, 3)
            root_pose = mano_pose[self.root_joint_idx, :]
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat, root_pose))
            mano_pose[self.root_joint_idx] = root_pose.reshape(3)
            mano_pose = mano_pose.reshape(-1)

            inputs = {'img': img}
            targets = {'joints_img': joints_img, 'joints_coord_cam': joints_coord_cam, 'mano_pose': mano_pose,
                       'mano_shape': mano_shape, 'mask': mask, 'mask_whole': mask_whole,
                       'mask_whole_patch_256': mask_whole_patch_256}
            meta_info = {'root_joint_cam': root_joint_cam, 'root_joint_cam_rot': root_joint_cam_rot, 'K': K}

        else:
            root_joint_cam = data['root_joint_cam']
            inputs = {'img': img}
            targets = {}
            meta_info = {'root_joint_cam': root_joint_cam}

        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]

            out = outs[n]

            verts_out = out['mesh_coord_cam']
            joints_out = out['joints_coord_cam']

            # root align
            gt_root_joint_cam = annot['root_joint_cam']
            verts_out = verts_out - joints_out[self.root_joint_idx] + gt_root_joint_cam
            joints_out = joints_out - joints_out[self.root_joint_idx] + gt_root_joint_cam

            # convert to openGL coordinate system.
            verts_out *= np.array([1, -1, -1])
            joints_out *= np.array([1, -1, -1])

            # convert joint ordering from MANO to HO3D.
            joints_out = transform_joint_to_other_db(joints_out, mano.joints_name, self.joints_name)

            self.eval_result[0].append(joints_out.tolist())
            self.eval_result[1].append(verts_out.tolist())

    def print_eval_result(self, test_epoch):
        output_json_file = osp.join(cfg.result_dir, 'pred{}.json'.format(test_epoch))
        output_zip_file = osp.join(cfg.result_dir, 'pred{}.zip'.format(test_epoch))

        with open(output_json_file, 'w') as f:
            json.dump(self.eval_result, f)
        print('Dumped %d joints and %d verts predictions to %s' % (
        len(self.eval_result[0]), len(self.eval_result[1]), output_json_file))

        cmd = 'zip -j ' + output_zip_file + ' ' + output_json_file
        print(cmd)
        os.system(cmd)

