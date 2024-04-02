import torch.nn as nn
import torch
from common.nets.morphology_attention import HMA



class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class SGD(nn.Module):
    def __init__(self):
        super(SGD, self).__init__()

        self.conv3 = VGGBlock(1024 + 1024, 1024, 1024)
        self.conv2 = VGGBlock(512 + 512, 512, 512)
        self.conv1 = VGGBlock(256 + 256, 256, 64)
        self.out_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(2048, 1024, 1, 1))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(1024, 512, 1, 1))
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(512, 256, 1, 1))

        self.hand_morphology_attention = HMA()


    def forward(self, x, structure_f):
        c5, c4, c3, c2 = x

        vis_seg_c4 = self.conv3(torch.cat([self.up3(c5), c4], dim=1))
        vis_seg_c3 = self.conv2(torch.cat([self.up2(vis_seg_c4), c3], dim=1))
        vis_seg_c2 = self.conv1(torch.cat([self.up1(vis_seg_c3), c2], dim=1))
        vis_seg_out = self.out_layer(vis_seg_c2)
        vis_seg = torch.sigmoid(vis_seg_out)

        whole_seg_f = self.hand_morphology_attention(structure_f[-1], vis_seg_c2)
        whole_seg = torch.sigmoid(whole_seg_f)

        return [whole_seg, vis_seg]


