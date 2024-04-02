import torch
import torch.nn as nn



class Residual(nn.Module):
    def __init__(self, numIn, numOut):
        super(Residual, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        self.bn = nn.BatchNorm2d(self.numIn)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.numIn, self.numOut // 2, bias=True, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.numOut // 2)
        self.conv2 = nn.Conv2d(self.numOut // 2, self.numOut // 2, bias=True, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.numOut // 2)
        self.conv3 = nn.Conv2d(self.numOut // 2, self.numOut, bias=True, kernel_size=1)

        if self.numIn != self.numOut:
            self.conv4 = nn.Conv2d(self.numIn, self.numOut, bias=True, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.bn(x)
        out = self.leakyrelu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.leakyrelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leakyrelu(out)
        out = self.conv3(out)

        if self.numIn != self.numOut:
            residual = self.conv4(x)

        return out + residual

class pre_block(nn.Module):
    def __init__(self, in_feature=False, out_feature=False):
        super(pre_block, self).__init__()
        self.res = Residual(256, 256)
        self.pooling = nn.MaxPool2d(2, 2)
        self.in_feature = in_feature
        self.out_feature = out_feature

    def forward(self, x, in_f=None):
        x = self.res1(x)
        x1 = self.pooling(x)
        if self.in_feature:
            x = x1 + in_f
        if self.out_feature:
            return x, x1
        else:
            return x


class post_block(nn.Module):
    def __init__(self, ):
        super(post_block, self).__init__()
        self.res = Residual(256, 256)
        self.pooling = nn.MaxPool2d(2, 2)


    def forward(self, x, in_f=None):
        x = self.res1(x)
        x = self.pooling(x)
        x = x.view(x.shape[0], -1)
        return x



class regression_block(nn.Module):
    def __init__(self, level=1):
        super(regression_block, self).__init__()

        self.pooling = nn.MaxPool2d(2, 2)
        self.res1 = Residual(256, 256)
        self.res_blocks = [self.res1]
        for i in range(level):
            setattr(self, 'res{}'.format(i+2), Residual(256, 256))
            exec('self.res_blocks.append(self.res{})'.format(i+2))
        self.level = level

    def forward(self, x, last_list):
        x_out = []
        # pre-block
        for level in range(self.level):
            x = self.res_blocks[level](x)
            x = self.pooling(x)
            x_out.append(x)
            if level >= 1:
                x = x + last_list[level-1]
        # post-block
        x = self.res_blocks[-1](x)
        x = self.pooling(x)
        x = x.view(x.shape[0], -1)
        return x, x_out


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size,groups=1):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=1, padding=((kernel_size - 1) // 2),
                      groups=groups,bias=True),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class seg_emb(nn.Module):
    def __init__(self):
        super(seg_emb, self).__init__()
        self.conv_pool = nn.Conv2d(2, 16, 2, 2)
        self.whole_block = BasicBlock(16, 256, 1)

    def forward(self, whole_seg, vis_seg):
        emb_f = self.whole_block(self.conv_pool(torch.cat([whole_seg, vis_seg], dim=1)))
        return emb_f









