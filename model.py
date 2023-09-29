from torch import nn
import torch


class Conv2dUnit(nn.Module): # Conv + BN + LeakyReLU
    def __init__(self, input_dim, filters, kernels, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, filters, kernel_size=kernels, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(filters)
        self.LeakyReLU = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.LeakyReLU(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, filters):
        super().__init__()
        self.conv1 = Conv2dUnit(input_dim, filters, (1, 1), stride=1, padding=0)
        self.conv2 = Conv2dUnit(filters, 2*filters, (3, 3), stride=1, padding=1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        return x


class StackResidualBlock(nn.Module):
    def __init__(self, input_dim, filters, n):
        super().__init__()
        self.resx = nn.Sequential()
        for i in range(n):
            self.resx.add_module('stack_%d' % (i+1,), ResidualBlock(input_dim, filters))

    def forward(self, x):
        return self.resx(x)


class Darknet(nn.Module):
    def __init__(self, initial_filters=32):
        super().__init__()
        i32 = initial_filters
        i64 = i32 * 2
        i128 = i32 * 4
        i256 = i32 * 8
        i512 = i32 * 16
        i1024 = i32 * 32

        # darknet53所有卷积层都没有偏移，bias=False
        self.conv1 = Conv2dUnit(3, i32, (3, 3), stride=1, padding=1)
        self.conv2 = Conv2dUnit(i32, i64, (3, 3), stride=2, padding=1)
        self.stack_residual_block_1 = StackResidualBlock(i64, i32, n=1)
        self.conv3 = Conv2dUnit(i64, i128, (3, 3), stride=2, padding=1)
        self.stack_residual_block_2 = StackResidualBlock(i128, i64, n=2)
        self.conv4 = Conv2dUnit(i128, i256, (3, 3), stride=2, padding=1)
        self.stack_residual_block_3 = StackResidualBlock(i256, i128, n=8)
        self.conv5 = Conv2dUnit(i256, i512, (3, 3), stride=2, padding=1)
        self.stack_residual_block_4 = StackResidualBlock(i512, i256, n=8)
        self.conv6 = Conv2dUnit(i512, i1024, (3, 3), stride=2, padding=1)
        self.stack_residual_block_5 = StackResidualBlock(i1024, i512, n=4)

        # FPN+YOLO head
        self.CBL5_1 = nn.Sequential(
            Conv2dUnit(i1024, i512, (1, 1), stride=1, padding=0),
            Conv2dUnit(i512, i1024, (3, 3), stride=1, padding=1),
            Conv2dUnit(i1024, i512, (1, 1), stride=1, padding=0),
            Conv2dUnit(i512, i1024, (3, 3), stride=1, padding=1),
            Conv2dUnit(i1024, i512, (1, 1), stride=1, padding=0)
        )
        self.yolo_head1 = nn.Sequential(
            Conv2dUnit(i512, i1024, (3, 3), stride=1, padding=1),
            nn.Conv2d(i1024, 3*5, kernel_size=(1, 1))
        )

        self.conv7 = Conv2dUnit(i512, i256, (1, 1), stride=1, padding=0)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.CBL5_2 = nn.Sequential(
            Conv2dUnit(i256+i512, i256, (1, 1), stride=1, padding=0),
            Conv2dUnit(i256, i512, (3, 3), stride=1, padding=1),
            Conv2dUnit(i512, i256, (1, 1), stride=1, padding=0),
            Conv2dUnit(i256, i512, (3, 3), stride=1, padding=1),
            Conv2dUnit(i512, i256, (1, 1), stride=1, padding=0)
        )
        self.yolo_head2 = nn.Sequential(
            Conv2dUnit(i256, i512, (3, 3), stride=1, padding=1),
            nn.Conv2d(i512, 3*5, kernel_size=(1, 1))
        )

        self.conv8 = Conv2dUnit(i256, i128, (1, 1), stride=1, padding=0)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.CBL5_3 = nn.Sequential(
            Conv2dUnit(i128+i256, i128, (1, 1), stride=1, padding=0),
            Conv2dUnit(i128, i256, (3, 3), stride=1, padding=1),
            Conv2dUnit(i256, i128, (1, 1), stride=1, padding=0),
            Conv2dUnit(i128, i256, (3, 3), stride=1, padding=1),
            Conv2dUnit(i256, i128, (1, 1), stride=1, padding=0)
        )
        self.yolo_head3 = nn.Sequential(
            Conv2dUnit(i128, i256, (3, 3), stride=1, padding=1),
            nn.Conv2d(i256, 3*5, kernel_size=(1, 1))
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.stack_residual_block_1(x)
        x = self.conv3(x)
        x = self.stack_residual_block_2(x)
        x = self.conv4(x)
        feat_downsample8 = self.stack_residual_block_3(x)
        x = self.conv5(feat_downsample8)
        feat_downsample16 = self.stack_residual_block_4(x)
        x = self.conv6(feat_downsample16)
        feat_downsample32 = self.stack_residual_block_5(x)

        downsample32 = self.CBL5_1(feat_downsample32)

        x = self.conv7(downsample32)
        x = self.upsample1(x)
        x = torch.cat((x, feat_downsample16), dim=1)
        downsample16 = self.CBL5_2(x)

        x = self.conv8(downsample16)
        x = self.upsample2(x)
        x = torch.cat((x, feat_downsample8), dim=1)
        downsample8 = self.CBL5_3(x)
        y3 = self.yolo_head3(downsample8)
        y3 = y3.view(y3.size(0), 3, 5, y3.size(2), y3.size(3))  # reshape

        y3 = y3.permute(0, 1, 3, 4, 2)
        return y3


class Darknet10(nn.Module):
    def __init__(self, initial_filters=32):
        super().__init__()
        i32 = initial_filters
        i64 = i32 * 2
        i128 = i32 * 4
        i256 = i32 * 8
        i512 = i32 * 16
        i1024 = i32 * 32

        # darknet53所有卷积层都没有偏移，bias=False
        self.conv1 = Conv2dUnit(3, i32, (3, 3), stride=1, padding=1)
        self.conv2 = Conv2dUnit(i32, i64, (3, 3), stride=2, padding=1)
        self.conv3 = Conv2dUnit(i64, i128, (3, 3), stride=2, padding=1)
        self.conv4 = Conv2dUnit(i128, i256, (3, 3), stride=2, padding=1)
        self.conv5 = Conv2dUnit(i256, i512, (3, 3), stride=2, padding=1)
        self.conv6 = Conv2dUnit(i512, i1024, (3, 3), stride=2, padding=1)

        # FPN+YOLO head
        self.conv7 = Conv2dUnit(i1024, i512, (1, 1), stride=1, padding=0)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv8 = Conv2dUnit(i1024, i256, (1, 1), stride=1, padding=0)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.yolo_head3 = nn.Sequential(
            Conv2dUnit(i512, i256, (3, 3), stride=1, padding=1),
            nn.Conv2d(i256, 3*5, kernel_size=(1, 1))
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        feat_downsample8 = self.conv4(x)
        feat_downsample16 = self.conv5(feat_downsample8)
        feat_downsample32 = self.conv6(feat_downsample16)

        x = self.conv7(feat_downsample32)
        x = self.upsample1(x)
        downsample16 = torch.cat((x, feat_downsample16), dim=1)

        x = self.conv8(downsample16)
        x = self.upsample2(x)
        downsample8 = torch.cat((x, feat_downsample8), dim=1)

        y3 = self.yolo_head3(downsample8)
        y3 = y3.view(y3.size(0), 3, 5, y3.size(2), y3.size(3))  # reshape

        y3 = y3.permute(0, 1, 3, 4, 2)
        return y3
