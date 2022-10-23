# !/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
from tasks.semantic.modules.SalsaNext import ResContextBlock, ResBlock, UpBlock


# MetaKernel definition
class MetaKernel(nn.Module):
    def __init__(self, in_filters=5, out_filters=32, kernel_size=(3, 3)):
        super(MetaKernel, self).__init__()
        self.kernel = kernel_size[0]
        self.mlp_conv1 = nn.Conv2d(4, out_filters, kernel_size=(1, 1), stride=1)
        self.mlp_act1 = nn.ReLU()
        self.mlp_bn1 = nn.BatchNorm2d(out_filters)
        self.mlp_conv2 = nn.Conv2d(out_filters, in_filters, kernel_size=(1, 1), stride=1)
        self.mlp_act2 = nn.ReLU()
        self.mlp_bn2 = nn.BatchNorm2d(in_filters * self.kernel * self.kernel)

        self.aggregate_conv = nn.Conv2d(in_filters * self.kernel * self.kernel, out_filters,
                                        kernel_size=(1, 1), stride=1)
        self.aggregate_act = nn.ReLU()
        self.aggregate_bn = nn.BatchNorm2d(out_filters)

    def forward(self, range_residual_image):
        coord_channels_num = 4  # 4 for (r, x, y, z)
        coord_data = range_residual_image[:, 0:coord_channels_num, :, :]
        num_batch = range_residual_image.size()[0]
        in_channels = range_residual_image.size()[1]
        H = range_residual_image.size()[2]
        W = range_residual_image.size()[3]
        dilate = 1
        pad = ((self.kernel - 1) * dilate + 1) // 2

        # sampling from range residual image
        unfold = torch.nn.Unfold(kernel_size=(self.kernel, self.kernel), dilation=1, padding=pad, stride=1)
        data_sample = unfold(range_residual_image)
        data_sample = torch.reshape(data_sample,
                                    shape=(num_batch, in_channels, self.kernel * self.kernel, H, W))
        coord_sample = unfold(coord_data)
        coord_sample = torch.reshape(coord_sample,
                                     shape=(num_batch, coord_channels_num, self.kernel * self.kernel, H, W))
        #  get the range difference and relative Cartesian coordinates of neighbors for the center
        center_coord = torch.unsqueeze(coord_data, 2)
        rel_coord = coord_sample - center_coord

        # get dynamic weights from the shared mlp
        rel_coord = torch.reshape(rel_coord, shape=(num_batch, coord_channels_num, -1, W))
        weights = self.mlp_conv1(rel_coord)
        weights = self.mlp_act1(weights)
        weights = self.mlp_conv2(weights)
        weights = torch.reshape(weights, shape=(num_batch, in_channels, -1, H, W))

        # element-wise product
        dynamic_out = data_sample * weights
        dynamic_out = torch.reshape(dynamic_out, shape=(num_batch, -1, H, W))
        dynamic_out = self.mlp_bn2(dynamic_out)
        dynamic_out = self.mlp_act2(dynamic_out)

        # 1x1 convolution for aggregation
        meta_feature = self.aggregate_conv(dynamic_out)
        meta_feature = self.aggregate_bn(meta_feature)
        meta_feature = self.aggregate_act(meta_feature)

        return meta_feature


# Backbone definition, we use backbone similar to salsanext here and you can replace with others
class U_Net_backbone(nn.Module):
    def __init__(self, meta_channel, multi_scale_channel):
        super(U_Net_backbone, self).__init__()
        self.resBlock1 = ResBlock(meta_channel, 2 * meta_channel, 0.2, pooling=True, drop_out=False)
        self.resBlock2 = ResBlock(2 * meta_channel, 2 * 2 * meta_channel, 0.2, pooling=True)
        self.resBlock3 = ResBlock(2 * 2 * meta_channel, 2 * 4 * meta_channel, 0.2, pooling=True)
        self.resBlock4 = ResBlock(2 * 4 * meta_channel, 2 * 4 * meta_channel, 0.2, pooling=True)
        self.resBlock5 = ResBlock(2 * 4 * meta_channel, 2 * 4 * multi_scale_channel, 0.2, pooling=False)

        self.upBlock1 = UpBlock(2 * 4 * multi_scale_channel, 4 * multi_scale_channel, 0.2)
        self.upBlock2 = UpBlock(4 * multi_scale_channel, 4 * multi_scale_channel, 0.2)
        self.upBlock3 = UpBlock(4 * multi_scale_channel, 2 * multi_scale_channel, 0.2)
        self.upBlock4 = UpBlock(2 * multi_scale_channel, multi_scale_channel, 0.2, drop_out=False)

    def forward(self, meta_feature):
        down0c, down0b = self.resBlock1(meta_feature)
        down1c, down1b = self.resBlock2(down0c)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down5c = self.resBlock5(down3c)

        up4e = self.upBlock1(down5c, down3b)
        up3e = self.upBlock2(up4e, down2b)
        up2e = self.upBlock3(up3e, down1b)
        up1e = self.upBlock4(up2e, down0b)

        multi_scale_feature = up1e

        return multi_scale_feature


# Feature Aggregation Module(FAM) definition
class FeatureAggregationModule(nn.Module):
    def __init__(self, range_channel, range_context_channel,
                 multi_scale_channel, meta_channel, final_channel, n_classes):
        super(FeatureAggregationModule, self).__init__()
        self.range_enc1 = ResContextBlock(range_channel, range_context_channel)
        self.range_enc2 = ResContextBlock(range_context_channel, range_context_channel)
        self.range_enc3 = ResContextBlock(range_context_channel, range_context_channel)

        self.rg_conv1 = nn.Conv2d(multi_scale_channel + range_context_channel, multi_scale_channel,
                                  kernel_size=3, padding=1, stride=1)
        self.rg_act1 = nn.LeakyReLU()
        self.rg_bn1 = nn.BatchNorm2d(multi_scale_channel)
        self.rg_conv2 = nn.Conv2d(multi_scale_channel, multi_scale_channel, kernel_size=3, padding=1, stride=1)
        self.rg_act2 = nn.ReLU(inplace=True)
        self.rg_bn2 = nn.BatchNorm2d(multi_scale_channel)
        self.rg_conv3 = nn.Conv2d(multi_scale_channel, multi_scale_channel, kernel_size=3, padding=1, stride=1)
        self.rg_act3 = nn.BatchNorm2d(multi_scale_channel)
        self.rg_bn3 = nn.Sigmoid()

        self.rc_conv1 = nn.Conv2d(multi_scale_channel + meta_channel, final_channel,
                                  kernel_size=(3, 3), padding=1, dilation=1)
        self.rc_act1 = nn.LeakyReLU()
        self.rc_bn1 = nn.BatchNorm2d(final_channel)
        self.rc_conv2 = nn.Conv2d(final_channel, final_channel, kernel_size=(3, 3), padding=2, dilation=2)
        self.rc_act2 = nn.LeakyReLU()
        self.rc_bn2 = nn.BatchNorm2d(final_channel)
        self.rc_conv3 = nn.Conv2d(2 * final_channel, final_channel, kernel_size=(1, 1), dilation=1)
        self.rc_act3 = nn.LeakyReLU()
        self.rc_bn3 = nn.BatchNorm2d(final_channel)

        self.range_logits = nn.Conv2d(final_channel, n_classes, kernel_size=(1, 1))

    def forward(self, range_channel, multi_scale_feature, meta_feature):
        # get range context information
        range_context = self.range_enc1(range_channel)
        range_context = self.range_enc2(range_context)
        range_context = self.range_enc3(range_context)

        # fuse range context information and multi-scale feature, get range guided feature
        concat_feature1 = torch.cat((multi_scale_feature, range_context), dim=1)
        fuse_feature1 = self.rg_conv1(concat_feature1)
        fuse_feature1 = self.rg_act1(fuse_feature1)
        fuse_feature1 = self.rg_bn1(fuse_feature1)

        attention_mask = self.rg_conv2(fuse_feature1)
        attention_mask = self.rg_bn2(attention_mask)
        attention_mask = self.rg_act2(attention_mask)
        attention_mask = self.rg_conv3(attention_mask)
        attention_mask = self.rg_bn3(attention_mask)
        attention_mask = self.rg_act3(attention_mask)

        range_guided_feature = fuse_feature1 * attention_mask + multi_scale_feature

        # fuse range guided feature and meta feature, get final feature
        concat_feature2 = torch.cat((range_guided_feature, meta_feature), dim=1)
        fuse_feature2 = self.rc_conv1(concat_feature2)
        fuse_feature2 = self.rc_bn1(fuse_feature2)
        fuse_feature2 = self.rc_act1(fuse_feature2)

        fuse_feature3 = self.rc_conv2(fuse_feature2)
        fuse_feature3 = self.rc_bn2(fuse_feature3)
        fuse_feature3 = self.rc_act2(fuse_feature3)

        concat_feature3 = torch.cat((fuse_feature2, fuse_feature3), dim=1)
        final_feature = self.rc_conv3(concat_feature3)
        final_feature = self.rc_bn3(final_feature)
        final_feature = self.rc_act3(final_feature)

        # 1x1 conv to get logits under range view
        range_view_logits = self.range_logits(final_feature)

        return range_view_logits


# Our Meta-RangeSeg
class MetaRangeSeg(nn.Module):
    def __init__(self, n_classes):
        super(MetaRangeSeg, self).__init__()
        if n_classes == 26:
            self.meta_kernel = MetaKernel(9, 32, (5, 5))
        else:
            self.meta_kernel = MetaKernel(6, 32, (5, 5))
        self.backbone = U_Net_backbone(32, 32)
        self.feature_aggregation = FeatureAggregationModule(1, 32, 32, 32, 32, n_classes)

    def forward(self, range_residual_image):
        # get range channel
        range_channel = range_residual_image[:, 0:1, :, :]
        # get meta feature
        meta_feature = self.meta_kernel(range_residual_image)
        # get multi-scale feature
        multi_scale_feature = self.backbone(meta_feature)
        # aggregate above features and get logits under range view
        range_view_logits = self.feature_aggregation(range_channel, multi_scale_feature, meta_feature)
        range_view_logits = F.softmax(range_view_logits, dim=1)

        return range_view_logits
