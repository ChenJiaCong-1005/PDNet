"""
Implementation of ESDNet for image demoireing
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from torch.nn.parameter import Parameter

class T_my_model(nn.Module):
    def __init__(self,
                 en_feature_num,
                 en_inter_num,
                 de_feature_num,
                 de_inter_num,
                 sam_number=1,
                 ):
        super(T_my_model, self).__init__()
        self.encoder = Encoder(feature_num=en_feature_num, inter_num=en_inter_num, sam_number=sam_number)
        self.decoder = Decoder(en_num=en_feature_num, feature_num=de_feature_num, inter_num=de_inter_num,
                               sam_number=sam_number)

    def forward(self, x):
        y_1, y_2, y_3 = self.encoder(x)
        out_1, out_2, out_3, feat_2, feat_3 = self.decoder(y_1, y_2, y_3)
        feature_map = []
        feature_map.append(y_1)
        feature_map.append(y_2)
        feature_map.append(y_3)
        feature_map.append(feat_2)
        feature_map.append(feat_3)
        return out_1, out_2, out_3, feature_map

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)


class Decoder(nn.Module):
    def __init__(self, en_num, feature_num, inter_num, sam_number):
        super(Decoder, self).__init__()
        self.preconv_3 = conv_relu(4 * en_num, feature_num, 3, padding=1)
        self.decoder_3 = Decoder_Level(feature_num, inter_num, sam_number)

        self.preconv_2 = conv_relu(2 * en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_2 = Decoder_Level(feature_num, inter_num, sam_number)

        self.preconv_1 = conv_relu(en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_1 = Decoder_Level(feature_num, inter_num, sam_number)

    def forward(self, y_1, y_2, y_3):
        x_3 = y_3
        x_3 = self.preconv_3(x_3)
        out_3, feat_3 = self.decoder_3(x_3)

        x_2 = torch.cat([y_2, feat_3], dim=1)
        x_2 = self.preconv_2(x_2)
        out_2, feat_2 = self.decoder_2(x_2)

        x_1 = torch.cat([y_1, feat_2], dim=1)
        x_1 = self.preconv_1(x_1)
        out_1 = self.decoder_1(x_1, feat=False)

        return out_1, out_2, out_3, feat_2, feat_3


class Encoder(nn.Module):
    def __init__(self, feature_num, inter_num, sam_number):
        super(Encoder, self).__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(12, feature_num, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True)
        )
        self.encoder_1 = Encoder_Level(feature_num, inter_num, level=1, sam_number=sam_number)
        self.encoder_2 = Encoder_Level(2 * feature_num, inter_num, level=2, sam_number=sam_number)
        self.encoder_3 = Encoder_Level(4 * feature_num, inter_num, level=3, sam_number=sam_number)

    def forward(self, x):
        x = F.pixel_unshuffle(x, 2)
        x = self.conv_first(x)

        out_feature_1, down_feature_1 = self.encoder_1(x)
        out_feature_2, down_feature_2 = self.encoder_2(down_feature_1)
        out_feature_3 = self.encoder_3(down_feature_2)

        return out_feature_1, out_feature_2, out_feature_3


class Encoder_Level(nn.Module):
    def __init__(self, feature_num, inter_num, level, sam_number):
        super(Encoder_Level, self).__init__()
        #self.rdb = RDB(in_channel=feature_num, d_list=(1, 2, 1), inter_num=inter_num)
        self.rdb = GhostModule(feature_num, inter_num, d_list=(1,2,1))
        self.sam_blocks = nn.ModuleList()
        for _ in range(sam_number):
            sam_block = SAM(in_channel=feature_num, d_list=(1, 2, 3, 4, 5, 4, 3, 2, 1), inter_num=inter_num)
            self.sam_blocks.append(sam_block)

        if level < 3:
            self.down = nn.Sequential(
                nn.Conv2d(feature_num, 2 * feature_num, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )
        self.level = level

    def forward(self, x):
        out_feature = self.rdb(x)
        for sam_block in self.sam_blocks:
            out_feature = sam_block(out_feature)
        if self.level < 3:
            down_feature = self.down(out_feature)
            return out_feature, down_feature
        return out_feature


class Decoder_Level(nn.Module):
    def __init__(self, feature_num, inter_num, sam_number):
        super(Decoder_Level, self).__init__()
        #self.rdb = RDB(feature_num, (1, 2, 1), inter_num)
        self.rdb = GhostModule(feature_num, inter_num, d_list=(1,2,1))
        self.sam_blocks = nn.ModuleList()
        for _ in range(sam_number):
            sam_block = SAM(in_channel=feature_num, d_list=(1, 2, 3, 4, 5, 4, 3, 2, 1), inter_num=inter_num)
            self.sam_blocks.append(sam_block)
        self.conv = conv(in_channel=feature_num, out_channel=12, kernel_size=3, padding=1)

    def forward(self, x, feat=True):
        x = self.rdb(x)
        for sam_block in self.sam_blocks:
            x = sam_block(x)
        out = self.conv(x)
        out = F.pixel_shuffle(out, 2)

        if feat:
            feature = F.interpolate(x, scale_factor=2, mode='bilinear')
            return out, feature
        else:
            return out

class GhostBlock(nn.Module):
    def __init__(self, inp, oup, d_list=1, padding=1, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostBlock, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
 
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
 
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, padding=padding, dilation=d_list,  bias=False),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
 
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

class GhostModule(nn.Module):
    def __init__(self, inp, inter, d_list):
        super(GhostModule, self).__init__()
        self.ghost_layers = nn.ModuleList()
        c = inp
        for i in range(len(d_list)):
            ghost_conv = GhostBlock(c, inter, d_list=d_list[i], padding=d_list[i])
            self.ghost_layers.append(ghost_conv)
            c = c + inter
        self.conv = nn.Conv2d(c, inp, 1)

    def forward(self, x):
        t = x
        for ghost_conv in self.ghost_layers:
            _t = ghost_conv(t)
            t = torch.cat([_t, t], dim=1)
        t = self.conv(t)
        return t + x

class GhostModule1(nn.Module):
    def __init__(self, inp, inter, d_list):
        super(GhostModule1, self).__init__()
        self.ghost_layers = nn.ModuleList()
        c = inp
        for i in range(len(d_list)):
            ghost_conv = GhostBlock(c, inter, d_list=d_list[i], padding=d_list[i])
            self.ghost_layers.append(ghost_conv)
            c = c + inter
        self.conv = nn.Conv2d(c, inp, 1)

    def forward(self, x):
        t = x
        for ghost_conv in self.ghost_layers:
            _t = ghost_conv(t)
            t = torch.cat([_t, t], dim=1)
        t = self.conv(t)
        return t 


class DB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(DB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)
        t = self.conv_post(t)
        return t


class SAM(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(SAM, self).__init__()
        self.basic_block = GhostModule1(in_channel, inter_num, d_list)
        self.basic_block_2 = GhostModule1(in_channel, inter_num, d_list)
        self.basic_block_4 = GhostModule1(in_channel, inter_num, d_list)
        self.fusion = ECA(3 * in_channel)

    def forward(self, x):
        x_0 = x
        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')

        y_0 = self.basic_block(x_0)
        y_2 = self.basic_block_2(x_2)
        y_4 = self.basic_block_4(x_4)

        y_2 = F.interpolate(y_2, scale_factor=2, mode='bilinear')
        y_4 = F.interpolate(y_4, scale_factor=4, mode='bilinear')

        y = torch.cat([y_0, y_2, y_4], dim=1)
        attn = self.fusion(y)
        w0, w2, w4 = torch.chunk(attn, 3, dim=1)
        y = w0 * y_0 + w2 * y_2 + w4 * y_4
        y = x + y

        return y


class ECA(nn.Module):           # Efficient Channel Attention module
    def __init__(self, c, b=1, gamma=2):
        super(ECA, self).__init__()
        #t = int(abs((math.log(c, 2) + b) / gamma))
        #k = t if t % 2 else t + 1
        k = 9

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channelshuffle = ChannelShuffle(3)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.channelunshuffle = ChannelUnshuffle(3)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv2(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        x = self.relu(x)
        x = self.channelshuffle(x)
        x = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        x = self.channelunshuffle(x)
        out = self.sigmoid(x)
        return out


class RDB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(RDB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)

        t = self.conv_post(t)
        return t + x


class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=True, dilation=dilation_rate)

    def forward(self, x_input):
        out = self.conv(x_input)
        return out


class conv_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=True, dilation=dilation_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_input):
        out = self.conv(x_input)
        return out

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups
    
    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        
        # reshape
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        
        # transpose
        x = torch.transpose(x, 1, 2).contiguous()
        
        # reshape back
        x = x.view(batchsize, -1, height, width)
        
        return x

class ChannelUnshuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelUnshuffle, self).__init__()
        self.groups = groups
    
    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        
        # reshape
        x = x.view(batchsize, channels_per_group, self.groups, height, width)
        
        # transpose
        x = torch.transpose(x, 1, 2).contiguous()
        
        # reshape back
        x = x.view(batchsize, -1, height, width)
        
        return x