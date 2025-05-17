import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
import math
import numpy as np
from mamba_ssm import Mamba

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super(PVMLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim // 4,  # 模型维度
            d_state=d_state,  # SSM 状态扩展因子
            d_conv=d_conv,  # 局部卷积宽度
            expand=expand  # 块扩展因子
        )
        self.proj = nn.Linear(input_dim, output_dim)

        # 定义多尺度卷积块
        self.conv_block1 = nn.Sequential(
            DepthwiseSeparableConv(input_dim // 4, input_dim // 4, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(input_dim // 4),
            DepthwiseSeparableConv(input_dim // 4, input_dim // 4, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(input_dim // 4),
            DepthwiseSeparableConv(input_dim // 4, input_dim // 4, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(input_dim // 4),
            DepthwiseSeparableConv(input_dim // 4, input_dim // 4, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(input_dim // 4)
        )

        self.conv_block2 = nn.Sequential(
            DepthwiseSeparableConv(input_dim // 4, input_dim // 4, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(input_dim // 4),
            DepthwiseSeparableConv(input_dim // 4, input_dim // 4, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(input_dim // 4),
            DepthwiseSeparableConv(input_dim // 4, input_dim // 4, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(input_dim // 4),
            DepthwiseSeparableConv(input_dim // 4, input_dim // 4, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(input_dim // 4)
        )

        self.conv_block3 = nn.Sequential(
            DepthwiseSeparableConv(input_dim // 4, input_dim // 4, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(input_dim // 4),
            DepthwiseSeparableConv(input_dim // 4, input_dim // 4, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(input_dim // 4),
            DepthwiseSeparableConv(input_dim // 4, input_dim // 4, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(input_dim // 4),
            DepthwiseSeparableConv(input_dim // 4, input_dim // 4, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(input_dim // 4)
        )

        self.conv_block4 = nn.Sequential(
            DepthwiseSeparableConv(input_dim // 4, input_dim // 4, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(input_dim // 4),
            DepthwiseSeparableConv(input_dim // 4, input_dim // 4, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(input_dim // 4),
            DepthwiseSeparableConv(input_dim // 4, input_dim // 4, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(input_dim // 4),
            DepthwiseSeparableConv(input_dim // 4, input_dim // 4, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(input_dim // 4)
        )

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + self.conv_block1(x1.transpose(-1, -2).reshape(B, C // 4, *img_dims)).reshape(B, C // 4, n_tokens).transpose(-1, -2)
        x_mamba2 = self.mamba(x2) + self.conv_block2(x2.transpose(-1, -2).reshape(B, C // 4, *img_dims)).reshape(B, C // 4, n_tokens).transpose(-1, -2)
        x_mamba3 = self.mamba(x3) + self.conv_block3(x3.transpose(-1, -2).reshape(B, C // 4, *img_dims)).reshape(B, C // 4, n_tokens).transpose(-1, -2)
        x_mamba4 = self.mamba(x4) + self.conv_block4(x4.transpose(-1, -2).reshape(B, C // 4, *img_dims)).reshape(B, C // 4, n_tokens).transpose(-1, -2)

        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=2)
        x_mamba = x_mamba.transpose(-1, -2).reshape(B, self.input_dim, *img_dims)
        # x_mamba = self.squeeze_excite(x_mamba)  # 添加挤压激励操作
        x_mamba = x_mamba.reshape(B, self.input_dim, n_tokens).transpose(-1, -2)
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out

# MDCA module
class StdPool(nn.Module):
    def __init__(self):
        super(StdPool, self).__init__()

    def forward(self, x):
        b, c, _, _ = x.size()
        std = x.view(b, c, -1).std(dim=2, keepdim=True)
        std = std.reshape(b, c, 1, 1)
        return std

class MCAGate(nn.Module):
    def __init__(self, k_size, pool_types=['avg', 'std']):
        """Constructs a MCAGate module.
        Args:
            k_size: kernel size
            pool_types: pooling type. 'avg': average pooling, 'max': max pooling, 'std': standard deviation pooling.
        """
        super(MCAGate, self).__init__()

        self.pools = nn.ModuleList([])
        for pool_type in pool_types:
            if pool_type == 'avg':
                self.pools.append(nn.AdaptiveAvgPool2d(1))
            elif pool_type == 'max':
                self.pools.append(nn.AdaptiveMaxPool2d(1))
            elif pool_type == 'std':
                self.pools.append(StdPool())
            else:
                raise NotImplementedError

        self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), stride=1, padding=(0, (k_size - 1) // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

        self.weight = nn.Parameter(torch.rand(2))

    def forward(self, x):
        feats = [pool(x) for pool in self.pools]

        if len(feats) == 1:
            out = feats[0]
        elif len(feats) == 2:
            weight = torch.sigmoid(self.weight)
            out = 1 / 2 * (feats[0] + feats[1]) + weight[0] * feats[0] + weight[1] * feats[1]
        else:
            assert False, "Feature Extraction Exception!"

        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.conv(out)
        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.sigmoid(out)
        return out * x

class MCALayer(nn.Module):
    def __init__(self, inp, oup=None, kernel_size=3, stride=1):
        super(MCALayer, self).__init__()
        if oup is None:
            oup = inp

        self.out_channels = oup
        self.conv = nn.Conv2d(inp, oup, kernel_size, stride, (kernel_size - 1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        self.act = nn.SiLU()
        self.mca = MCAGate(kernel_size)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.mca(out)
        return out

# DC Block
def get_conv_block(channels):
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True))

class UltraLight_VM_UNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64],
                 split_att='fc', bridge=True, img_size=592, dra_config=None):
        super().__init__()

        self.bridge = bridge

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[5], 3, stride=1, padding=1),
            nn.Conv2d(c_list[5], c_list[3], 3, stride=1, padding=1),
            nn.Conv2d(c_list[3], c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            PVMLayer(input_dim=c_list[0], output_dim=c_list[1]),
            PVMLayer(input_dim=c_list[1], output_dim=c_list[1]),
        )
        self.encoder3 = nn.Sequential(
            PVMLayer(input_dim=c_list[1], output_dim=c_list[2]),
            PVMLayer(input_dim=c_list[2], output_dim=c_list[2]),
        )
        self.encoder4 = nn.Sequential(
            PVMLayer(input_dim=c_list[2], output_dim=c_list[3]),
            PVMLayer(input_dim=c_list[3], output_dim=c_list[3]),
        )
        self.encoder5 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[4]),
            PVMLayer(input_dim=c_list[4], output_dim=c_list[4]),
        )
        self.encoder6 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[5]),
            PVMLayer(input_dim=c_list[5], output_dim=c_list[5]),
        )

        # Initialize MDCA modules
        self.mca_bridge = MCALayer(inp=c_list[5])
        self.mca_bridge1 = MCALayer(inp=c_list[4])
        self.mca_bridge2 = MCALayer(inp=c_list[3])
        self.mca_bridge3 = MCALayer(inp=c_list[2])
        self.mca_bridge4 = MCALayer(inp=c_list[1])
        self.mca_bridge5 = MCALayer(inp=c_list[0])

        # Initialize DRA_S modules
        self.dra_s1 = DRA_S(skip_dim=c_list[4], decoder_dim=c_list[4], img_size=18, config=dra_config)
        self.dra_s2 = DRA_S(skip_dim=c_list[3], decoder_dim=c_list[3], img_size=37, config=dra_config)
        self.dra_s3 = DRA_S(skip_dim=c_list[2], decoder_dim=c_list[2], img_size=74, config=dra_config)
        self.dra_s4 = DRA_S(skip_dim=c_list[1], decoder_dim=c_list[1], img_size=148, config=dra_config)
        self.dra_s5 = DRA_S(skip_dim=c_list[0], decoder_dim=c_list[0], img_size=296, config=dra_config)

        # Initialize DC Block
        self.conv_block_1 = get_conv_block(c_list[4])
        self.conv_block_2 = get_conv_block(c_list[3])
        self.conv_block_3 = get_conv_block(c_list[2])
        self.conv_block_4 = get_conv_block(c_list[1])
        self.conv_block_5 = get_conv_block(c_list[0])

        self.decoder1 = nn.Sequential(
            PVMLayer(input_dim=c_list[5], output_dim=c_list[4]),
            PVMLayer(input_dim=c_list[4], output_dim=c_list[4]),
        )
        self.decoder2 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[3]),
            PVMLayer(input_dim=c_list[3], output_dim=c_list[3]),
        )
        self.decoder3 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[2]),
            PVMLayer(input_dim=c_list[2], output_dim=c_list[2]),
        )
        self.decoder4 = nn.Sequential(
            PVMLayer(input_dim=c_list[2], output_dim=c_list[1]),
            PVMLayer(input_dim=c_list[1], output_dim=c_list[1]),
        )
        self.decoder5 = nn.Sequential(
            PVMLayer(input_dim=c_list[1], output_dim=c_list[0]),
            PVMLayer(input_dim=c_list[0], output_dim=c_list[0]),
        )

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final1 = nn.Conv2d(c_list[0], c_list[1], kernel_size=1)
        self.final = nn.Conv2d(c_list[1], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out

        mca5 = self.mca_bridge1(t5)
        t5 = torch.add(mca5, t5)

        mca4 = self.mca_bridge2(t4)
        t4 = torch.add(mca4, t4)

        mca3 = self.mca_bridge3(t3)
        t3 = torch.add(mca3, t3)

        mca2 = self.mca_bridge4(t2)
        t2 = torch.add(mca2, t2)

        mca1 = self.mca_bridge5(t1)
        t1 = torch.add(mca1, t1)

        out = F.gelu(self.encoder6(out))

        mca = self.mca_bridge(out)
        out = torch.add(mca, out)

        out5 = F.gelu(self.dbn1(self.decoder1(out)))
        out5 = torch.add(out5, t5)
        out5 = self.dra_s1(out5, t5)
        out5 = self.conv_block_1(out5)

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear', align_corners=True)) # 由于尺寸问题，在训练和测试592×592时，修改scale_factor=(2, 2)为size=(37, 37)
        out4 = torch.add(out4, t4)
        out4 = self.dra_s2(out4, t4)
        out4 = self.conv_block_2(out4)

        out3 = F.gelu(
            F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        out3 = torch.add(out3, t3)
        out3 = self.dra_s3(out3, t3)
        out3 = self.conv_block_3(out3)

        out2 = F.gelu(
            F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        out2 = torch.add(t2, out2)
        out2 = self.dra_s4(out2, t2)
        out2 = self.conv_block_4(out2)

        out1 = F.gelu(
            F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        out1 = torch.add(t1, out1)
        out1 = self.dra_s5(out1, t1)
        out1 = self.conv_block_5(out1)

        out0_0 = self.final1(out1)  # b, num_class, H, W
        out0 = F.interpolate(self.final(out0_0), scale_factor=(2, 2), mode='bilinear', align_corners=True)  # b, num_class, H, W
        return torch.sigmoid(out0)

# DGRA module
class DRA_S(nn.Module):
    """ Spatial-wise DRA Module"""
    def __init__(self, skip_dim, decoder_dim, img_size, config):
        super().__init__()
        self.patch_size = img_size // 14
        self.ft_size = img_size
        self.patch_embeddings = nn.Conv2d(in_channels=decoder_dim,
                                          out_channels=decoder_dim,
                                          kernel_size=self.patch_size,
                                          stride=self.patch_size)
        self.conv = nn.Sequential(
            nn.Conv2d(decoder_dim, skip_dim, kernel_size=(1,1), bias=True),
            nn.BatchNorm2d(skip_dim),
            nn.ReLU(inplace=True))
        self.query = nn.Linear(decoder_dim, skip_dim, bias=False)
        self.key = nn.Linear(decoder_dim, skip_dim, bias=False)
        self.value = nn.Linear(decoder_dim, skip_dim, bias=False)
        self.out = nn.Linear(skip_dim, skip_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.psi = nn.InstanceNorm2d(1)
        self.reconstruct = Reconstruct(skip_dim, skip_dim, kernel_size=1,scale_factor=(self.patch_size,self.patch_size))

    def forward(self, decoder, trans):
        decoder_mask = self.conv(decoder)
        decoder_L = self.patch_embeddings(decoder).flatten(2).transpose(-1, -2)
        query = self.query(decoder_L)
        trans = trans.flatten(2).transpose(-1, -2)  # 将 trans 转换为二维输入
        key = self.key(trans).transpose(-1, -2)
        value = self.value(trans)
        sp_similarity_matrix = torch.matmul(query, key)
        sp_similarity_matrix = self.softmax(self.psi(sp_similarity_matrix.unsqueeze(0)).squeeze(0))
        out = torch.matmul(sp_similarity_matrix, value)
        out = self.out(out)
        out = self.reconstruct(out)
        out = F.interpolate(out, size=(decoder_mask.shape[2], decoder_mask.shape[3]), mode='bilinear', align_corners=False)
        out = out * decoder_mask
        return out

class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        if self.scale_factor[0] > 1:
            x = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear', align_corners=False)(x)
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out
