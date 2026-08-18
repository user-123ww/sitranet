import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAdapter(nn.Module):
    def __init__(self, in_channels, out_channels=512, stride=1):
        super(FeatureAdapter, self).__init__()

        if in_channels == out_channels:
            self.model = nn.Identity()

        elif in_channels > out_channels:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 2, 3, stride=stride, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_channels // 2, out_channels, 3, stride=stride, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, x):
        return self.model(x)


class AdaIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def c_norm(self, x, bs, ch, eps=1e-7):
        x_var = x.var(dim=-1) + eps
        x_std = x_var.sqrt().view(bs, ch, 1, 1)
        x_mean = x.mean(dim=-1).view(bs, ch, 1, 1)
        return x_std, x_mean

    def forward(self, x, y):
        assert x.size(0) == y.size(0)
        size = x.size()
        bs, ch = size[:2]
        x_ = x.view(bs, ch, -1)
        y_ = y.reshape(bs, ch, -1)
        x_std, x_mean = self.c_norm(x_, bs, ch, eps=self.eps)
        y_std, y_mean = self.c_norm(y_, bs, ch, eps=self.eps)
        out = ((x - x_mean.expand(size)) / x_std.expand(size)) \
              * y_std.expand(size) + y_mean.expand(size)
        return out


class ReconstructionDecoder(nn.Module):

    def _r_double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def __init__(self, in_channels_a=512, in_channels_c=512, mid_channels=512):
        super().__init__()

        self.adapter_c = FeatureAdapter(in_channels_c, mid_channels)
        self.adapter_a = FeatureAdapter(in_channels_a, mid_channels)

        self.base_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.base_up1 = self._r_double_conv(mid_channels, 256)
        self.base_up2 = self._r_double_conv(256, 128)
        self.base_up3 = self._r_double_conv(128, 64)
        self.base_conv_last = nn.Conv2d(64, 3, 1)
        self.base_activation = nn.Tanh()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dropout = nn.Dropout(p=0.3)

        self.adain3 = AdaIN()
        self.adain2 = AdaIN()
        self.adain1 = AdaIN()

        self.dconv_up3 = self._r_double_conv(mid_channels, 256)
        self.dconv_up2 = self._r_double_conv(256, 128)
        self.dconv_up1 = self._r_double_conv(128, 64)

        self.conv_last = nn.Conv2d(64, 3, 1)
        self.activation = nn.Tanh()

    def forward(self, semantic_feature, trace_feature=None, target_size=None):
        semantic_path = self.adapter_c(semantic_feature)

        base = self.base_upsample(semantic_path)
        base = self.base_up1(base)
        base = self.base_upsample(base)
        base = self.base_up2(base)
        base = self.base_upsample(base)
        base = self.base_up3(base)
        img_base = self.base_activation(self.base_conv_last(base))

        if trace_feature is None:
            return img_base, None

        trace_path = self.adapter_a(trace_feature)

        semantic_path = self.adain3(semantic_path, trace_path)
        semantic_path = self.upsample(semantic_path)
        semantic_path = self.dropout(semantic_path)
        semantic_path = self.dconv_up3(semantic_path)
        trace_path = self.upsample(trace_path)
        trace_path = self.dropout(trace_path)
        trace_path = self.dconv_up3(trace_path)

        semantic_path = self.adain2(semantic_path, trace_path)
        semantic_path = self.upsample(semantic_path)
        semantic_path = self.dropout(semantic_path)
        semantic_path = self.dconv_up2(semantic_path)
        trace_path = self.upsample(trace_path)
        trace_path = self.dropout(trace_path)
        trace_path = self.dconv_up2(trace_path)

        semantic_path = self.adain1(semantic_path, trace_path)
        semantic_path = self.upsample(semantic_path)
        semantic_path = self.dropout(semantic_path)
        semantic_path = self.dconv_up1(semantic_path)

        semantic_path = self.conv_last(semantic_path)
        if target_size is not None:
            semantic_path = F.interpolate(
                semantic_path, size=target_size,
                mode='bilinear', align_corners=False
            )
        img_full = self.activation(semantic_path)

        return img_base, img_full

