import torch
import torch.nn as nn
import torch.nn.functional as F


class StatisticalPooling(nn.Module):
    def forward(self, x):
        mean = torch.mean(x, dim=[2, 3])
        std = torch.std(x, dim=[2, 3]) + 1e-6
        return torch.cat([mean, std], dim=1)


class LinearBN(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        return self.dropout(self.act(x))


class TraceFeatureFusion(nn.Module):
    def __init__(self, in_channels=512, reduce_dim=256, dropout_rate=0.4):
        super().__init__()

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, reduce_dim, 1, bias=False),
            nn.BatchNorm2d(reduce_dim),
        )

        self.stat_pool = StatisticalPooling()

        fusion_dim = (reduce_dim * 2) * 3 + 1

        self.classifier = nn.Sequential(
            LinearBN(fusion_dim, 1024, dropout=dropout_rate),
            LinearBN(1024, 512),
            LinearBN(512, 256),
        )

    def forward(self, x1, x2):
        f1 = self.reduce_conv(x1)
        f2 = self.reduce_conv(x2)

        v1 = self.stat_pool(f1)
        v2 = self.stat_pool(f2)

        feat_max = torch.max(v1, v2)
        feat_diff = torch.abs(v1 - v2)
        feat_prod = v1 * v2
        cosine_sim = F.cosine_similarity(v1, v2, dim=1).unsqueeze(1)

        fused_vector = torch.cat([feat_max, feat_diff, feat_prod, cosine_sim], dim=1)

        output = self.classifier(fused_vector)

        return output
