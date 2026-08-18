import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.xception_encoder import XceptionEncoder
from loss import FeatureDecorrelationLoss
from subnets.reconstruction_decoder import ReconstructionDecoder
from subnets.trace_feature_fusion import TraceFeatureFusion

class SITraNet(nn.Module):
    def __init__(self, backbone='Xception', dropout_rate=0.5,
                 num_prototypes=8, use_multi_prototype=True,
                 classification_mode='hybrid'):

        super(SITraNet, self).__init__()

        self.feature_dim = 256
        self.in_channels = 0
        self.in_channels_a = 0
        self.in_channels_c = 0

        if backbone != 'Xception':
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.encoder_a = XceptionEncoder()
        self.encoder_c = XceptionEncoder()
        self.in_channels = 512

        if self.in_channels_a == 0:
            self.in_channels_a = self.in_channels
            self.in_channels_c = self.in_channels

        self.con_gan = ReconstructionDecoder(
            in_channels_a=self.in_channels_a,
            in_channels_c=self.in_channels_c,
        )

        self.adjust_channel = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
        )

        self.fusion_module = TraceFeatureFusion(
            in_channels=self.in_channels_a,
            reduce_dim=256,
            dropout_rate=dropout_rate,
        )

        self.use_multi_prototype = use_multi_prototype
        self.num_prototypes = num_prototypes if use_multi_prototype else 1
        self.prototype_dim = 128

        self.prototypes = nn.Parameter(
            torch.randn(self.num_prototypes, self.prototype_dim) * 0.01
        )

        self.register_buffer('prototypes_initialized', torch.tensor(False))

        self.classification_mode = classification_mode

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 2)
        )

        if classification_mode in ['distance', 'hybrid']:
            self.distance_to_logits = nn.Sequential(
                nn.Linear(self.num_prototypes, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 2)
            )


    def initialize_prototypes_from_data(self, cover_features):
        random_protos = torch.randn(self.num_prototypes, self.prototype_dim, device=cover_features.device)
        self.prototypes.data = F.normalize(random_protos, p=2, dim=1)
        self.prototypes_initialized.fill_(True)


    def compute_prototype_distances(self, features, k_nearest=3, prototypes=None):
        if prototypes is None:
            prototypes = self.prototypes

        features = F.normalize(features, p=2, dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=1)

        feature_norm = (features ** 2).sum(dim=1, keepdim=True)
        proto_norm = (prototypes ** 2).sum(dim=1, keepdim=True).t()
        cross_term = torch.mm(features, prototypes.t())

        all_distances = feature_norm + proto_norm - 2 * cross_term
        all_distances = torch.sqrt(all_distances + 1e-8)

        if self.use_multi_prototype and k_nearest > 1:
            topk_distances, _ = torch.topk(all_distances, k=min(k_nearest, self.num_prototypes),
                                           largest=False, dim=1)
            distances = topk_distances.mean(dim=1)
        else:
            distances = all_distances.min(dim=1)[0]

        return distances, all_distances

    def forward(self, x):
        original_blocks = self._split_image_blocks(x)

        trace_view1 = self.encoder_a(original_blocks[0])
        trace_view2 = self.encoder_a(original_blocks[1])

        semantic_view1 = None
        semantic_view2 = None
        if self.training:
            semantic_view1 = self.encoder_c(original_blocks[0])
            semantic_view2 = self.encoder_c(original_blocks[1])

        fused_trace = self.fusion_module(trace_view1, trace_view2)
        normalized_trace = self.adjust_channel(fused_trace).squeeze(-1).squeeze(-1)
        normalized_trace = F.normalize(normalized_trace, p=2, dim=1)

        raw_features = None
        reconstruction_imgs = None

        if self.training:
            raw_features = {
                'trace_view1': trace_view1,
                'semantic_view1': semantic_view1,
                'trace_view2': trace_view2,
                'semantic_view2': semantic_view2,
            }

            block_size = (original_blocks[0].shape[2], original_blocks[0].shape[3])

            rec_base_1, self_recon_1 = self.con_gan(
                trace_feature=trace_view1,
                semantic_feature=semantic_view1,
                target_size=block_size,
            )
            rec_base_2, self_recon_2 = self.con_gan(
                trace_feature=trace_view2,
                semantic_feature=semantic_view2,
                target_size=block_size,
            )

            _, cross_recon_1 = self.con_gan(
                trace_feature=trace_view2,
                semantic_feature=semantic_view1,
                target_size=block_size,
            )
            _, cross_recon_2 = self.con_gan(
                trace_feature=trace_view1,
                semantic_feature=semantic_view2,
                target_size=block_size,
            )

            reconstruction_imgs = (
                self_recon_1, self_recon_2,
                cross_recon_1, cross_recon_2,
                rec_base_1, rec_base_2,
            )

            raw_features['trace_from_cross_1'] = self.encoder_a(cross_recon_1)
            raw_features['semantic_from_cross_1'] = self.encoder_c(cross_recon_1)
            raw_features['trace_from_cross_2'] = self.encoder_a(cross_recon_2)
            raw_features['semantic_from_cross_2'] = self.encoder_c(cross_recon_2)

        sample_feature = normalized_trace
        proto_dist, all_proto_dists = self.compute_prototype_distances(
            sample_feature, k_nearest=3
        )

        proto_dist_sg, _ = self.compute_prototype_distances(
            sample_feature, k_nearest=3, prototypes=self.prototypes.detach()
        )

        proto_distances = {
            'mean_distance': proto_dist,
            'mean_distance_sg': proto_dist_sg,
            'all_distances': all_proto_dists,
            'sample_feature': sample_feature,
            'prototypes': self.prototypes,
        }

        if self.classification_mode == 'classifier':
            output = self.classifier(fused_trace)

        elif self.classification_mode == 'distance':
            output = self.distance_to_logits(all_proto_dists)

        elif self.classification_mode == 'hybrid':
            logits_cls = self.classifier(fused_trace)

            logits_dist = self.distance_to_logits(all_proto_dists)

            output = 0.5 * logits_cls + 0.5 * logits_dist

        else:
            raise ValueError(f"Unknown classification_mode: {self.classification_mode}")

        return output, reconstruction_imgs, original_blocks, raw_features, proto_distances

    def _split_image_blocks(self, x):
        mid_w = x.size(3) // 2
        block1 = x[:, :, :, :mid_w]
        block2 = x[:, :, :, mid_w:]
        return block1, block2



class SITraNetLoss(nn.Module):
    def __init__(self, gamma=1.5, delta=0.3, margin=1.5,
                 lambda_proto=1.5, cover_tolerance=0.1):
        super(SITraNetLoss, self).__init__()

        self.gamma = gamma
        self.delta = delta
        self.margin = margin
        self.lambda_proto = lambda_proto
        self.cover_tolerance = cover_tolerance

        self.cross_entropy = nn.CrossEntropyLoss()
        self.rec_loss = nn.L1Loss()
        self.ps_loss = FeatureDecorrelationLoss()

    def compute_prototype_loss(self, proto_distances, labels,
                               use_cover_only=False):
        distances = proto_distances['mean_distance']
        distances_sg = proto_distances.get('mean_distance_sg', distances)

        cover_mask = labels == 0
        stego_mask = labels == 1

        if cover_mask.any():
            loss_cover = F.relu(
                distances[cover_mask] - self.cover_tolerance
            ).pow(2).mean()
        else:
            loss_cover = distances.sum() * 0.0

        if use_cover_only:
            loss_proto = loss_cover

        else:
            if stego_mask.any():
                loss_stego = F.softplus(
                    self.margin - distances_sg[stego_mask]
                ).mean()
            else:
                loss_stego = distances_sg.sum() * 0.0
            loss_proto = loss_cover + loss_stego

        return loss_proto

    def forward(self, outputs, labels, reconstruction_imgs,
                original_blocks, raw_features=None, proto_distances=None,
                use_cover_only=False):
        loss_cls = self.cross_entropy(outputs, labels)

        loss_rec = torch.tensor(0.0, device=outputs.device)
        loss_rec1 = torch.tensor(0.0, device=outputs.device)
        loss_rec2 = torch.tensor(0.0, device=outputs.device)
        loss_rec3 = torch.tensor(0.0, device=outputs.device)
        loss_ps = torch.tensor(0.0, device=outputs.device)
        loss_proto = torch.tensor(0.0, device=outputs.device)

        if reconstruction_imgs is not None:
            self_loss_reconstruction_1 = self.rec_loss(original_blocks[0], reconstruction_imgs[0])
            self_loss_reconstruction_2 = self.rec_loss(original_blocks[1], reconstruction_imgs[1])

            base_size_1 = reconstruction_imgs[4].shape[2:]
            base_size_2 = reconstruction_imgs[5].shape[2:]
            target_base_1 = F.interpolate(original_blocks[0], size=base_size_1,
                                          mode='bilinear', align_corners=False)
            target_base_2 = F.interpolate(original_blocks[1], size=base_size_2,
                                          mode='bilinear', align_corners=False)

            rec_base_1 = self.rec_loss(target_base_1, reconstruction_imgs[4])
            rec_base_2 = self.rec_loss(target_base_2, reconstruction_imgs[5])

            loss_a_cross_1 = self.rec_loss(
                raw_features['trace_from_cross_1'], raw_features['trace_view2']
            )
            loss_c_cross_1 = self.rec_loss(
                raw_features['semantic_from_cross_1'], raw_features['semantic_view1']
            )
            loss_a_cross_2 = self.rec_loss(
                raw_features['trace_from_cross_2'], raw_features['trace_view1']
            )
            loss_c_cross_2 = self.rec_loss(
                raw_features['semantic_from_cross_2'], raw_features['semantic_view2']
            )

            loss_rec2 = loss_a_cross_1 + loss_c_cross_1 + loss_a_cross_2 + loss_c_cross_2
            loss_rec1 = self_loss_reconstruction_1 + self_loss_reconstruction_2
            loss_rec3 = rec_base_1 + rec_base_2

            loss_rec = 1.00 * loss_rec1 + 0.1 * loss_rec2 + 0.1 * loss_rec3

        if raw_features is not None:
            trace_view1 = raw_features['trace_view1']
            semantic_view1 = raw_features['semantic_view1']
            trace_view2 = raw_features['trace_view2']
            semantic_view2 = raw_features['semantic_view2']
            if trace_view1.shape == semantic_view1.shape:
                loss_ps_1 = self.ps_loss(trace_view1, semantic_view1)
                loss_ps_2 = self.ps_loss(trace_view2, semantic_view2)
            else:
                loss_ps_1 = self.ps_loss(trace_view1)
                loss_ps_2 = self.ps_loss(trace_view2)

            loss_ps = 0.5 * (loss_ps_1 + loss_ps_2)

        if proto_distances is not None:
            loss_proto = self.compute_prototype_loss(
                proto_distances, labels,
                use_cover_only=use_cover_only
            )

        total_loss = (loss_cls +
                      self.gamma * loss_rec +
                      self.delta * loss_ps +
                      self.lambda_proto * loss_proto)

        loss_dict = {
            'total_loss': total_loss.item(),
            'loss_cls': loss_cls.item(),
            'loss_rec': loss_rec.item(),
            'loss_ps': loss_ps.item(),
            'loss_proto': loss_proto.item(),
        }

        return total_loss, loss_dict
