import torch
import torch.nn as nn


class FeatureDecorrelationLoss(nn.Module):
    @staticmethod
    def _pearson_loss(trace_features, semantic_features):
        trace_mean = torch.mean(trace_features, dim=1, keepdim=True)
        semantic_mean = torch.mean(semantic_features, dim=1, keepdim=True)

        centered_trace = trace_features - trace_mean
        centered_semantic = semantic_features - semantic_mean
        covariance = torch.mean(centered_trace * centered_semantic, dim=1)

        trace_std = torch.sqrt(torch.mean(centered_trace ** 2, dim=1) + 1e-8)
        semantic_std = torch.sqrt(
            torch.mean(centered_semantic ** 2, dim=1) + 1e-8
        )
        correlation = covariance / (trace_std * semantic_std + 1e-8)
        return torch.mean(torch.abs(correlation))

    def forward(self, trace_features, semantic_features=None):
        if semantic_features is None or trace_features.shape != semantic_features.shape:
            return torch.tensor(
                0.0, device=trace_features.device, dtype=trace_features.dtype
            )

        trace_flat = trace_features.reshape(trace_features.size(0), -1)
        semantic_flat = semantic_features.reshape(semantic_features.size(0), -1)
        return self._pearson_loss(trace_flat, semantic_flat)
