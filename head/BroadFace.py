import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BroadFace(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        scale_factor=64.0,
        margin=0.50,
        queue_size=10000,
        compensate=False,
    ):
        super(BroadFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.criterion = nn.CrossEntropyLoss(reduction="none")

        self.margin = margin
        self.scale_factor = scale_factor

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        feature_mb = torch.zeros(0, in_features)
        label_mb = torch.zeros(0, dtype=torch.int64)
        proxy_mb = torch.zeros(0, in_features)
        self.register_buffer("feature_mb", feature_mb)
        self.register_buffer("label_mb", label_mb)
        self.register_buffer("proxy_mb", proxy_mb)

        self.queue_size = queue_size
        self.compensate = compensate

    def compute_arcface(self, x, y, w):
        cosine = F.linear(F.normalize(x), F.normalize(w))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, y.view(-1, 1).long(), 1)

        logit = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logit *= self.scale_factor

        ce_loss = self.criterion(logit, y)
        return ce_loss.mean()

    def forward(self, input, label):
        # input is not l2 normalized
        weight_now = self.weight.data[self.label_mb]
        delta_weight = weight_now - self.proxy_mb

        if self.compensate:
            update_feature_mb = (
                self.feature_mb
                + (
                    self.feature_mb.norm(p=2, dim=1, keepdim=True)
                    / self.proxy_mb.norm(p=2, dim=1, keepdim=True)
                )
                * delta_weight
            )
        else:
            update_feature_mb = self.feature_mb

        large_input = torch.cat([update_feature_mb, input.data], dim=0)
        large_label = torch.cat([self.label_mb, label], dim=0)

        batch_loss = self.compute_arcface(input, label, self.weight.data)
        broad_loss = self.compute_arcface(large_input, large_label, self.weight)

        loss = batch_loss + broad_loss

        return loss