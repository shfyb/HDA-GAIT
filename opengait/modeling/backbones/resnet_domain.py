import math

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

from ..modules import BasicConv2d


block_map = {
    'BasicBlock': BasicBlock,
    'Bottleneck': Bottleneck,
}


class InputAdaptiveBN2d(nn.Module):
    """输入自适应 BN：使用路由网络融合多个 BN expert。"""

    def __init__(self, num_features, num_experts=2, reduction=16, eps=1e-5, momentum=0.1):
        super().__init__()
        hidden_dim = max(num_features // reduction, 4)
        self.num_experts = num_experts
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
            for _ in range(num_experts)
        ])
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_experts, kernel_size=1, bias=True),
        )

    def forward(self, x):
        weights = torch.softmax(self.router(x), dim=1)
        expert_outs = [bn(x) for bn in self.bns]
        stacked = torch.stack(expert_outs, dim=1)
        return torch.sum(stacked * weights.unsqueeze(2), dim=1)


class InputAdaptiveINSyncBN2d(nn.Module):
    """Input-adaptive fusion of instance and shared batch normalization.

    Both normalization branches process all channels.  A lightweight router
    predicts a per-sample, per-channel gate from the unnormalized input:

        y = gamma * (alpha(x) * IN(x) + (1 - alpha(x)) * BN(x)) + beta.

    ``bn_norm`` is declared as BatchNorm2d and is converted recursively to
    SyncBatchNorm by the training engine when ``sync_BN`` is enabled.
    """

    def __init__(
        self,
        num_features,
        reduction=16,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        init_alpha=0.25,
    ):
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        if reduction <= 0:
            raise ValueError("reduction must be positive")
        if not 0.0 < init_alpha < 1.0:
            raise ValueError(
                f"init_alpha must be strictly between 0 and 1, got {init_alpha}"
            )

        hidden_dim = max(num_features // reduction, 4)
        self.in_norm = nn.InstanceNorm2d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=False,
            track_running_stats=False,
        )
        self.bn_norm = nn.BatchNorm2d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=False,
        )
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_features, kernel_size=1, bias=True),
        )

        # Start from a stable, BN-dominant mixture.  Zero weights make the
        # initial gate input-independent; it becomes adaptive during training.
        nn.init.zeros_(self.router[-1].weight)
        nn.init.constant_(
            self.router[-1].bias,
            math.log(init_alpha / (1.0 - init_alpha)),
        )

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(
                f"Expected a 4D tensor [N, C, H, W], got {tuple(x.shape)}"
            )

        alpha = torch.sigmoid(self.router(x))
        out = alpha * self.in_norm(x) + (1.0 - alpha) * self.bn_norm(x)
        if self.weight is not None:
            out = (
                out * self.weight.view(1, -1, 1, 1)
                + self.bias.view(1, -1, 1, 1)
            )
        return out



class StatINBNNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps

        # BN：只用 running statistics（跨 batch / 跨 domain）
        self.bn = nn.BatchNorm2d(channels, affine=False)

        # 可学习 alpha（标量）
        self.alpha = nn.Parameter(
            torch.ones(1, channels, 1, 1) * 0.5
        )

    def forward(self, x):
        """
        x: [N, C, H, W]
        """
        _ = self.bn(x)
        # ---------- BN statistics ----------
        mean_bn = self.bn.running_mean.view(1, -1, 1, 1)
        var_bn  = self.bn.running_var.view(1, -1, 1, 1)

        # ---------- IN statistics ----------
        # per-sample, per-channel
        mean_in = x.mean(dim=[2, 3], keepdim=True)
        var_in  = x.var(dim=[2, 3], keepdim=True, unbiased=False)

        # ---------- statistics fusion ----------
        alpha = torch.sigmoid(self.alpha)

        mean = alpha * mean_in + (1 - alpha) * mean_bn
        var  = alpha * var_in  + (1 - alpha) * var_bn

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat



class ResNet9_domain(ResNet):
    def __init__(
        self,
        block,
        channels=[32, 64, 128, 256],
        in_channel=1,
        layers=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        maxpool=True,
        num_bn_experts=2,
        bn_reduction=16,
        use_adaptive_bn=False,
        stem_norm=None,
        adaptive_in_init_alpha=0.25,
    ):
        if block in block_map:
            block = block_map[block]
        else:
            raise ValueError("Error type for -block-Cfg-, supported: 'BasicBlock' or 'Bottleneck'.")

        self.maxpool_flag = maxpool
        super(ResNet9_domain, self).__init__(block, layers)

        self.fc = None
        self.inplanes = channels[0]
        
        self.conv1 = BasicConv2d(in_channel, self.inplanes, 3, 1, 1)
        '''
        self.bn1 = InputAdaptiveBN2d(
            channels[0],
            num_experts=num_bn_experts,
            reduction=bn_reduction,
        )
        '''
        # ``use_adaptive_bn`` is kept for backward compatibility.  New
        # configurations should select the stem behavior explicitly.
        if stem_norm is None:
            stem_norm = 'adaptive_bn' if use_adaptive_bn else 'bn'

        if stem_norm == 'adaptive_in_syncbn':
            self.bn1 = InputAdaptiveINSyncBN2d(
                self.inplanes,
                reduction=bn_reduction,
                init_alpha=adaptive_in_init_alpha,
            )
        elif stem_norm == 'adaptive_bn':
            self.bn1 = InputAdaptiveBN2d(
                self.inplanes,
                num_experts=num_bn_experts,
                reduction=bn_reduction,
            )
        elif stem_norm == 'bn':
            self.bn1 = nn.BatchNorm2d(self.inplanes)
        else:
            raise ValueError(
                f"Unsupported stem_norm '{stem_norm}'. Expected one of "
                "['bn', 'adaptive_bn', 'adaptive_in_syncbn']."
            )
        #self.bn1 = StatINBNNorm(self.inplanes)

        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=strides[0], dilate=False)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=strides[1], dilate=False)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=strides[2], dilate=False)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=strides[3], dilate=False)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        if blocks >= 1:
            return super()._make_layer(block, planes, blocks, stride=stride, dilate=dilate)

        def identity_layer(x):
            return x

        return identity_layer

    def forward(self, x, domain=None, return_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        stem = x

        if self.maxpool_flag:
            x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        if return_features:
            return {
                'stem': stem,
                'layer1': x1,
                'layer2': x2,
                'layer3': x3,
                'layer4': x4,
            }
        return x4
