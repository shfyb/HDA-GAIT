'''
Modifed fromhttps://github.com/BNU-IVC/FastPoseGait/blob/main/fastposegait/modeling/losses/supconloss.py
'''

import torch.nn as nn
import torch
from .base import BaseLoss, gather_and_scale_wrapper


class SupConLoss_Re(BaseLoss):
    def __init__(self, temperature=0.01):
        super(SupConLoss_Re, self).__init__()
        self.train_loss = SupConLoss(temperature=temperature)

    @gather_and_scale_wrapper
    def forward(self, features, labels=None, mask=None):
        loss = self.train_loss(features, labels)
        self.info.update({
            'loss': loss.detach().clone()})
        return loss, self.info


class SupConLoss_Lp(BaseLoss):
    def __init__(self, temperature=0.01):
        super(SupConLoss_Lp, self).__init__()
        self.train_loss = SupConLoss(
            temperature=temperature, base_temperature=temperature, reduce_zero=True, p=2)

    @gather_and_scale_wrapper
    def forward(self, features, labels=None, mask=None):
        # Accept both:
        # - single-view embeddings: [bsz, dim] (or [bsz, ...])  -> unsqueeze to [bsz, 1, ...]
        # - multi-view embeddings:  [bsz, n_views, ...]         -> keep as-is (2-view SupCon etc.)
        #
        # This makes it compatible with models that already output 2-view features, e.g. stack([z_q, z_k], dim=1).
        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(features)
        if features.dim() >= 3:
            loss = self.train_loss(features, labels)
        else:
            loss = self.train_loss(features.unsqueeze(1), labels)
        self.info.update({
            'loss': loss.detach().clone(),
            'valid_anchor_ratio':
                self.train_loss.last_valid_anchor_ratio.clone(),
            'positive_count_mean':
                self.train_loss.last_positive_count_mean.clone(),
            'positive_count_min':
                self.train_loss.last_positive_count_min.clone(),
        })
        return loss, self.info


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.01, contrast_mode='all',
                 base_temperature=0.07, reduce_zero=False, p=None):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.reduce_zero = reduce_zero
        self.p = p

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # Compute the similarity matrix.
        if self.p is None:
            mat = torch.matmul(
                anchor_feature, contrast_feature.T)
        else:
            anchor_feature = torch.nn.functional.normalize(
                anchor_feature, p=self.p, dim=1)
            contrast_feature = torch.nn.functional.normalize(
                contrast_feature, p=self.p, dim=1)
            mat = -torch.cdist(
                anchor_feature, contrast_feature, p=self.p)
        logits = mat / self.temperature

        # Tile the identity mask to cover all anchor/contrast views.
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask, dtype=torch.bool),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            False,
        )
        positive_mask = mask.bool() & logits_mask
        positive_count = positive_mask.sum(dim=1)

        # logsumexp is stable even when the positive pair is much more similar
        # than every negative.  The previous exp/sum/log implementation could
        # round the probability to exactly one and produce a zero anchor loss.
        masked_logits = logits.masked_fill(~logits_mask, float('-inf'))
        log_denominator = torch.logsumexp(
            masked_logits, dim=1, keepdim=True
        )
        log_prob = logits - log_denominator

        # torch.where avoids the undefined 0 * (-inf) operation.  An anchor is
        # valid based on whether it has a positive after self-removal, not on
        # whether its numerically computed loss happens to be greater than 0.
        positive_log_prob = torch.where(
            positive_mask, log_prob, torch.zeros_like(log_prob)
        )
        mean_log_prob_pos = positive_log_prob.sum(dim=1) / \
            positive_count.clamp_min(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        valid_anchor = positive_count > 0

        self.last_valid_anchor_ratio = valid_anchor.float().mean().detach()
        self.last_positive_count_mean = positive_count.float().mean().detach()
        self.last_positive_count_min = positive_count.min().detach()

        if valid_anchor.any():
            valid_loss = loss[valid_anchor]
            if not torch.isfinite(valid_loss).all():
                raise FloatingPointError(
                    "SupCon produced non-finite loss for valid anchors: "
                    f"finite={torch.isfinite(valid_loss).sum().item()}/"
                    f"{valid_loss.numel()}, "
                    f"positive_count_min={positive_count[valid_anchor].min().item()}, "
                    f"positive_count_max={positive_count[valid_anchor].max().item()}"
                )
        else:
            # Keep a valid zero connected to the graph.  This can occur for a
            # single-view batch with globally unique labels, but must never
            # become empty.mean() -> NaN.
            return features.sum() * 0.0

        if self.reduce_zero:
            loss = loss[valid_anchor]
        else:
            loss = torch.where(valid_anchor, loss, torch.zeros_like(loss))

        return loss.mean()
