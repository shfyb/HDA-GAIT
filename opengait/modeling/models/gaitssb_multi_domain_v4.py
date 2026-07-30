import torch
from einops import rearrange

from .gaitssb_mutil_domain_v3 import GaitSSB_Pretrain_MD_V3, grad_reverse


class GaitSSB_Pretrain_MD_V4(GaitSSB_Pretrain_MD_V3):
    """Unpaired multi-domain pretraining with a shared identity space.

    V4 differs from V3 in two important ways:
    1. identities from both domains participate in one SupCon objective, so
       cross-domain identities are explicit negatives;
    2. the GRL coefficient is linearly interpolated between schedule points.

    The input-adaptive BN and independent q/k sampling are enabled through the
    V4 YAML while remaining opt-in and backward compatible for older configs.
    """

    def __init__(self, cfgs, training=True):
        super().__init__(cfgs, training=training)

    def build_network(self, model_cfg):
        super().build_network(model_cfg)
        # Opt-in for backward compatibility with existing V4 checkpoints and
        # configs whose BN statistics were produced by four single-domain
        # backbone calls.
        self.merge_domain_forward = bool(
            model_cfg.get('merge_domain_forward', False)
        )

    def _get_grl_lambda(self) -> float:
        schedule = self.grl_lambda_schedule
        if not schedule:
            return 1.0

        points = []
        for item in schedule:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                points.append((int(item[0]), float(item[1])))
        if not points:
            return 1.0

        points.sort(key=lambda point: point[0])
        iteration = int(getattr(self, 'iteration', 0))
        if iteration <= points[0][0]:
            return points[0][1]
        if iteration >= points[-1][0]:
            return points[-1][1]

        for (it0, lambda0), (it1, lambda1) in zip(points[:-1], points[1:]):
            if it0 <= iteration < it1:
                ratio = float(iteration - it0) / float(max(it1 - it0, 1))
                return lambda0 + ratio * (lambda1 - lambda0)
        return points[-1][1]

    def forward(self, inputs):
        if not self.training:
            return super().forward(inputs)

        (sils_q, sils_k), labels, domain_types, _, (seqL_q, seqL_k) = inputs
        batch_size = sils_q.size(0)
        a_idx = slice(0, batch_size, 2)
        b_idx = slice(1, batch_size, 2)

        sils_A_q, sils_A_k = sils_q[a_idx].float(), sils_k[a_idx].float()
        sils_B_q, sils_B_k = sils_q[b_idx].float(), sils_k[b_idx].float()

        seqL_A_q = self._slice_seqL(seqL_q, a_idx)
        seqL_A_k = self._slice_seqL(seqL_k, a_idx)
        seqL_B_q = self._slice_seqL(seqL_q, b_idx)
        seqL_B_k = self._slice_seqL(seqL_k, b_idx)

        labels_A, labels_B = labels[a_idx], labels[b_idx]
        domain_A = domain_types[a_idx].detach().long()
        domain_B = domain_types[b_idx].detach().long()

        if self.merge_domain_forward:
            # Encode a domain-balanced batch in each backbone call.  Besides
            # reducing four backbone calls to two, this makes every shared
            # (Sync)BN layer estimate its batch statistics from both domains
            # instead of updating them in the fixed A_q -> A_k -> B_q -> B_k
            # order.
            num_A = sils_A_q.size(0)
            q_sils = torch.cat([sils_A_q, sils_B_q], dim=0)
            k_sils = torch.cat([sils_A_k, sils_B_k], dim=0)
            q_seqL = torch.cat([seqL_A_q, seqL_B_q], dim=-1)
            k_seqL = torch.cat([seqL_A_k, seqL_B_k], dim=-1)
            q_domains = torch.cat([domain_A, domain_B], dim=0)
            k_domains = torch.cat([domain_A, domain_B], dim=0)

            q_pack = self.encoder(
                (q_sils, q_seqL), domain=q_domains, return_features=True
            )
            k_pack = self.encoder(
                (k_sils, k_seqL), domain=k_domains, return_features=True
            )

            A_q_pack = {name: feat[:num_A] for name, feat in q_pack.items()}
            B_q_pack = {name: feat[num_A:] for name, feat in q_pack.items()}
            A_k_pack = {name: feat[:num_A] for name, feat in k_pack.items()}
            B_k_pack = {name: feat[num_A:] for name, feat in k_pack.items()}
        else:
            A_q_pack = self.encoder(
                (sils_A_q, seqL_A_q), domain=domain_A, return_features=True
            )
            A_k_pack = self.encoder(
                (sils_A_k, seqL_A_k), domain=domain_A, return_features=True
            )
            B_q_pack = self.encoder(
                (sils_B_q, seqL_B_q), domain=domain_B, return_features=True
            )
            B_k_pack = self.encoder(
                (sils_B_k, seqL_B_k), domain=domain_B, return_features=True
            )

        z_A_q = self.projector(A_q_pack['high'])
        z_A_k = self.projector(A_k_pack['high'])
        z_B_q = self.projector(B_q_pack['high'])
        z_B_k = self.projector(B_k_pack['high'])

        # All identities share one contrastive space.  Since the two datasets
        # use disjoint global labels, identities from the other domain become
        # valid negatives while repeated crops of the same identity are positives.
        z_q = torch.cat([z_A_q, z_B_q], dim=0)
        z_k = torch.cat([z_A_k, z_B_k], dim=0)
        supcon_features = torch.stack([z_q, z_k], dim=1)
        supcon_labels = torch.cat([labels_A, labels_B], dim=0)

        mid_features_A = torch.cat([A_q_pack['mid'], A_k_pack['mid']], dim=0)
        mid_features_B = torch.cat([B_q_pack['mid'], B_k_pack['mid']], dim=0)

        high_features = torch.cat([z_A_q, z_A_k, z_B_q, z_B_k], dim=0)
        part_features = high_features.permute(0, 2, 1).contiguous()
        part_features = part_features.view(-1, part_features.size(-1))
        domain_logits = self.domain_classifier(
            grad_reverse(part_features, lambd=self._get_grl_lambda())
        )
        domain_labels = torch.cat([domain_A, domain_A, domain_B, domain_B], dim=0)
        domain_labels = domain_labels.unsqueeze(1).repeat(1, self.p).reshape(-1)

        return {
            'training_feat': {
                'supcon_all': {
                    'features': supcon_features,
                    'labels': supcon_labels,
                },
                'mid_mmd_loss': {
                    'logits': (mid_features_A, mid_features_B),
                    'labels': None,
                },
                'adv_domain_high': {
                    'logits': domain_logits,
                    'labels': domain_labels,
                },
            },
            'visual_summary': {
                'image/encoder_A_q': rearrange(sils_A_q, 'b c t h w -> (b t) c h w'),
                'image/encoder_B_q': rearrange(sils_B_q, 'b c t h w -> (b t) c h w'),
            },
            'inference_feat': None,
        }
