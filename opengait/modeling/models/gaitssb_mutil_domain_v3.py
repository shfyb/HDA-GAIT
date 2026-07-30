import torch
import torch.nn as nn
import torch.nn.functional as F
from data.transform import get_transform
from einops import rearrange
from torch.autograd import Function

from ..base_model import BaseModel
from ..modules import (
    HorizontalPoolingPyramid,
    PackSequenceWrapper,
    ParallelBN1d,
    SeparateFCs,
    SetBlockWrapper,
)


class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradientReversalLayer.apply(x, lambd)


class GaitSSB_Pretrain_MD_V3(BaseModel):
    """多域分层预训练模型：浅层自适应 BN，中层 CORAL，高层 SupCon 与 GRL。"""

    def __init__(self, cfgs, training=True):
        super(GaitSSB_Pretrain_MD_V3, self).__init__(cfgs, training=training)

    def build_network(self, model_cfg):
        self.p = model_cfg['parts_num']

        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid([16, 8, 4, 2, 1])

        out_channels = model_cfg['backbone_cfg']['channels'][-1]
        hidden_dim = out_channels

        self.projector = nn.Sequential(
            SeparateFCs(self.p, out_channels, hidden_dim),
            ParallelBN1d(self.p, hidden_dim),
            nn.ReLU(inplace=True),
            SeparateFCs(self.p, hidden_dim, out_channels),
            ParallelBN1d(self.p, out_channels),
        )

        # 保留 predictor 仅为兼容旧 checkpoint，当前前向不再使用 D 分支。
        self.predictor = nn.Sequential(
            SeparateFCs(self.p, out_channels, hidden_dim),
            ParallelBN1d(self.p, hidden_dim),
            nn.ReLU(inplace=True),
            SeparateFCs(self.p, hidden_dim, out_channels),
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(out_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )
        # GRL lambda warmup schedule (piecewise linear).
        # Example:
        #   grl_lambda_schedule:
        #     - [0, 0.0]
        #     - [60000, 0.5]
        #     - [100000, 1.0]
        self.grl_lambda_schedule = model_cfg.get('grl_lambda_schedule', None)

    def _get_grl_lambda(self) -> float:
        """Compute GRL lambda from self.iteration using a piecewise-constant schedule.

        Given points: [(it0, l0), (it1, l1), ...] (sorted by iteration),
        this returns:
        - l0 for it <= it0
        - l_k for it_k <= it < it_{k+1}
        - last lambda for it >= last_it
        """
        sched = self.grl_lambda_schedule
        if not sched:
            return 1.0

        points = []
        for item in sched:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                it, lam = item
                points.append((int(it), float(lam)))
        if len(points) == 0:
            return 1.0
        points.sort(key=lambda x: x[0])

        it_now = int(getattr(self, 'iteration', 0))
        if it_now <= points[0][0]:
            return float(points[0][1])
        if it_now >= points[-1][0]:
            return float(points[-1][1])

        for (it0, l0), (it1, _) in zip(points[:-1], points[1:]):
            if it0 <= it_now < it1:
                return float(l0)
        return float(points[-1][1])


    def inputs_pretreament(self, inputs):
        """将一个 batch 的多域 pair 展开为模型输入。"""
        if not self.training:
            return super().inputs_pretreament(inputs)

        pairs = inputs
        trf_cfgs = self.engine_cfg['transform']
        seq_trfs = get_transform(trf_cfgs)

        if not isinstance(seq_trfs, list):
            seq_trfs = [seq_trfs, seq_trfs]
        elif len(seq_trfs) == 1:
            seq_trfs = [seq_trfs[0], seq_trfs[0]]

        sils_q_list, sils_k_list = [], []
        labs_list, domain_list = [], []
        seqL_q_list, seqL_k_list = [], []

        for pair in pairs:
            for side, trf in zip(['A', 'B'], seq_trfs):
                q = pair[f'{side}_q']
                k = pair[f'{side}_k']

                q_seq = trf(q[0]).copy()
                k_seq = trf(k[0]).copy()

                q_tensor = torch.tensor(q_seq, dtype=torch.float32, device='cuda').unsqueeze(0).unsqueeze(1)
                k_tensor = torch.tensor(k_seq, dtype=torch.float32, device='cuda').unsqueeze(0).unsqueeze(1)

                sils_q_list.append(q_tensor)
                sils_k_list.append(k_tensor)
                labs_list.append(pair[f'label{side}'])
                domain_list.append(pair[f'domain{side}'])
                seqL_q_list.append(q_tensor.size(2))
                seqL_k_list.append(k_tensor.size(2))

        sils_q = torch.cat(sils_q_list, dim=0)
        sils_k = torch.cat(sils_k_list, dim=0)
        labs = torch.tensor(labs_list, dtype=torch.long, device='cuda')
        domain_typs = torch.tensor(domain_list, dtype=torch.long, device='cuda')
        seqL_q = torch.tensor(seqL_q_list, dtype=torch.int64, device='cuda').unsqueeze(0)
        seqL_k = torch.tensor(seqL_k_list, dtype=torch.int64, device='cuda').unsqueeze(0)

        return (sils_q, sils_k), labs, domain_typs, domain_typs, (seqL_q, seqL_k)

    def _format_seqL(self, seqL, device, seq_len=None, batch_size=None):
        if seqL is None:
            if seq_len is None or batch_size is None:
                return None
            return torch.full((1, batch_size), seq_len, dtype=torch.long, device=device)

        if isinstance(seqL, int):
            if batch_size is None:
                batch_size = 1
            return torch.full((1, batch_size), seqL, dtype=torch.long, device=device)

        if not isinstance(seqL, torch.Tensor):
            seqL = torch.tensor(seqL, dtype=torch.long, device=device)
        else:
            seqL = seqL.to(dtype=torch.long, device=device)

        if seqL.dim() == 0:
            if batch_size is None:
                batch_size = 1
            return seqL.view(1, 1).repeat(1, batch_size)
        if seqL.dim() == 1:
            return seqL.unsqueeze(0)
        return seqL

    def _slice_seqL(self, seqL, idx):
        if seqL is None:
            return None
        if seqL.dim() == 1:
            return seqL[idx].unsqueeze(0)
        return seqL[:, idx]

    def _resolve_pool_seqL(self, seqL, seqs, dim=2):
        """
        PackSequenceWrapper 仅适用于“沿时间维拼接后的序列”。
        当前模型大多数情况下输入是规则 batch 张量 [B, C, T, H, W]，
        这时传入逐样本 seqL 会导致窄化越界，因此回退到整段时间池化。
        """
        if seqL is None:
            return None

        seqL = self._format_seqL(seqL, seqs.device)
        temporal_size = seqs.size(dim)
        total_seq_len = int(seqL.sum().item())
        if total_seq_len != temporal_size:
            return None
        return seqL

    def _run_backbone(self, sils, domain=None, return_features=False):
        backbone = self.Backbone.forward_block if isinstance(self.Backbone, SetBlockWrapper) else self.Backbone
        n, c, s, h, w = sils.size()
        flat = sils.transpose(1, 2).reshape(-1, c, h, w)
        outs = backbone(flat, domain=domain, return_features=return_features)

        if return_features:
            reshaped = {}
            for key, value in outs.items():
                reshaped[key] = value.reshape(n, s, *value.shape[1:]).transpose(1, 2).contiguous()
            return reshaped

        return outs.reshape(n, s, *outs.shape[1:]).transpose(1, 2).contiguous()

    def encoder(self, inputs, domain=None, return_features=False):
        """提取高层 HPP 特征；可选同时返回中层时空特征。"""
        sils, seqL = inputs
        batch_size = sils.size(0)
        seqL = self._format_seqL(seqL, sils.device, seq_len=sils.size(2), batch_size=batch_size)

        if return_features:
            stage_feats = self._run_backbone(sils, domain=domain, return_features=True)
            pool_seqL_mid = self._resolve_pool_seqL(seqL, stage_feats['layer2'], dim=2)
            pool_seqL_high = self._resolve_pool_seqL(seqL, stage_feats['layer4'], dim=2)
            mid_map = self.TP(stage_feats['layer2'], pool_seqL_mid, options={'dim': 2})[0]
            high_map = self.TP(stage_feats['layer4'], pool_seqL_high, options={'dim': 2})[0]
            return {
                'mid': mid_map.mean(dim=(-1, -2)),
                'high': self.HPP(high_map),
            }

        outs = self._run_backbone(sils, domain=domain, return_features=False)
        pool_seqL = self._resolve_pool_seqL(seqL, outs, dim=2)
        outs = self.TP(outs, pool_seqL, options={'dim': 2})[0]
        return self.HPP(outs)

    def forward(self, inputs):
        if self.training:
            (sils_q, sils_k), labs, domain_typs, _, (seqL_q, seqL_k) = inputs
            batch_size = sils_q.size(0)
            a_idx = slice(0, batch_size, 2)
            b_idx = slice(1, batch_size, 2)

            sils_A_q, sils_A_k = sils_q[a_idx].float(), sils_k[a_idx].float()
            sils_B_q, sils_B_k = sils_q[b_idx].float(), sils_k[b_idx].float()

            seqL_A_q = self._slice_seqL(seqL_q, a_idx)
            seqL_A_k = self._slice_seqL(seqL_k, a_idx)
            seqL_B_q = self._slice_seqL(seqL_q, b_idx)
            seqL_B_k = self._slice_seqL(seqL_k, b_idx)

            labs_A = labs[a_idx]
            labs_B = labs[b_idx]
            domain_A = domain_typs[a_idx].detach().long()
            domain_B = domain_typs[b_idx].detach().long()

            A_q_pack = self.encoder((sils_A_q, seqL_A_q), domain=domain_A, return_features=True)
            A_k_pack = self.encoder((sils_A_k, seqL_A_k), domain=domain_A, return_features=True)
            B_q_pack = self.encoder((sils_B_q, seqL_B_q), domain=domain_B, return_features=True)
            B_k_pack = self.encoder((sils_B_k, seqL_B_k), domain=domain_B, return_features=True)

            A_q_feats, A_q_mid = A_q_pack['high'], A_q_pack['mid']
            A_k_feats, A_k_mid = A_k_pack['high'], A_k_pack['mid']
            B_q_feats, B_q_mid = B_q_pack['high'], B_q_pack['mid']
            B_k_feats, B_k_mid = B_k_pack['high'], B_k_pack['mid']

            z1_A = self.projector(A_q_feats)
            z2_A = self.projector(A_k_feats)
            z3_B = self.projector(B_q_feats)
            z4_B = self.projector(B_k_feats)


            # 域内监督对比：labels 显式参与 SupCon，保证各域内保留身份判别性。
            supcon_features_A = torch.stack([z1_A, z2_A], dim=1)
            supcon_features_B = torch.stack([z3_B, z4_B], dim=1)

            mid_featuresA = torch.cat([A_q_mid, A_k_mid], dim=0)
            mid_featuresB = torch.cat([B_q_mid, B_k_mid], dim=0)
            # Mid-level domain alignment: use MMD between two domains' mid features.
            # MultiDomainMMDLoss accepts logits as (feat_domain0, feat_domain1).
            mid_mmd_input = {
                'logits': (mid_featuresA, mid_featuresB),
                'labels': None,
            }

            high_features = torch.cat([z1_A, z2_A, z3_B, z4_B], dim=0)  # [B, C, P]
            part_features = high_features.permute(0, 2, 1).contiguous()  # [B, P, C]
            part_features = part_features.view(-1, part_features.size(-1))  # [B*P, C]
            


            grl_feats = grad_reverse(part_features, lambd=self._get_grl_lambda())
            domain_logits = self.domain_classifier(grl_feats)

            domain_labels = torch.cat([domain_A, domain_A, domain_B, domain_B], dim=0)
            domain_labels = domain_labels.unsqueeze(1).repeat(1, self.p).reshape(-1)

            retval = {
                'training_feat': {
                    'supcon_A': {'features': supcon_features_A, 'labels': labs_A},
                    'supcon_B': {'features': supcon_features_B, 'labels': labs_B},
                    'mid_mmd_loss': mid_mmd_input,
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
        else:
            sils, _, _, _, seqL = inputs
            feat = self.encoder((sils, seqL), return_features=False)
            feat = self.projector(feat)
            retval = {
                'training_feat': None,
                'visual_summary': None,
                'inference_feat': {'embeddings': F.normalize(feat, dim=1)},
            }

        return retval


class no_grad:
    """按需启用 no_grad 的上下文管理器。"""

    def __init__(self, enable=True):
        self.enable = bool(enable)
        self.context = torch.no_grad() if self.enable else None

    def __enter__(self):
        if self.enable:
            return self.context.__enter__()
        return self

    def __exit__(self, *args):
        if self.enable:
            return self.context.__exit__(*args)
        return False
