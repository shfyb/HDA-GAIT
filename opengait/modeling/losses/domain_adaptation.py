import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseLoss


class MultiDomainMMDLoss(BaseLoss):
    """多域MMD损失，支持自动sigma和多尺度高斯核"""

    def __init__(
        self,
        loss_term_weight=1.0,
        sigma=None,
        multi_sigma=False,
        sigma_list=None,
        sigma_ema_momentum=0.0,
        min_sigma=1e-3,
    ):
        """
        Args:
            loss_term_weight: MMD损失权重
            sigma: 固定sigma，如果为None则自动计算
            multi_sigma: 是否使用多尺度高斯核
            sigma_list: 如果multi_sigma=True，可以指定多个sigma列表
            sigma_ema_momentum: 当 sigma=None 时，对每个 batch 估计的 sigma 做 EMA 平滑的动量系数。
                - 0.0: 不做 EMA，每次 forward 使用当前 batch 的 sigma（推荐用于消融/不稳定排查）
                - 0.9: 典型 EMA，能显著缓解不同 batch 组成引起的 sigma 波动
            min_sigma: sigma 下界，避免数值问题
        """
        super().__init__(loss_term_weight)
        self.sigma = sigma
        self.multi_sigma = multi_sigma
        self.sigma_list = sigma_list if sigma_list is not None else [1.0, 2.0, 4.0]
        self.sigma_ema_momentum = float(sigma_ema_momentum)
        self.min_sigma = float(min_sigma)
        self._sigma_ema = None

    def gaussian_kernel(self, x, y, sigma):
        beta = 1.0 / (2.0 * sigma ** 2)
        dist = torch.cdist(x.float(), y.float(), p=2) ** 2
        K = torch.exp(-beta * dist)
        return K

    def forward(self, logits, labels=None):
        # ---------- 1. 解析输入 ----------
        if isinstance(logits, (tuple, list)):
            feat_drone, feat_ground = logits
        else:
            if labels is None:
                raise ValueError("Domain labels are required when logits is a single tensor")
            device = logits.device
            if isinstance(labels, (list, tuple)):
                labels = torch.tensor([int(l) for l in labels], device=device)
            feat_drone = logits[labels == 0]
            feat_ground = logits[labels == 1]

        # ---------- 2. 空检查 ----------
        if feat_drone.numel() == 0 or feat_ground.numel() == 0:
            loss = torch.tensor(0.0, device=logits[0].device if isinstance(logits, (tuple, list)) else logits.device)
            self.info.update({'loss': loss.detach(), 'accuracy': torch.tensor(1.0)})
            print("⚠️ One domain has no samples, skipping MMD computation")
            return loss, self.info

        # ---------- 3. 特征归一化 ----------
        feat_drone = feat_drone.float()
        feat_ground = feat_ground.float()

        # 方法1：标准化到均值0，方差1
        #feat_drone = (feat_drone - feat_drone.mean(dim=0, keepdim=True)) / (feat_drone.std(dim=0, keepdim=True) + 1e-6)
        #feat_ground = (feat_ground - feat_ground.mean(dim=0, keepdim=True)) / (feat_ground.std(dim=0, keepdim=True) + 1e-6)


        # 方法2（可选）：L2归一化每个样本向量
        #feat_drone = F.normalize(feat_drone, p=2, dim=1)
        #feat_ground = F.normalize(feat_ground, p=2, dim=1)

        # ---------- 4. sigma 估计 ----------
        # 若 sigma 在构造时显式给定，则视为固定带宽；否则对每个 batch 估计并可选 EMA 平滑。
        if self.sigma is None:
            with torch.no_grad():
                dist = torch.cdist(feat_drone, feat_ground, p=2)
                sigma_cur = torch.median(dist).item()
            sigma_cur = max(float(sigma_cur), self.min_sigma)

            if self.sigma_ema_momentum > 0.0:
                if self._sigma_ema is None:
                    self._sigma_ema = sigma_cur
                else:
                    m = self.sigma_ema_momentum
                    self._sigma_ema = m * float(self._sigma_ema) + (1.0 - m) * sigma_cur
                sigma_used = max(float(self._sigma_ema), self.min_sigma)
            else:
                sigma_used = sigma_cur
        else:
            sigma_used = max(float(self.sigma), self.min_sigma)

        # ---------- 5. MMD计算 ----------
        if self.multi_sigma:
            K_xx_list, K_yy_list, K_xy_list = [], [], []
            for sigma in self.sigma_list:
                K_xx_list.append(self.gaussian_kernel(feat_drone, feat_drone, sigma))
                K_yy_list.append(self.gaussian_kernel(feat_ground, feat_ground, sigma))
                K_xy_list.append(self.gaussian_kernel(feat_drone, feat_ground, sigma))
            K_xx = sum(K_xx_list) / len(K_xx_list)
            K_yy = sum(K_yy_list) / len(K_yy_list)
            K_xy = sum(K_xy_list) / len(K_xy_list)
        else:
            K_xx = self.gaussian_kernel(feat_drone, feat_drone, sigma_used)
            K_yy = self.gaussian_kernel(feat_ground, feat_ground, sigma_used)
            K_xy = self.gaussian_kernel(feat_drone, feat_ground, sigma_used)

        # ---------- 6. 调试信息 ----------
        '''
        print(f"[MMD Debug] feat_drone: mean {feat_drone.mean():.6f}, std {feat_drone.std():.6f}")
        print(f"[MMD Debug] feat_ground: mean {feat_ground.mean():.6f}, std {feat_ground.std():.6f}")
        print(f"[MMD Debug] K_xx: mean {K_xx.mean():.6f}, min {K_xx.min():.6f}, max {K_xx.max():.6f}")
        print(f"[MMD Debug] K_yy: mean {K_yy.mean():.6f}, min {K_yy.min():.6f}, max {K_yy.max():.6f}")
        print(f"[MMD Debug] K_xy: mean {K_xy.mean():.6f}, min {K_xy.min():.6f}, max {K_xy.max():.6f}")
        '''

        mmd_loss = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()

        if torch.isnan(mmd_loss):
            print("⚠️ NaN detected in MMD loss, replacing with 0.")
            mmd_loss = torch.tensor(0.0, device=feat_drone.device)

        # ---------- 7. 域对齐精度估计 ----------
        with torch.no_grad():
            mean_dist = torch.mean(torch.cdist(feat_drone.float(), feat_ground.float(), p=2))
            alignment_accuracy = 1.0 - torch.clamp(mean_dist / (mean_dist + 1.0), 0.0, 1.0)

        # 权重仅由 LossAggregator 统一乘以 loss_term_weight，避免与 forward 内重复相乘

        # ---------- 8. 结果返回 ----------
        self.info.update({
            'loss': mmd_loss.detach(),
            'accuracy': alignment_accuracy.detach()
        })

        return mmd_loss, self.info






class MultiDomainCORALLoss(nn.Module):
    def __init__(self, weight=1.0, loss_term_weight=None, debug=False):
        super(MultiDomainCORALLoss, self).__init__()
        self.info = {}
        # YAML 里通常写 loss_term_weight；与 weight 二选一，同时给时以 loss_term_weight 为准
        if loss_term_weight is not None:
            weight = float(loss_term_weight)
        self.loss_term_weight = weight
        self.debug = debug

    def forward(self, logits, labels=None):
        if not isinstance(logits, (tuple, list)) or len(logits) != 2:
            raise ValueError("Expected logits as (feat_drone, feat_ground) tuple.")

        feat_drone, feat_ground = logits
        device = feat_drone.device
        eps = 1e-6

        # 空 batch 保护
        if feat_drone.numel() == 0 or feat_ground.numel() == 0:
            loss = torch.tensor(0.0, device=device)
            self.info.update({'loss': loss.detach(), 'accuracy': torch.tensor(1.0)})
            return loss, self.info

        ns, nt = feat_drone.size(0), feat_ground.size(0)
        d = feat_drone.size(1)

        # 小样本保护
        if ns < 2 or nt < 2:
            loss = torch.tensor(0.0, device=device)
            self.info.update({'loss': loss.detach(), 'accuracy': torch.tensor(1.0)})
            return loss, self.info

        # 转 float32（避免 fp16 数值炸裂）
        feat_drone = feat_drone.float()
        feat_ground = feat_ground.float()

        # 去均值
        xm_d = feat_drone - feat_drone.mean(dim=0, keepdim=True)
        xm_g = feat_ground - feat_ground.mean(dim=0, keepdim=True)

        # 协方差（更稳定写法）
        cov_d = (xm_d.t() @ xm_d) / (ns - 1)
        cov_g = (xm_g.t() @ xm_g) / (nt - 1)

        # 数值稳定（防奇异矩阵）
        cov_d = cov_d + eps * torch.eye(d, device=device)
        cov_g = cov_g + eps * torch.eye(d, device=device)

        # ===== 改进1：去掉 4*d*d（否则512维几乎没梯度）=====
        coral_loss_raw = torch.mean((cov_d - cov_g) ** 2)

        # 不在此处乘 loss_term_weight，由 LossAggregator 统一加权，避免重复

        # 对齐指标（可解释性更强）
        loss = coral_loss_raw
        diff = torch.mean(torch.abs(cov_d - cov_g))
        coral_accuracy = 1.0 / (1.0 + diff)

        self.info.update({
            'loss': loss.detach(),
            'accuracy': coral_accuracy.detach()
        })
        return loss, self.info


class AdvLoss(BaseLoss):
    def __init__(self, loss_term_weight=1.0, log_prefix="adv_domain"):
        super().__init__(loss_term_weight)
        self.log_prefix = log_prefix

    def forward(self, **kwargs):
        """
        kwargs should contain:
            logits: Tensor [N, 2]
            labels: Tensor [N]
        """
        logits = kwargs.get("logits")
        labels = kwargs.get("labels")
        if logits is None or labels is None:
            raise ValueError("AdvLoss requires 'logits' and 'labels' in kwargs")
        
        #print("adv_logits:",logits)
        #print("adv_labels:",labels)

        # ----- 1. adversarial classification loss -----
        loss = F.cross_entropy(logits, labels)

        # ----- 2. compute accuracy -----
        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).float().sum()
        acc = correct / labels.numel()

        # 权重由 LossAggregator 统一乘以 loss_term_weight

        # ----- 4. return info -----
        self.info.update({
            "loss": loss.detach(),
            "accuracy": acc.detach()
        })
        return loss, self.info
