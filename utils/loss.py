"""
loss.py - 完整的损失函数模块
解决特征坍塌 + 数值稳定 + 层次化对比学习

主要功能:
1. ImprovedSupConLoss - 改进的监督对比学习损失
2. WhiteningLoss - 白化损失（防止特征坍塌）
3. HierarchicalContrastiveLoss - 层次化对比学习损失
4. evaluate_feature_collapse - 特征坍塌评估
5. get_stage3_weight_schedule - Stage3权重调度
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterLoss(nn.Module):
    """中心损失：强制同类样本靠近类中心"""

    def __init__(self, num_classes, feat_dim, device):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

        self.centers = nn.Parameter(
            torch.randn(num_classes, feat_dim, device=device)
        )

    def forward(self, features, labels):
        """
        Args:
            features: [batch_size, feat_dim] - L2归一化后的特征
            labels: [batch_size] - 类别标签
        Returns:
            loss: 标量
        """
        batch_size = features.size(0)

        features_norm = F.normalize(features, dim=1)
        centers_norm = F.normalize(self.centers, dim=1)

        centers_batch = centers_norm[labels]  # [batch_size, feat_dim]

        loss = (1 - (features_norm * centers_batch).sum(dim=1)).mean()

        return loss


class ImprovedSupConLoss(nn.Module):
    """
    改进的监督对比学习损失（SupCon）

    核心改进:
    - 数值稳定的实现（避免exp溢出）
    - 支持多正样本情况
    - 自动处理batch中没有正样本的情况
    """

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(ImprovedSupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        Args:
            features: [N, D] 归一化后的特征向量
            labels: [N] 类别标签

        Returns:
            loss: 标量
        """
        device = features.device
        batch_size = features.shape[0]

        if len(labels.shape) == 1:
            labels = labels.contiguous().view(-1, 1)

        # mask[i,j] = 1 表示样本i和样本j属于同一类
        mask = torch.eq(labels, labels.T).float().to(device)

        features = F.normalize(features, dim=1)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # 数值稳定技巧: 减去最大值
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mask_sum = mask.sum(1)

        valid_anchors = mask_sum > 0

        if valid_anchors.sum() == 0:
            # 如果batch中没有任何正样本对，返回0
            return features.sum() * 0.0

        mean_log_prob_pos = (mask * log_prob).sum(1)[valid_anchors] / mask_sum[valid_anchors]

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


class WhiteningLoss(nn.Module):
    """Variance floor and covariance decorrelation on *unnormalized* z.

    This is a regularizer, not an assertion that a B<D batch can be fully
    whitened. The legacy class name is retained for import compatibility.
    """

    def __init__(self, feature_dim=128, std_target=0.1, covariance_weight=0.01):
        super().__init__()
        if std_target <= 0 or covariance_weight < 0:
            raise ValueError('Invalid variance/covariance regularizer parameters')
        self.feature_dim = feature_dim
        self.std_target = std_target
        self.covariance_weight = covariance_weight

    def components(self, features):
        if features.ndim != 2:
            raise ValueError("Expected unnormalized features [B, D]")
        if features.shape[0] < 2:
            zero = features.sum() * 0.0
            return zero, zero, zero, zero
        x = features.float() - features.float().mean(0, keepdim=True)
        cov = x.T @ x / (len(x) - 1)
        std = torch.sqrt(cov.diagonal() + (self.std_target * .01) ** 2)
        variance = F.relu(1 - std / self.std_target).mean()
        off_diagonal = cov - torch.diag_embed(cov.diagonal())
        covariance = (off_diagonal / self.std_target ** 2).square().sum() / features.shape[1]
        return variance, covariance, std.mean(), (std >= self.std_target).float().mean()

    def forward(self, features):
        variance, covariance, _, _ = self.components(features)
        return variance + self.covariance_weight * covariance


class HierarchicalContrastiveLoss(nn.Module):
    """Fine/coarse SupCon and a relative margin for hard negatives.

    Hard negatives share a coarse pseudo label but have different fine labels.
    For each eligible anchor, the most similar hard negative must fall below
    its least similar fine positive by `hard_margin` (cosine units).
    """

    def __init__(self, temperature=0.07, coarse_weight=0.3, fine_weight=0.7,
                 hard_margin=0.2, hard_weight=0.2):
        super().__init__()
        self.coarse_weight = coarse_weight
        self.fine_weight = fine_weight
        self.hard_margin = hard_margin
        self.hard_weight = hard_weight
        self.supcon = ImprovedSupConLoss(temperature=temperature)

    def hard_negative_loss(self, features, coarse_labels, fine_labels):
        features = F.normalize(features, dim=1)
        same_coarse = coarse_labels[:, None] == coarse_labels[None, :]
        same_fine = fine_labels[:, None] == fine_labels[None, :]
        positives = same_fine & ~torch.eye(len(features), device=features.device,
                                          dtype=torch.bool)
        negatives = same_coarse & ~same_fine
        valid = positives.any(1) & negatives.any(1)
        if not valid.any():
            return features.sum() * 0.0
        similarity = features @ features.T
        # Select valid rows first: no infinities enter the hinge or its gradient.
        similarity = similarity[valid]
        positive = similarity.masked_fill(~positives[valid], float('inf')).amin(1)
        negative = similarity.masked_fill(~negatives[valid], -float('inf')).amax(1)
        return F.relu(self.hard_margin + negative - positive).mean()

    def forward(self, features, coarse_labels, fine_labels):
        features = F.normalize(features, dim=1)
        zero = features.sum() * 0.0
        loss_fine = self.supcon(features, fine_labels) if self.fine_weight else zero
        loss_coarse = (self.supcon(features, coarse_labels)
                       if self.coarse_weight and coarse_labels.unique().numel() > 1 else zero)
        loss_hard = (self.hard_negative_loss(features, coarse_labels, fine_labels)
                     if self.hard_weight else zero)
        total = (self.fine_weight * loss_fine + self.coarse_weight * loss_coarse
                 + self.hard_weight * loss_hard)
        return total, loss_coarse, loss_fine


def evaluate_feature_collapse(embeddings):
    """
    评估特征是否发生坍塌

    两个指标:
    1. 活跃维度比例: 方差 > 阈值的维度占比
    2. 有效秩: 奇异值分布的均匀程度

    Args:
        embeddings: [N, D] 特征矩阵（Tensor或numpy）

    Returns:
        active_ratio: 活跃维度比例 (0-1)
        effective_rank_ratio: 有效秩比例 (0-1)
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = embeddings

    N, D = embeddings_np.shape

    var_per_dim = np.var(embeddings_np, axis=0)

    # 使用相对阈值：平均方差的1%
    threshold = np.mean(var_per_dim) * 0.01

    active_dims = np.sum(var_per_dim > threshold)
    active_ratio = active_dims / D

    embeddings_centered = embeddings_np - embeddings_np.mean(axis=0, keepdims=True)

    try:
        U, S, Vt = np.linalg.svd(embeddings_centered, full_matrices=False)

        eigenvalues = (S ** 2) / N

        # 有效秩的计算：Shannon熵
        eigenvalues_norm = eigenvalues / (eigenvalues.sum() + 1e-10)

        entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10))

        max_entropy = np.log(D)

        effective_rank_ratio = entropy / max_entropy

    except:
        # 如果SVD失败，返回保守估计
        effective_rank_ratio = active_ratio

    return active_ratio, effective_rank_ratio


def get_stage3_weight_schedule(current_epoch, total_epochs, warmup_epochs=10):
    """
    Stage3的warmup权重调度

    防止Stage3开始时loss突增

    Args:
        current_epoch: 当前epoch（从0开始）
        total_epochs: Stage3总epoch数
        warmup_epochs: warmup的epoch数

    Returns:
        weight: 权重系数 (0-1)

    调度策略:
    - 前warmup_epochs个epoch: 线性从0增长到1
    - 之后: 保持为1
    """
    if current_epoch < warmup_epochs:
        weight = (current_epoch + 1) / warmup_epochs
    else:
        weight = 1.0

    return weight


def build_balanced_batches(labels, batch_size, min_samples_per_class=3, seed=42):
    """
    构建类别平衡的批次

    目标:
    - 每个batch包含多个类别
    - 每个类别至少有min_samples_per_class个样本

    Args:
        labels: [N] 标签数组
        batch_size: 批次大小
        min_samples_per_class: 每个类别最少样本数
        seed: 随机种子

    Returns:
        batches: List[List[int]] 每个batch的样本索引
    """
    rng = np.random.RandomState(seed)

    class_to_indices = {}
    for idx, label in enumerate(labels):
        label_int = int(label)
        if label_int not in class_to_indices:
            class_to_indices[label_int] = []
        class_to_indices[label_int].append(idx)

    for cls in class_to_indices:
        rng.shuffle(class_to_indices[cls])

    valid_classes = [cls for cls, indices in class_to_indices.items()
                     if len(indices) >= min_samples_per_class]

    if not valid_classes:
        all_indices = list(range(len(labels)))
        rng.shuffle(all_indices)
        return [all_indices[i:i + batch_size]
                for i in range(0, len(all_indices), batch_size)]

    classes_per_batch = max(1, batch_size // min_samples_per_class)

    batches = []
    class_pointers = {cls: 0 for cls in class_to_indices}

    rng.shuffle(valid_classes)

    while True:
        available_classes = [cls for cls in valid_classes
                             if class_pointers[cls] + min_samples_per_class
                             <= len(class_to_indices[cls])]

        if len(available_classes) < classes_per_batch:
            break

        rng.shuffle(available_classes)
        selected_classes = available_classes[:classes_per_batch]

        batch = []
        for cls in selected_classes:
            start = class_pointers[cls]
            end = start + min_samples_per_class
            batch.extend(class_to_indices[cls][start:end])
            class_pointers[cls] = end

        while len(batch) < batch_size and available_classes:
            cls = rng.choice(available_classes)
            if class_pointers[cls] < len(class_to_indices[cls]):
                batch.append(class_to_indices[cls][class_pointers[cls]])
                class_pointers[cls] += 1

        if len(batch) >= batch_size // 2:  # 至少半满
            rng.shuffle(batch)
            batches.append(batch[:batch_size])

    leftovers = []
    for cls in valid_classes:
        leftovers.extend(class_to_indices[cls][class_pointers[cls]:])

    if leftovers:
        rng.shuffle(leftovers)
        for i in range(0, len(leftovers), batch_size):
            batch = leftovers[i:i + batch_size]
            if len(batch) >= batch_size // 2:
                batches.append(batch)

    return batches


def compute_feature_statistics(features):
    """
    计算特征的详细统计信息

    Args:
        features: [N, D] 特征矩阵

    Returns:
        stats: dict 包含各种统计指标
    """
    if isinstance(features, torch.Tensor):
        features_np = features.detach().cpu().numpy()
    else:
        features_np = features

    N, D = features_np.shape

    mean_per_dim = features_np.mean(axis=0)
    var_per_dim = features_np.var(axis=0)
    std_per_dim = features_np.std(axis=0)

    active_dims = np.sum(var_per_dim > 1e-3)
    zero_dims = np.sum(np.abs(var_per_dim) < 1e-6)

    features_centered = features_np - mean_per_dim
    corr_matrix = np.corrcoef(features_centered.T)

    off_diag_corr = corr_matrix[~np.eye(D, dtype=bool)]
    avg_abs_corr = np.abs(off_diag_corr).mean()

    stats = {
        'n_samples': N,
        'n_dims': D,
        'active_dims': int(active_dims),
        'active_ratio': active_dims / D,
        'zero_dims': int(zero_dims),
        'mean_var': float(var_per_dim.mean()),
        'std_var': float(var_per_dim.std()),
        'min_var': float(var_per_dim.min()),
        'max_var': float(var_per_dim.max()),
        'avg_abs_corr': float(avg_abs_corr),
    }

    return stats
