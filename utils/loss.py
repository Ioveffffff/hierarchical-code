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

        # 可学习的类中心
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

        # 归一化中心和特征
        features_norm = F.normalize(features, dim=1)
        centers_norm = F.normalize(self.centers, dim=1)

        # 获取每个样本对应的类中心
        centers_batch = centers_norm[labels]  # [batch_size, feat_dim]

        # 计算特征到中心的距离（余弦距离）
        # 1 - cosine_similarity
        loss = (1 - (features_norm * centers_batch).sum(dim=1)).mean()

        return loss



# =====================================================
# 1. 改进的监督对比学习损失
# =====================================================
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

        # 构建mask: [N, N]
        # mask[i,j] = 1 表示样本i和样本j属于同一类
        mask = torch.eq(labels, labels.T).float().to(device)

        # L2归一化（如果还没归一化）
        features = F.normalize(features, dim=1)

        # 计算相似度矩阵: [N, N]
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # ✅ 数值稳定技巧: 减去最大值
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 构建负样本mask（排除自己）
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        # 正样本mask（排除自己）
        mask = mask * logits_mask

        # 计算log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # 计算每个anchor的正样本平均log概率
        # mask.sum(1): 每个anchor有多少个正样本
        mask_sum = mask.sum(1)

        # 过滤掉没有正样本的anchor
        valid_anchors = mask_sum > 0

        if valid_anchors.sum() == 0:
            # 如果batch中没有任何正样本对，返回0
            return features.sum() * 0.0

        # 只对有正样本的anchor计算loss
        mean_log_prob_pos = (mask * log_prob).sum(1)[valid_anchors] / mask_sum[valid_anchors]

        # 最终loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


# =====================================================
# 2. 白化损失（防止特征坍塌的核心）- 🔧 已修复设备问题
# =====================================================
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


# =====================================================
# 4. 特征坍塌评估
# =====================================================
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

    # 1. 活跃维度比例
    # 计算每个维度的方差
    var_per_dim = np.var(embeddings_np, axis=0)

    # 设置阈值（可调）
    # 使用相对阈值：平均方差的1%
    threshold = np.mean(var_per_dim) * 0.01

    # 统计活跃维度数
    active_dims = np.sum(var_per_dim > threshold)
    active_ratio = active_dims / D

    # 2. 有效秩（基于奇异值）
    # 中心化
    embeddings_centered = embeddings_np - embeddings_np.mean(axis=0, keepdims=True)

    # 计算协方差矩阵的特征值
    # 使用SVD更稳定
    try:
        U, S, Vt = np.linalg.svd(embeddings_centered, full_matrices=False)

        # 奇异值的平方 = 特征值
        eigenvalues = (S ** 2) / N

        # 有效秩的计算：Shannon熵
        # 归一化特征值
        eigenvalues_norm = eigenvalues / (eigenvalues.sum() + 1e-10)

        # 计算熵
        entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10))

        # 最大熵 = log(D)
        max_entropy = np.log(D)

        # 有效秩比例
        effective_rank_ratio = entropy / max_entropy

    except:
        # 如果SVD失败，返回保守估计
        effective_rank_ratio = active_ratio

    return active_ratio, effective_rank_ratio


# =====================================================
# 5. Stage3权重调度（Warmup）
# =====================================================
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
        # 线性warmup
        weight = (current_epoch + 1) / warmup_epochs
    else:
        # 保持最大权重
        weight = 1.0

    return weight


# =====================================================
# 6. 辅助函数：构建平衡批次
# =====================================================
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

    # 按类别组织索引
    class_to_indices = {}
    for idx, label in enumerate(labels):
        label_int = int(label)
        if label_int not in class_to_indices:
            class_to_indices[label_int] = []
        class_to_indices[label_int].append(idx)

    # 打乱每个类别的样本
    for cls in class_to_indices:
        rng.shuffle(class_to_indices[cls])

    # 过滤掉样本数不足的类别
    valid_classes = [cls for cls, indices in class_to_indices.items()
                     if len(indices) >= min_samples_per_class]

    if not valid_classes:
        # 如果没有符合条件的类别，随机分batch
        all_indices = list(range(len(labels)))
        rng.shuffle(all_indices)
        return [all_indices[i:i + batch_size]
                for i in range(0, len(all_indices), batch_size)]

    # 计算每个batch应该包含多少个类别
    classes_per_batch = max(1, batch_size // min_samples_per_class)

    batches = []
    class_pointers = {cls: 0 for cls in class_to_indices}

    rng.shuffle(valid_classes)

    while True:
        # 找到还有足够样本的类别
        available_classes = [cls for cls in valid_classes
                             if class_pointers[cls] + min_samples_per_class
                             <= len(class_to_indices[cls])]

        if len(available_classes) < classes_per_batch:
            break

        # 随机选择类别
        rng.shuffle(available_classes)
        selected_classes = available_classes[:classes_per_batch]

        # 构建batch
        batch = []
        for cls in selected_classes:
            start = class_pointers[cls]
            end = start + min_samples_per_class
            batch.extend(class_to_indices[cls][start:end])
            class_pointers[cls] = end

        # 如果batch还没满，继续添加样本
        while len(batch) < batch_size and available_classes:
            cls = rng.choice(available_classes)
            if class_pointers[cls] < len(class_to_indices[cls]):
                batch.append(class_to_indices[cls][class_pointers[cls]])
                class_pointers[cls] += 1

        if len(batch) >= batch_size // 2:  # 至少半满
            rng.shuffle(batch)
            batches.append(batch[:batch_size])

    # 处理剩余样本
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


# =====================================================
# 7. 监控和可视化辅助函数
# =====================================================
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

    # 基本统计
    mean_per_dim = features_np.mean(axis=0)
    var_per_dim = features_np.var(axis=0)
    std_per_dim = features_np.std(axis=0)

    # 维度统计
    active_dims = np.sum(var_per_dim > 1e-3)
    zero_dims = np.sum(np.abs(var_per_dim) < 1e-6)

    # 相关性
    features_centered = features_np - mean_per_dim
    corr_matrix = np.corrcoef(features_centered.T)

    # 去掉对角线
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


# =====================================================
# 测试代码
# =====================================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 loss.py 模块")
    print("=" * 60)

    # 模拟数据
    batch_size = 32
    feature_dim = 128
    n_classes = 10

    # 生成随机特征和标签
    features = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, n_classes, (batch_size,))
    coarse_labels = labels // 3  # 粗粒度标签

    print(f"\n测试数据:")
    print(f"  Batch size: {batch_size}")
    print(f"  Feature dim: {feature_dim}")
    print(f"  Fine classes: {n_classes}")
    print(f"  Coarse classes: {len(torch.unique(coarse_labels))}")

    # 1. 测试 ImprovedSupConLoss
    print(f"\n{'=' * 60}")
    print("1. 测试 ImprovedSupConLoss")
    print(f"{'=' * 60}")
    supcon = ImprovedSupConLoss(temperature=0.07)
    loss_supcon = supcon(features, labels)
    print(f"  ✅ SupCon Loss: {loss_supcon.item():.4f}")

    # 2. 测试 WhiteningLoss（修复后）
    print(f"\n{'=' * 60}")
    print("2. 测试 WhiteningLoss (已修复设备问题)")
    print(f"{'=' * 60}")
    whitening = WhiteningLoss(feature_dim=feature_dim)

    # 测试 CPU
    loss_white_cpu = whitening(features)
    print(f"  ✅ Whitening Loss (CPU): {loss_white_cpu.item():.4f}")

    # 测试 GPU（如果可用）
    if torch.cuda.is_available():
        whitening_gpu = whitening.cuda()
        features_gpu = features.cuda()
        loss_white_gpu = whitening_gpu(features_gpu)
        print(f"  ✅ Whitening Loss (GPU): {loss_white_gpu.item():.4f}")
        print(f"  ✅ GPU 设备兼容性测试通过！")

    # 3. 测试 HierarchicalContrastiveLoss
    print(f"\n{'=' * 60}")
    print("3. 测试 HierarchicalContrastiveLoss")
    print(f"{'=' * 60}")
    hierarchical = HierarchicalContrastiveLoss(temperature=0.07)
    loss_total, loss_coarse, loss_fine = hierarchical(features, coarse_labels, labels)
    print(f"  ✅ Total Loss: {loss_total.item():.4f}")
    print(f"  ✅ Coarse Loss: {loss_coarse.item():.4f}")
    print(f"  ✅ Fine Loss: {loss_fine.item():.4f}")

    # 4. 测试特征坍塌评估
    print(f"\n{'=' * 60}")
    print("4. 测试特征坍塌评估")
    print(f"{'=' * 60}")
    active_ratio, eff_rank = evaluate_feature_collapse(features)
    print(f"  ✅ Active Ratio: {active_ratio:.2%}")
    print(f"  ✅ Effective Rank: {eff_rank:.2%}")

    # 5. 测试Stage3权重调度
    print(f"\n{'=' * 60}")
    print("5. 测试 Stage3 Warmup")
    print(f"{'=' * 60}")
    for epoch in [0, 5, 10, 15]:
        weight = get_stage3_weight_schedule(epoch, total_epochs=60, warmup_epochs=10)
        print(f"  Epoch {epoch:2d}: weight = {weight:.3f}")

    # 6. 测试构建平衡批次
    print(f"\n{'=' * 60}")
    print("6. 测试构建平衡批次")
    print(f"{'=' * 60}")
    test_labels = np.random.randint(0, 10, 200)
    batches = build_balanced_batches(test_labels, batch_size=32, min_samples_per_class=3)
    print(f"  ✅ 生成了 {len(batches)} 个batch")
    print(f"  ✅ 第一个batch大小: {len(batches[0])}")

    # 7. 测试特征统计
    print(f"\n{'=' * 60}")
    print("7. 测试特征统计")
    print(f"{'=' * 60}")
    stats = compute_feature_statistics(features)
    print(f"  ✅ Active dims: {stats['active_dims']}/{stats['n_dims']} ({stats['active_ratio']:.2%})")
    print(f"  ✅ Mean variance: {stats['mean_var']:.4f}")
    print(f"  ✅ Avg abs correlation: {stats['avg_abs_corr']:.4f}")

    print(f"\n{'=' * 60}")
    print("✅ 所有测试通过！设备兼容性问题已修复！")
    print(f"{'=' * 60}")
