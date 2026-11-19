import os
import random
from pathlib import Path
import numpy as np

from tqdm import tqdm
import json
from datetime import datetime

import torch

from dual_channel_model import EnhancedEntityEmbedding
from load_dataset import load_entities_from_csv, load_texts_from_csv
from clustering_utils import (
    evaluate_clustering,
    evaluate_clustering_comparison,
    evaluate_clustering_hierarchical_auto,
    build_coarse_labels_with_acrl,
    evaluate_clustering_acrl_kmeans
)

from utils.loss import (
    ImprovedSupConLoss,
    WhiteningLoss,
    HierarchicalContrastiveLoss,
    get_stage3_weight_schedule,
    evaluate_feature_collapse
)
from utils.improved_hierarchy import (
    build_hierarchical_labels_hybrid,
    build_hierarchical_labels,
    evaluate_coarse_quality,
    augment_entity
)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  # 服务器环境使用非交互式后端


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def extract_embeddings(model, entities, batch_size=64, desc="Embedding"):
    model.eval()
    all_emb = []
    for i in tqdm(range(0, len(entities), batch_size), desc=desc, ncols=100, leave=False):
        batch = entities[i:i + batch_size]
        emb = model(batch).detach().cpu()
        all_emb.append(emb)
    embeddings = torch.cat(all_emb, dim=0)
    return embeddings


def visualize_clustering_tsne(embeddings, labels, save_path, title="Clustering Visualization",
                              perplexity=30, max_iter=1000, random_state=42):
    """
    使用t-SNE降维并可视化聚类结果

    Args:
        embeddings: 高维特征向量 (n_samples, n_features)
        labels: 聚类标签 (n_samples,)
        save_path: 保存路径
        title: 图片标题
        perplexity: t-SNE的perplexity参数
        max_iter: t-SNE的最大迭代次数 (注意：新版scikit-learn使用max_iter而非n_iter)
        random_state: 随机种子
    """
    print(f"   生成: {title}")

    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter,
                random_state=random_state, verbose=0)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 10), dpi=150)

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # 使用tab20颜色映射（支持最多20个类别）
    if n_clusters <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    elif n_clusters <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    else:
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, n_clusters))

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[idx]],
            label=f'Cluster {label}',
            alpha=0.7,
            s=50,  # 点的大小
            edgecolors='white',
            linewidths=0.5
        )

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)

    if n_clusters <= 15:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                   frameon=True, fontsize=10, ncol=1)
    else:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                   frameon=True, fontsize=8, ncol=2)

    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"      已保存至: {save_path}")

    return embeddings_2d


def build_balanced_batches(labels, batch_size, min_samples_per_class=3, seed=42):
    """构建类别平衡的批次"""
    np.random.seed(seed)
    unique_labels = np.unique(labels)

    label_to_indices = {}
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        if len(indices) >= min_samples_per_class:
            label_to_indices[label] = indices

    batches = []
    all_indices = []

    for indices in label_to_indices.values():
        all_indices.extend(indices.tolist())

    np.random.shuffle(all_indices)

    for i in range(0, len(all_indices), batch_size):
        batch = all_indices[i:i + batch_size]
        if len(batch) >= min_samples_per_class:
            batches.append(batch)

    return batches


def save_checkpoint(stage_dir, model, epoch, metrics, config, checkpoint_type='final'):
    """
    保存checkpoint

    Args:
        stage_dir: Stage目录路径
        model: 模型
        epoch: 当前epoch
        metrics: 评估指标
        config: 配置信息
        checkpoint_type: 'final' 或 'best'
    """
    checkpoint_path = os.path.join(stage_dir, f'{checkpoint_type}.pt')

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'config': config
    }, checkpoint_path)

    return checkpoint_path


def save_stage_results(stage_dir, stage_num, epoch, model, metrics, config, history, save_best=True):
    """
    保存每个Stage的完整结果

    Args:
        stage_dir: Stage目录路径
        stage_num: Stage编号 (1, 2, 3)
        epoch: 当前epoch
        model: 模型
        metrics: 评估指标
        config: 配置信息
        history: 训练历史
        save_best: 是否同时保存best模型
    """
    os.makedirs(stage_dir, exist_ok=True)

    final_path = save_checkpoint(stage_dir, model, epoch, metrics, config, 'final')

    if save_best and 'best_metrics' in config:
        best_path = save_checkpoint(
            stage_dir, model,
            config.get('best_epoch', epoch),
            config['best_metrics'],
            config,
            'best'
        )
        print(f"   - best checkpoint: {best_path}")

    config_path = os.path.join(stage_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    history_path = os.path.join(stage_dir, 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    metrics_path = os.path.join(stage_dir, 'final_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Stage {stage_num} 结果已保存至: {stage_dir}")
    print(f"   - final checkpoint: {final_path}")
    print(f"   - config: {config_path}")
    print(f"   - history: {history_path}")
    print(f"   - metrics: {metrics_path}")


# CGCL training is isolated from the BSE warm-up.
from utils.cgcl_training import train_three_stage_improved
from unsupervised_bse import pretrain_bse_instance, save_bse_history
from bse_checkpoint import (expected_metadata, load_bse_checkpoint,
                            save_bse_checkpoint)


class CharVocab:
    def __init__(self):
        self.char2id = {"<PAD>": 0, "<UNK>": 1}

    def build(self, entities):
        for s in entities:
            for ch in str(s):
                if ch not in self.char2id:
                    self.char2id[ch] = len(self.char2id)

    def __len__(self):
        return len(self.char2id)

    def get(self, key, default=1):
        return self.char2id.get(key, default)


def main():
    import argparse
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from clustering_utils import cluster_acc
    parser = argparse.ArgumentParser(description='Unsupervised instance BSE + pseudo-label CGCL')
    parser.add_argument('--csv', required=True, help='Input CSV with text and label columns')
    parser.add_argument('--backbone', required=True, help='Local Transformer checkpoint')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs-stage1', type=int, default=0,
                        help='Optional additional instance stage after unsupervised BSE')
    parser.add_argument('--epochs-stage2', type=int, default=100)
    parser.add_argument('--epochs-stage3', type=int, default=90)
    parser.add_argument('--epochs-bse', type=int, default=100,
                        help='Unsupervised instance contrastive BSE epochs')
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--pseudo-update-interval', type=int, default=5)
    parser.add_argument('--cluster-k-min', type=int, default=2)
    parser.add_argument('--cluster-k-max', type=int, default=256)
    parser.add_argument('--fixed-k', type=int, default=None,
                        help='Optional prespecified fine cluster count; omitted means label-free silhouette search')
    parser.add_argument('--cluster-sample-size', type=int, default=1024)
    parser.add_argument('--confidence-margin', type=float, default=0.02)
    parser.add_argument('--skip-comparison', action='store_true',
                        help='Only evaluate the final partition; skip legacy comparisons/plots')
    parser.add_argument('--seed', type=int, default=42)
    bse_group = parser.add_mutually_exclusive_group()
    bse_group.add_argument('--bse-checkpoint', type=Path,
                           help='Load an existing unsupervised BSE start for paired CGCL runs')
    bse_group.add_argument('--save-bse-checkpoint', type=Path,
                           help='Train BSE once and save the shared starting checkpoint')
    parser.add_argument('--final-embedding-source', choices=('cgcl', 'bse_anchor'), default='cgcl')
    parser.add_argument('--stage2-fine-weight', type=float, default=0.0)
    parser.add_argument('--disable-coarse', action='store_true')
    parser.add_argument('--disable-fine', action='store_true')
    parser.add_argument('--freeze-pseudo-labels', action='store_true')
    parser.add_argument('--whitening-weight', type=float, default=0.1)
    parser.add_argument('--pseudo-noise-rate', type=float, default=0.0)
    parser.add_argument('--pseudo-noise-scope', choices=('fine', 'coarse'), default='fine')
    args = parser.parse_args()
    if args.disable_coarse and args.disable_fine:
        parser.error('At least one contrastive granularity must remain enabled')
    if not 0 <= args.stage2_fine_weight <= 1:
        parser.error('--stage2-fine-weight must be between 0 and 1')
    if args.stage2_fine_weight and (args.disable_coarse or args.disable_fine):
        parser.error('--stage2-fine-weight requires both granularities')
    if args.whitening_weight < 0:
        parser.error('--whitening-weight must be nonnegative')
    if not 0 <= args.pseudo_noise_rate <= 1:
        parser.error('--pseudo-noise-rate must be between 0 and 1')
    if args.fixed_k is not None and args.fixed_k < 2:
        parser.error('--fixed-k must be at least 2')
    if args.epochs_bse < 1:
        parser.error('--epochs-bse must be at least 1')
    set_seed(args.seed)

    csv_path = args.csv
    backbone_path = args.backbone
    device = args.device

    entities = load_texts_from_csv(csv_path)
    if args.fixed_k is not None and args.fixed_k > len(entities) // 2:
        parser.error('--fixed-k cannot exceed half the number of training texts')
    print(f"\n{'=' * 60}")
    print(f"数据集信息")
    print(f"{'=' * 60}")
    print(f"  样本数: {len(entities)}")
    print(f"  CGCL 细簇数: {args.fixed_k if args.fixed_k is not None else '自动选择'}")
    print(f"{'=' * 60}\n")

    vocab = CharVocab()
    vocab.build(entities)

    model = EnhancedEntityEmbedding(
        vocab=vocab,
        backbone_model_name=backbone_path,
        char_embed_dim=128,
        char_cnn_output_dim=512,
        backbone_dim=768,
        fusion_dim=768,
        instance_dim=128,
        cluster_num=None,
        use_sequence_features=True,
        use_bi_mcm_attn=True,
        fusion_method="gate",
        freeze_backbone_layers=2,
        max_seq_length=64,
        device=device
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # BSE sees texts only. Two independent dropout views form each positive pair.
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_dir = Path('./outputs') / timestamp
    bse_dir = output_dir / 'bse_instance_pretrain'
    bse_dir.mkdir(parents=True, exist_ok=False)
    bse_metadata = expected_metadata(csv_path, backbone_path, args.seed,
                                     args.epochs_bse, args.batch_size,
                                     len(entities), len(vocab))
    if args.bse_checkpoint is not None:
        bse_history, bse_sha256 = load_bse_checkpoint(
            args.bse_checkpoint, model, bse_metadata)
        bse_source = str(args.bse_checkpoint.resolve())
        print('Loaded shared BSE checkpoint:', bse_source, bse_sha256)
    else:
        bse_history = pretrain_bse_instance(
            model=model,
            entities=entities,
            epochs=args.epochs_bse,
            batch_size=args.batch_size,
            lr_backbone=1e-6,
            lr_other=5e-5,
            device=device,
            seed=args.seed,
        )
        if args.save_bse_checkpoint is not None:
            bse_sha256 = save_bse_checkpoint(
                args.save_bse_checkpoint, model, bse_metadata, bse_history)
            bse_source = str(args.save_bse_checkpoint.resolve())
            print('Saved shared BSE checkpoint:', bse_source, bse_sha256)
        else:
            bse_sha256 = None
            bse_source = None
    save_bse_history(bse_dir, bse_history)
    (bse_dir / 'config.json').write_text(json.dumps({
        'mode': 'unsupervised_instance_nt_xent_v1',
        'epochs': args.epochs_bse,
        'batch_size': args.batch_size,
        'lr_backbone': 1e-6,
        'lr_other': 5e-5,
        'temperature': 0.1,
        'views': 'same_text_twice_with_independent_dropout',
        'representation': 'z_normalized',
        'classifier_head': None,
        'bse_checkpoint_source': bse_source,
        'bse_checkpoint_sha256': bse_sha256,
        'bse_cache_key': bse_metadata['cache_key'],
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    # Reset CGCL random streams whether BSE was trained here or loaded.
    set_seed(args.seed)
    args.bse_checkpoint_source = bse_source
    args.bse_checkpoint_sha256 = bse_sha256
    args.bse_cache_key = bse_metadata['cache_key']
    fixed_k = args.fixed_k
    print('无监督 BSE 训练完成，开始 CGCL 伪标签训练。')

    output_dir, timestamp = train_three_stage_improved(
        model=model,
        entities=entities,
        output_dir=str(output_dir),
        timestamp=timestamp,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        epochs_stage3=args.epochs_stage3,
        batch_size=args.batch_size,
        lr_backbone=1e-6,
        lr_other=5e-5,
        temperature=0.07,
        alpha=0.1,
        whitening_weight=args.whitening_weight,
        device=device,
        pseudo_update_interval=args.pseudo_update_interval,
        cluster_k_min=args.cluster_k_min,
        cluster_k_max=args.cluster_k_max,
        cluster_sample_size=args.cluster_sample_size,
        confidence_margin=args.confidence_margin,
        fixed_k=fixed_k,
        final_embedding_source=args.final_embedding_source,
        use_coarse_contrast=not args.disable_coarse,
        use_fine_contrast=not args.disable_fine,
        dynamic_pseudo_labels=not args.freeze_pseudo_labels,
        stage2_fine_weight=args.stage2_fine_weight,
        pseudo_noise_rate=args.pseudo_noise_rate,
        pseudo_noise_scope=args.pseudo_noise_scope,
        seed=args.seed,
    )

    with open(os.path.join(output_dir, 'entry_config.json'), 'w', encoding='utf-8') as file:
        entry_config = {key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()}
        json.dump(entry_config, file, ensure_ascii=False, indent=2)

    # Reload labels only after all CGCL training and checkpoint selection are finished.
    evaluation_entities, evaluation_labels = load_entities_from_csv(csv_path)
    if evaluation_entities != entities:
        raise ValueError('Dataset entity order/content changed during training; cannot score predictions')
    evaluation_labels = np.asarray(evaluation_labels)
    n_clusters = len(np.unique(evaluation_labels))
    with np.load(os.path.join(output_dir, 'final_unsupervised.npz')) as final_partition:
        predicted = final_partition['fine'].copy()
    comparison = {}
    for source, filename in (('bse_anchor', 'bse_anchor_unsupervised.npz'),
                             ('cgcl', 'cgcl_final_unsupervised.npz')):
        with np.load(os.path.join(output_dir, filename)) as partition:
            candidate = partition['fine']
            comparison[source] = {
                'n_clusters': int(len(np.unique(candidate))),
                'NMI': float(normalized_mutual_info_score(evaluation_labels, candidate)),
                'ARI': float(adjusted_rand_score(evaluation_labels, candidate)),
                'ACC': float(cluster_acc(evaluation_labels, candidate)),
            }
    with open(os.path.join(output_dir, 'embedding_comparison.json'), 'w', encoding='utf-8') as file:
        json.dump(comparison, file, ensure_ascii=False, indent=2)
    final_metrics = {
        'method': 'unsupervised_instance_bse_then_pseudo_label_cgcl',
        'embedding_source': args.final_embedding_source,
        'cluster_count_source': ('silhouette_search' if fixed_k is None else
                                 'prespecified_fixed_k'),
        'fixed_k': fixed_k,
        **comparison[args.final_embedding_source],
    }
    with open(os.path.join(output_dir, 'final_metrics.json'), 'w', encoding='utf-8') as file:
        json.dump(final_metrics, file, ensure_ascii=False, indent=2)
    print('最终聚类划分评估:', final_metrics)
    if args.skip_comparison:
        return
    print('以下保留的多方法对比使用真实类别数，并按真实指标比较方法，仅作训练后参考。')

    print(f"\n{'=' * 60}")
    print("最终评估 - 对比多种聚类方法（使用Stage3 best模型）")
    print(f"{'=' * 60}\n")

    embeddings = extract_embeddings(model, entities, batch_size=128)
    embeddings_np = embeddings.numpy()
    verbose = True
    methods = [
        'hierarchical_gap',
        'hierarchical_silhouette',
        'kmeans',
        'acrl_kmeans',
        'acrl_hierarchical_gap',
        'acrl_hierarchical_silhouette'
    ]

    print(f"\n{'=' * 70}")
    print("实验对比：余弦距离 vs 欧氏距离")
    print(f"{'=' * 70}\n")

    print("=" * 70)
    print("【实验组：余弦距离】")
    print("=" * 70)
    comparison_cosine = evaluate_clustering_comparison(
        embeddings_np,
        evaluation_labels,
        n_clusters,
        methods=methods,
        use_cosine=True,
        verbose=True
    )

    print("\n" + "=" * 70)
    print("【对照组：欧氏距离】")
    print("=" * 70)
    comparison_euclidean = evaluate_clustering_comparison(
        embeddings_np,
        evaluation_labels,
        n_clusters,
        methods=methods,
        use_cosine=False,
        verbose=True
    )

    print(f"\n{'=' * 70}")
    print("余弦距离 vs 欧氏距离 - 详细对比")
    print(f"{'=' * 70}")

    print(f"\n总体性能对比:")
    print(f"{'指标':<15} {'余弦距离':>12} {'欧氏距离':>12} {'差异':>12} {'相对提升':>12}")
    print("-" * 70)

    nmi_diff = comparison_cosine['overall_metrics']['NMI'] - comparison_euclidean['overall_metrics']['NMI']
    nmi_improve = (nmi_diff / comparison_euclidean['overall_metrics']['NMI']) * 100 if \
    comparison_euclidean['overall_metrics']['NMI'] > 0 else 0

    ari_diff = comparison_cosine['overall_metrics']['ARI'] - comparison_euclidean['overall_metrics']['ARI']
    ari_improve = (ari_diff / comparison_euclidean['overall_metrics']['ARI']) * 100 if \
    comparison_euclidean['overall_metrics']['ARI'] > 0 else 0

    acc_diff = comparison_cosine['overall_metrics']['ACC'] - comparison_euclidean['overall_metrics']['ACC']
    acc_improve = (acc_diff / comparison_euclidean['overall_metrics']['ACC']) * 100 if \
    comparison_euclidean['overall_metrics']['ACC'] > 0 else 0

    print(
        f"{'NMI':<15} {comparison_cosine['overall_metrics']['NMI']:>12.4f} {comparison_euclidean['overall_metrics']['NMI']:>12.4f} "
        f"{nmi_diff:>+12.4f} {nmi_improve:>+11.2f}%")
    print(
        f"{'ARI':<15} {comparison_cosine['overall_metrics']['ARI']:>12.4f} {comparison_euclidean['overall_metrics']['ARI']:>12.4f} "
        f"{ari_diff:>+12.4f} {ari_improve:>+11.2f}%")
    print(
        f"{'ACC':<15} {comparison_cosine['overall_metrics']['ACC']:>12.4f} {comparison_euclidean['overall_metrics']['ACC']:>12.4f} "
        f"{acc_diff:>+12.4f} {acc_improve:>+11.2f}%")

    print(f"\n分方法对比:")
    print(f"{'方法':<30} {'度量':<10} {'NMI':>10} {'ACC':>10} {'ARI':>10}")
    print("-" * 70)

    for method in methods:
        if method not in comparison_cosine['method_results'] or method not in comparison_euclidean['method_results']:
            if verbose:
                print(f"{method:<30} {'跳过':<10} {'方法未执行或失败'}")
            continue

        result_cos = comparison_cosine['method_results'][method]
        result_euc = comparison_euclidean['method_results'][method]

        print(f"{method:<30} {'余弦':<10} {result_cos['NMI']:>10.4f} "
              f"{result_cos['ACC']:>10.4f} {result_cos['ARI']:>10.4f}")
        print(f"{'':<30} {'欧氏':<10} {result_euc['NMI']:>10.4f} "
              f"{result_euc['ACC']:>10.4f} {result_euc['ARI']:>10.4f}")

        nmi_method_diff = result_cos['NMI'] - result_euc['NMI']
        acc_method_diff = result_cos['ACC'] - result_euc['ACC']
        ari_method_diff = result_cos['ARI'] - result_euc['ARI']

        print(f"{'':<30} {'差异':<10} {nmi_method_diff:>+10.4f} "
              f"{acc_method_diff:>+10.4f} {ari_method_diff:>+10.4f}")
        print("-" * 70)

    print(f"\n结论:")
    if nmi_diff > 0.01:
        print(f"   余弦距离显著优于欧氏距离 (NMI提升 {nmi_diff:.4f}, +{nmi_improve:.2f}%)")
        print(f"      推荐使用余弦距离作为最终配置")
    elif nmi_diff > 0:
        print(f"   余弦距离略优于欧氏距离 (NMI提升 {nmi_diff:.4f}, +{nmi_improve:.2f}%)")
        print(f"      余弦距离表现更好")
    elif nmi_diff > -0.01:
        print(f"   两种距离度量表现相当 (NMI差异 {nmi_diff:.4f})")
        print(f"      可以根据具体场景选择")
    else:
        print(f"   欧氏距离优于余弦距离 (NMI差异 {nmi_diff:.4f})")
        print(f"      当前数据集可能更适合欧氏距离")

    print(f"\n最佳配置: {comparison_cosine['best_method']} + {'余弦距离' if nmi_diff > 0 else '欧氏距离'}")

    print(f"\n{'=' * 60}")
    print(f"Final Metrics (最佳方法: {comparison_cosine['best_method']}, 余弦距离):")
    print(f"{'=' * 60}")
    print(f"  NMI: {comparison_cosine['overall_metrics']['NMI']:.4f}")
    print(f"  ARI: {comparison_cosine['overall_metrics']['ARI']:.4f}")
    print(f"  ACC: {comparison_cosine['overall_metrics']['ACC']:.4f}")

    print(f"\n所有方法对比 (余弦距离):")
    for method, result in comparison_cosine['method_results'].items():
        optimal_k = result.get('optimal_k', 'N/A')
        print(f"  {method:30s} | NMI={result['NMI']:.4f} | ACC={result['ACC']:.4f} | K={optimal_k}")

    active_ratio, eff_rank = evaluate_feature_collapse(embeddings)
    print(f"\n最终特征质量:")
    print(f"  活跃维度比例: {active_ratio:.2%}")
    print(f"  有效秩比例: {eff_rank:.2%}")

    comparison_path = os.path.join(output_dir, 'clustering_comparison.json')
    with open(comparison_path, 'w', encoding='utf-8') as f:
        save_data = {
            'cosine_distance': {
                'best_method': comparison_cosine['best_method'],
                'overall_metrics': {
                    'NMI': float(comparison_cosine['overall_metrics']['NMI']),
                    'ACC': float(comparison_cosine['overall_metrics']['ACC']),
                    'ARI': float(comparison_cosine['overall_metrics']['ARI'])
                },
                'method_results': {
                    method: {
                        'NMI': float(result['NMI']),
                        'ACC': float(result['ACC']),
                        'ARI': float(result['ARI']),
                        'optimal_k': int(result.get('optimal_k', 0))
                    }
                    for method, result in comparison_cosine['method_results'].items()
                }
            },

            'euclidean_distance': {
                'best_method': comparison_euclidean['best_method'],
                'overall_metrics': {
                    'NMI': float(comparison_euclidean['overall_metrics']['NMI']),
                    'ACC': float(comparison_euclidean['overall_metrics']['ACC']),
                    'ARI': float(comparison_euclidean['overall_metrics']['ARI'])
                },
                'method_results': {
                    method: {
                        'NMI': float(result['NMI']),
                        'ACC': float(result['ACC']),
                        'ARI': float(result['ARI']),
                        'optimal_k': int(result.get('optimal_k', 0))
                    }
                    for method, result in comparison_euclidean['method_results'].items()
                }
            },

            'comparison_analysis': {
                'nmi_difference': float(nmi_diff),
                'nmi_improvement_percent': float(nmi_improve),
                'ari_difference': float(ari_diff),
                'ari_improvement_percent': float(ari_improve),
                'acc_difference': float(acc_diff),
                'acc_improvement_percent': float(acc_improve),
                'better_metric': 'cosine' if nmi_diff > 0 else 'euclidean',
                'recommendation': 'cosine_distance' if nmi_diff > 0.01 else (
                    'euclidean_distance' if nmi_diff < -0.01 else 'either_works'
                )
            },

            'feature_quality': {
                'active_ratio': float(active_ratio),
                'effective_rank': float(eff_rank)
            }
        }
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n详细对比结果已保存至: {comparison_path}")
    print(f"   包含余弦距离和欧氏距离的完整对比数据")

    print(f"\n{'=' * 70}")
    print("生成聚类可视化（每个方法的最佳结果）")
    print(f"{'=' * 70}\n")

    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    print(f"可视化保存目录: {vis_dir}\n")

    print("[1/N] 可视化真实标签分布...")
    vis_path_true = os.path.join(vis_dir, f'{timestamp}_true_labels.png')

    print("   执行t-SNE降维...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, verbose=0)
    embeddings_2d = tsne.fit_transform(embeddings_np)

    plt.figure(figsize=(10, 8), dpi=150)
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                          c=evaluation_labels,
                          cmap='nipy_spectral',
                          s=20,
                          alpha=0.7,
                          edgecolors='none')
    plt.colorbar(scatter, label='Class ID (1-130)', fraction=0.046, pad=0.04)
    plt.title('True Labels (Ground Truth)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(vis_path_true, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"      已保存至: {vis_path_true}")

    print(f"\n为每个余弦距离方法生成可视化...")
    cosine_vis_paths = {}
    for idx, (method, result) in enumerate(comparison_cosine['method_results'].items(), start=2):
        print(f"\n   [{idx}/{len(comparison_cosine['method_results']) + 1}] 可视化方法: {method} (余弦距离)")

        pred_labels = result['pred_labels']
        nmi = result['NMI']
        acc = result['ACC']
        ari = result['ARI']
        optimal_k = result.get('optimal_k', len(np.unique(pred_labels)))

        vis_path = os.path.join(vis_dir, f'{timestamp}_cosine_{method}.png')
        cosine_vis_paths[method] = vis_path

        plt.figure(figsize=(10, 8), dpi=150)
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                              c=pred_labels,
                              cmap='nipy_spectral',
                              s=20,
                              alpha=0.7,
                              edgecolors='none')
        plt.colorbar(scatter, label='Predicted Cluster ID', fraction=0.046, pad=0.04)

        plt.title(f'Cosine Distance - {method}\n' +
                  f'NMI={nmi:.4f}, ACC={acc:.4f}, ARI={ari:.4f}, K={optimal_k}',
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(vis_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"      已保存至: {vis_path}")

    print(f"\n为每个欧氏距离方法生成可视化...")
    euclidean_vis_paths = {}
    for idx, (method, result) in enumerate(comparison_euclidean['method_results'].items(),
                                           start=len(comparison_cosine['method_results']) + 2):
        print(
            f"\n   [{idx}/{len(comparison_cosine['method_results']) + len(comparison_euclidean['method_results']) + 1}] 可视化方法: {method} (欧氏距离)")

        pred_labels = result['pred_labels']
        nmi = result['NMI']
        acc = result['ACC']
        ari = result['ARI']
        optimal_k = result.get('optimal_k', len(np.unique(pred_labels)))

        vis_path = os.path.join(vis_dir, f'{timestamp}_euclidean_{method}.png')
        euclidean_vis_paths[method] = vis_path

        plt.figure(figsize=(10, 8), dpi=150)
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                              c=pred_labels,
                              cmap='nipy_spectral',
                              s=20,
                              alpha=0.7,
                              edgecolors='none')
        plt.colorbar(scatter, label='Predicted Cluster ID', fraction=0.046, pad=0.04)

        plt.title(f'Euclidean Distance - {method}\n' +
                  f'NMI={nmi:.4f}, ACC={acc:.4f}, ARI={ari:.4f}, K={optimal_k}',
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(vis_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"      已保存至: {vis_path}")

    print(f"\n生成最佳结果对比可视化...")

    all_results = []
    for method, result in comparison_cosine['method_results'].items():
        all_results.append({
            'method': method,
            'distance': 'cosine',
            'nmi': result['NMI'],
            'acc': result['ACC'],
            'ari': result['ARI'],
            'pred_labels': result['pred_labels'],
            'optimal_k': result.get('optimal_k', 0)
        })

    for method, result in comparison_euclidean['method_results'].items():
        all_results.append({
            'method': method,
            'distance': 'euclidean',
            'nmi': result['NMI'],
            'acc': result['ACC'],
            'ari': result['ARI'],
            'pred_labels': result['pred_labels'],
            'optimal_k': result.get('optimal_k', 0)
        })

    all_results_sorted_by_nmi = sorted(all_results, key=lambda x: x['nmi'], reverse=True)
    best_nmi = all_results_sorted_by_nmi[0]

    all_results_sorted_by_acc = sorted(all_results, key=lambda x: x['acc'], reverse=True)
    best_acc = all_results_sorted_by_acc[0]

    print(f"\n   NMI最高: {best_nmi['method']} ({best_nmi['distance']}) = {best_nmi['nmi']:.4f}")
    print(f"   ACC最高: {best_acc['method']} ({best_acc['distance']}) = {best_acc['acc']:.4f}")

    comparison_vis_path = os.path.join(vis_dir, f'{timestamp}_best_comparison.png')

    if best_nmi['method'] == best_acc['method'] and best_nmi['distance'] == best_acc['distance']:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=150)

        scatter1 = axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                   c=evaluation_labels, cmap='nipy_spectral', s=20, alpha=0.7, edgecolors='none')
        axes[0].set_title('True Labels (Ground Truth)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('t-SNE Dimension 1', fontsize=11)
        axes[0].set_ylabel('t-SNE Dimension 2', fontsize=11)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        cbar1 = plt.colorbar(scatter1, ax=axes[0], fraction=0.046, pad=0.04)
        cbar1.set_label('Class ID', fontsize=10)

        scatter2 = axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                   c=best_nmi['pred_labels'], cmap='nipy_spectral', s=20, alpha=0.7, edgecolors='none')
        axes[1].set_title(f"Predicted Labels\n" ,
                          fontsize=14, fontweight='bold')
        axes[1].set_xlabel('t-SNE Dimension 1', fontsize=11)
        axes[1].set_ylabel('t-SNE Dimension 2', fontsize=11)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        cbar2 = plt.colorbar(scatter2, ax=axes[1], fraction=0.046, pad=0.04)
        cbar2.set_label('Cluster ID', fontsize=10)

    else:
        fig, axes = plt.subplots(1, 3, figsize=(24, 7), dpi=150)

        scatter1 = axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                   c=evaluation_labels, cmap='nipy_spectral', s=20, alpha=0.7, edgecolors='none')
        axes[0].set_title('True Labels (Ground Truth)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('t-SNE Dimension 1', fontsize=11)
        axes[0].set_ylabel('t-SNE Dimension 2', fontsize=11)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        cbar1 = plt.colorbar(scatter1, ax=axes[0], fraction=0.046, pad=0.04)
        cbar1.set_label('Class ID', fontsize=10)

        scatter2 = axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                   c=best_nmi['pred_labels'], cmap='nipy_spectral', s=20, alpha=0.7, edgecolors='none')
        axes[1].set_title(f"Best NMI: {best_nmi['method']} ({best_nmi['distance']})\n" +
                          f"NMI={best_nmi['nmi']:.4f}, ACC={best_nmi['acc']:.4f}, K={best_nmi['optimal_k']}",
                          fontsize=14, fontweight='bold')
        axes[1].set_xlabel('t-SNE Dimension 1', fontsize=11)
        axes[1].set_ylabel('t-SNE Dimension 2', fontsize=11)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        cbar2 = plt.colorbar(scatter2, ax=axes[1], fraction=0.046, pad=0.04)
        cbar2.set_label('Cluster ID', fontsize=10)

        scatter3 = axes[2].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                   c=best_acc['pred_labels'], cmap='nipy_spectral', s=20, alpha=0.7, edgecolors='none')
        axes[2].set_title(f"Best ACC: {best_acc['method']} ({best_acc['distance']})\n" +
                          f"NMI={best_acc['nmi']:.4f}, ACC={best_acc['acc']:.4f}, K={best_acc['optimal_k']}",
                          fontsize=14, fontweight='bold')
        axes[2].set_xlabel('t-SNE Dimension 1', fontsize=11)
        axes[2].set_ylabel('t-SNE Dimension 2', fontsize=11)
        axes[2].grid(True, alpha=0.3, linestyle='--')
        cbar3 = plt.colorbar(scatter3, ax=axes[2], fraction=0.046, pad=0.04)
        cbar3.set_label('Cluster ID', fontsize=10)

    plt.tight_layout()
    plt.savefig(comparison_vis_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"   最佳对比可视化已保存至: {comparison_vis_path}")

    print(f"\n{'=' * 70}")
    print("所有可视化已生成!")
    print(f"{'=' * 70}")
    print(f"   可视化目录: {vis_dir}")
    print(f"   生成的可视化文件:")
    print(f"      1. {timestamp}_true_labels.png (真实标签)")
    print(f"\n      余弦距离方法:")
    for idx, method in enumerate(comparison_cosine['method_results'].keys(), start=2):
        print(f"      {idx}. {timestamp}_cosine_{method}.png")
    print(f"\n      欧氏距离方法:")
    start_idx = len(comparison_cosine['method_results']) + 2
    for idx, method in enumerate(comparison_euclidean['method_results'].keys(), start=start_idx):
        print(f"      {idx}. {timestamp}_euclidean_{method}.png")
    print(
        f"\n      {len(comparison_cosine['method_results']) + len(comparison_euclidean['method_results']) + 2}. {timestamp}_best_comparison.png (最佳结果对比)")
    print(f"\n   每个方法显示其自己的NMI、ACC、ARI指标")
    print(f"   NMI最高: {best_nmi['method']} ({best_nmi['distance']}) = {best_nmi['nmi']:.4f}")
    print(f"   ACC最高: {best_acc['method']} ({best_acc['distance']}) = {best_acc['acc']:.4f}")
    print(f"{'=' * 70}\n")

    print(f"\n{'='*60}")
    print("所有流程完成!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
