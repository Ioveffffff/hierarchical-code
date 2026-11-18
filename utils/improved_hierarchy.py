# """
# 改进的层次标签构建 + 数据增强 - 完整版
#
# 主要功能:
# 1. 扩充关键词规则（从4个扩展到15+个）
# 2. 添加混合策略（规则+聚类）
# 3. 统计粗类分布，确保平衡性
# 4. 数据增强函数（文本扰动）
# """
#
# import numpy as np
# from sklearn.cluster import KMeans
#
#
# # =====================================================
# # 数据增强函数
# # =====================================================
# def augment_entity(entity_str, aug_prob=0.3):
#     """
#     实体文本增强（用于训练时的数据扰动）
#
#     参数:
#         entity_str: 原始实体文本
#         aug_prob: 每种增强方式的概率
#
#     增强方式:
#         1. 同义词替换
#         2. 随机插入
#         3. 随机删除（保留至少50%字符）
#
#     返回:
#         增强后的文本
#     """
#     import random
#
#     # 如果是纯数字或很短的文本，不做增强
#     if len(str(entity_str)) < 3 or str(entity_str).replace('.', '').replace('-', '').isdigit():
#         return entity_str
#
#     text = str(entity_str)
#
#     # 1. 同义词替换（针对电机领域）
#     synonyms = {
#         '电机': ['电动机', '马达', 'motor'],
#         '电压': ['电势', 'voltage'],
#         '电流': ['电荷流动', 'current'],
#         '转速': ['转动速度', '旋转速度', 'rpm'],
#         '功率': ['输出功率', 'power'],
#         '故障': ['失效', '损坏', 'fault'],
#         '控制': ['调节', '调控', 'control'],
#         '定子': ['静止部分', 'stator'],
#         '转子': ['旋转部分', 'rotor']
#     }
#
#     if random.random() < aug_prob:
#         for original, replacements in synonyms.items():
#             if original in text and random.random() < 0.5:
#                 text = text.replace(original, random.choice(replacements), 1)
#
#     # 2. 随机插入噪声（添加空格或标点）
#     if random.random() < aug_prob * 0.5:
#         insert_pos = random.randint(0, len(text))
#         noise = random.choice([' ', '-', '_'])
#         text = text[:insert_pos] + noise + text[insert_pos:]
#
#     # 3. 随机删除（但保留至少50%的字符）
#     if random.random() < aug_prob * 0.3 and len(text) > 5:
#         keep_ratio = random.uniform(0.5, 0.9)
#         keep_len = int(len(text) * keep_ratio)
#         start_pos = random.randint(0, len(text) - keep_len)
#         text = text[start_pos:start_pos + keep_len]
#
#     return text
#
#
# # =====================================================
# # 层次标签构建函数
# # =====================================================
# def build_hierarchical_labels_v2(entities, fine_labels):
#     """
#     改进版本 - 扩充关键词规则
#
#     原始问题：
#         规则太简单，大量样本被分到"其他"类(class 6)
#
#     改进方案：
#         每个粗类增加15+个关键词，覆盖更多样本
#     """
#     coarse_labels = []
#
#     # ✅ 扩充后的规则（每类15+关键词）
#     coarse_rules = {
#         0: [  # 电机类型
#             "电机", "电动机", "发电机", "马达", "motor",
#             "异步", "同步", "直流", "交流", "伺服",
#             "步进", "无刷", "有刷", "永磁", "感应"
#         ],
#         1: [  # 电气参数
#             "电压", "电流", "功率", "电阻", "频率",
#             "相数", "电能", "电荷", "电场", "磁场",
#             "电容", "电感", "阻抗", "导纳", "电势"
#         ],
#         2: [  # 机械参数
#             "转速", "扭矩", "负载", "转矩", "惯量",
#             "速度", "加速", "减速", "力矩", "功",
#             "能量", "动能", "势能", "机械", "运动"
#         ],
#         3: [  # 控制方式
#             "控制", "调速", "启动", "制动", "保护",
#             "调节", "开关", "继电器", "接触器", "变频",
#             "PWM", "PID", "反馈", "闭环", "开环"
#         ],
#         4: [  # 部件
#             "定子", "转子", "绕组", "轴承", "外壳",
#             "端盖", "机座", "风扇", "铁芯", "线圈",
#             "槽", "极", "轴", "联轴器", "底座"
#         ],
#         5: [  # 故障
#             "故障", "损坏", "磨损", "过热", "短路",
#             "断路", "漏电", "振动", "噪音", "异常",
#             "失效", "老化", "击穿", "烧毁", "卡死"
#         ],
#         6: []  # 其他（默认类）
#     }
#
#     # 统计匹配情况
#     match_stats = {i: 0 for i in range(7)}
#
#     for entity in entities:
#         entity_str = str(entity).lower()  # 转小写增加匹配率
#         matched = False
#
#         for coarse_id, keywords in coarse_rules.items():
#             if any(kw in entity_str for kw in keywords):
#                 coarse_labels.append(coarse_id)
#                 match_stats[coarse_id] += 1
#                 matched = True
#                 break
#
#         if not matched:
#             coarse_labels.append(6)
#             match_stats[6] += 1
#
#     coarse_labels = np.array(coarse_labels)
#
#     # 打印统计信息
#     print(f"\n✅ 层次标签统计 (改进版):")
#     print(f"   粗粒度: {len(np.unique(coarse_labels))} 类")
#     print(f"   细粒度: {len(np.unique(fine_labels))} 类")
#     print(f"\n   各粗类分布:")
#
#     class_names = ["电机类型", "电气参数", "机械参数", "控制方式", "部件", "故障", "其他"]
#     for coarse_id in range(7):
#         mask = (coarse_labels == coarse_id)
#         n_samples = mask.sum()
#         fine_in_coarse = len(np.unique(fine_labels[mask]))
#         percentage = n_samples / len(coarse_labels) * 100
#         print(f"   粗类{coarse_id} ({class_names[coarse_id]:6s}): "
#               f"{n_samples:4d} 样本 ({percentage:5.1f}%), {fine_in_coarse:3d} 细类")
#
#     # ⚠️ 警告：如果"其他"类占比超过30%，建议使用混合策略
#     other_ratio = match_stats[6] / len(entities)
#     if other_ratio > 0.3:
#         print(f"\n⚠️  警告: '其他'类占比过高 ({other_ratio:.1%})")
#         print(f"   建议使用 build_hierarchical_labels_hybrid() 混合策略")
#
#     return coarse_labels
#
#
# def build_hierarchical_labels_hybrid(entities, fine_labels, embeddings,
#                                      n_coarse_clusters=7, rule_weight=0.6):
#     """
#     混合策略 - 规则 + 聚类
#
#     适用场景：
#         - 规则匹配率 < 70%（即"其他"类 > 30%）
#         - 希望自动发现隐藏的粗类结构
#
#     原理：
#         1. 先用规则分配有把握的样本
#         2. 对"其他"类样本用聚类重新分配
#         3. 最终粗类数 = 规则类数 + 聚类发现的新类
#     """
#     from sklearn.cluster import KMeans
#
#     print(f"\n🔄 使用混合策略构建层次标签...")
#
#     # Step 1: 规则匹配
#     rule_labels = build_hierarchical_labels_v2(entities, fine_labels)
#
#     # Step 2: 对"其他"类用聚类细分
#     other_mask = (rule_labels == 6)
#     n_other = other_mask.sum()
#
#     if n_other > 100:  # 只有"其他"类样本足够多才聚类
#         print(f"\n   正在对 {n_other} 个'其他'类样本进行聚类...")
#
#         other_embeddings = embeddings[other_mask]
#
#         # 聚类数 = min(3, 其他类样本数/50)
#         n_sub_clusters = min(3, max(2, n_other // 50))
#
#         kmeans = KMeans(n_clusters=n_sub_clusters, random_state=42, n_init=20)
#         sub_labels = kmeans.fit_predict(other_embeddings)
#
#         # 重新分配"其他"类样本
#         # 原始规则类: 0-5, 新聚类类: 7, 8, 9...
#         hybrid_labels = rule_labels.copy()
#         hybrid_labels[other_mask] = 7 + sub_labels  # 从7开始编号
#
#         print(f"   ✅ '其他'类被细分为 {n_sub_clusters} 个子类")
#     else:
#         hybrid_labels = rule_labels
#         print(f"   ℹ️ '其他'类样本较少 ({n_other})，保持不变")
#
#     # 统计最终分布
#     print(f"\n✅ 混合策略最终统计:")
#     print(f"   粗粒度: {len(np.unique(hybrid_labels))} 类")
#     print(f"   细粒度: {len(np.unique(fine_labels))} 类")
#
#     for coarse_id in np.unique(hybrid_labels):
#         mask = (hybrid_labels == coarse_id)
#         n_samples = mask.sum()
#         fine_in_coarse = len(np.unique(fine_labels[mask]))
#         percentage = n_samples / len(hybrid_labels) * 100
#
#         if coarse_id < 6:
#             class_names = ["电机类型", "电气参数", "机械参数", "控制方式", "部件", "故障"]
#             name = class_names[coarse_id]
#         elif coarse_id == 6:
#             name = "其他(未分)"
#         else:
#             name = f"聚类子类{coarse_id - 6}"
#
#         print(f"   粗类{coarse_id} ({name:10s}): "
#               f"{n_samples:4d} 样本 ({percentage:5.1f}%), {fine_in_coarse:3d} 细类")
#
#     return hybrid_labels
#
#
# def build_hierarchical_labels_auto(embeddings, fine_labels, n_coarse_clusters=7):
#     """
#     纯聚类策略（保留原有功能）
#
#     适用场景：
#         - 完全不了解数据的语义结构
#         - 希望让模型自动发现层次
#     """
#     from sklearn.cluster import KMeans
#
#     print(f"\n🔄 使用纯聚类策略构建层次标签...")
#
#     kmeans = KMeans(n_clusters=n_coarse_clusters, random_state=42, n_init=20)
#     coarse_labels = kmeans.fit_predict(embeddings)
#
#     print(f"\n✅ 自动构建层次标签:")
#     print(f"   粗粒度: {n_coarse_clusters} 类")
#     print(f"   细粒度: {len(np.unique(fine_labels))} 类")
#
#     for coarse_id in range(n_coarse_clusters):
#         mask = (coarse_labels == coarse_id)
#         n_samples = mask.sum()
#         fine_in_coarse = len(np.unique(fine_labels[mask]))
#         percentage = n_samples / len(coarse_labels) * 100
#         print(f"   粗类{coarse_id}: {n_samples:4d} 样本 ({percentage:5.1f}%), "
#               f"{fine_in_coarse:3d} 细类")
#
#     return coarse_labels
#
#
# # =====================================================
# # 辅助函数：评估粗类质量
# # =====================================================
# def evaluate_coarse_quality(coarse_labels, fine_labels):
#     """
#     评估粗类标签质量
#
#     指标：
#     1. 平衡性：各粗类样本数方差（越小越好）
#     2. 纯度：每个粗类内细类数量（适中为好）
#     3. 覆盖率：被正确分配的样本比例
#     """
#     n_coarse = len(np.unique(coarse_labels))
#
#     # 1. 平衡性
#     class_counts = [np.sum(coarse_labels == i) for i in range(n_coarse)]
#     balance_score = 1.0 - (np.std(class_counts) / (np.mean(class_counts) + 1e-8))
#
#     # 2. 纯度（每个粗类平均包含多少细类）
#     fine_counts = []
#     for i in range(n_coarse):
#         mask = (coarse_labels == i)
#         fine_in_coarse = len(np.unique(fine_labels[mask]))
#         fine_counts.append(fine_in_coarse)
#     avg_fine_per_coarse = np.mean(fine_counts)
#
#     # 3. 理想状态：每个粗类包含 总细类数/粗类数 个细类
#     ideal_fine_per_coarse = len(np.unique(fine_labels)) / n_coarse
#     purity_score = 1.0 - abs(avg_fine_per_coarse - ideal_fine_per_coarse) / ideal_fine_per_coarse
#
#     print(f"\n📊 粗类质量评估:")
#     print(f"   平衡性得分: {balance_score:.3f} (越接近1越好)")
#     print(f"   每粗类平均细类数: {avg_fine_per_coarse:.1f} (理想值: {ideal_fine_per_coarse:.1f})")
#     print(f"   纯度得分: {purity_score:.3f} (越接近1越好)")
#
#     return {
#         'balance_score': balance_score,
#         'purity_score': purity_score,
#         'avg_fine_per_coarse': avg_fine_per_coarse
#     }
#
#
# # =====================================================
# # 使用示例
# # =====================================================
# if __name__ == "__main__":
#     # 模拟数据
#     entities = ["异步电机", "额定电压", "转速", "启动方式", "定子", "过热故障", "未知术语"] * 100
#     fine_labels = np.random.randint(0, 50, len(entities))
#     embeddings = np.random.randn(len(entities), 128)
#
#     # 测试数据增强
#     print("=== 数据增强测试 ===")
#     test_entities = ["异步电机", "额定电压380V", "转速1500rpm"]
#     for ent in test_entities:
#         aug_ent = augment_entity(ent, aug_prob=0.5)
#         print(f"原始: {ent:20s} -> 增强: {aug_ent}")
#
#     print("\n" + "="*60 + "\n")
#
#     # 方法1: 改进的规则匹配
#     coarse_v2 = build_hierarchical_labels_v2(entities, fine_labels)
#
#     # 方法2: 混合策略（推荐）
#     coarse_hybrid = build_hierarchical_labels_hybrid(entities, fine_labels, embeddings)
#
#     # 方法3: 纯聚类
#     coarse_auto = build_hierarchical_labels_auto(embeddings, fine_labels)
#
#     # 评估质量
#     evaluate_coarse_quality(coarse_hybrid, fine_labels)

# """
# 改进的层次标签构建 + 数据增强 - 完整版
#
# 主要改进:
# 1. build_hierarchical_labels_auto() - 使用Gap Statistic自动确定最佳粗类数量
# 2. 避免"其他"类占比过高（原来49.2%）
# 3. 扩充关键词规则（从4个扩展到15+个）
# 4. 数据增强函数（文本扰动）
# """
#
# import numpy as np
# from sklearn.cluster import KMeans
#
#
# # =====================================================
# # 数据增强函数
# # =====================================================
# def augment_entity(entity_str, aug_prob=0.3):
#     """
#     实体文本增强（用于训练时的数据扰动）
#
#     参数:
#         entity_str: 原始实体文本
#         aug_prob: 每种增强方式的概率
#
#     增强方式:
#         1. 同义词替换
#         2. 随机插入
#         3. 随机删除（保留至少50%字符）
#
#     返回:
#         增强后的文本
#     """
#     import random
#
#     # 如果是纯数字或很短的文本，不做增强
#     if len(str(entity_str)) < 3 or str(entity_str).replace('.', '').replace('-', '').isdigit():
#         return entity_str
#
#     text = str(entity_str)
#
#     # 1. 同义词替换（针对电机领域）
#     synonyms = {
#         '电机': ['电动机', '马达', 'motor'],
#         '电压': ['电势', 'voltage'],
#         '电流': ['电荷流动', 'current'],
#         '转速': ['转动速度', '旋转速度', 'rpm'],
#         '功率': ['输出功率', 'power'],
#         '故障': ['失效', '损坏', 'fault'],
#         '控制': ['调节', '调控', 'control'],
#         '定子': ['静止部分', 'stator'],
#         '转子': ['旋转部分', 'rotor']
#     }
#
#     if random.random() < aug_prob:
#         for original, replacements in synonyms.items():
#             if original in text and random.random() < 0.5:
#                 text = text.replace(original, random.choice(replacements), 1)
#
#     # 2. 随机插入噪声（添加空格或标点）
#     if random.random() < aug_prob * 0.5:
#         insert_pos = random.randint(0, len(text))
#         noise = random.choice([' ', '-', '_'])
#         text = text[:insert_pos] + noise + text[insert_pos:]
#
#     # 3. 随机删除（但保留至少50%的字符）
#     if random.random() < aug_prob * 0.3 and len(text) > 5:
#         keep_ratio = random.uniform(0.5, 0.9)
#         keep_len = int(len(text) * keep_ratio)
#         start_pos = random.randint(0, len(text) - keep_len)
#         text = text[start_pos:start_pos + keep_len]
#
#     return text
#
#
# # =====================================================
# # Gap Statistic 计算（用于自动确定最佳簇数）
# # =====================================================
# def compute_gap_statistic(embeddings, max_k=15, n_refs=10, random_state=42):
#     """
#     使用Gap Statistic自动确定最佳聚类数
#
#     原理:
#         Gap(k) = E[log(W_k)] - log(W_k)
#         其中 W_k 是簇内距离和, E[log(W_k)] 是随机数据的期望
#
#     返回:
#         optimal_k: 最佳簇数
#         gaps: 每个k的gap值
#     """
#     from sklearn.cluster import KMeans
#
#     np.random.seed(random_state)
#
#     n_samples, n_features = embeddings.shape
#     gaps = []
#
#     # 计算数据的边界
#     mins = embeddings.min(axis=0)
#     maxs = embeddings.max(axis=0)
#
#     print(f"\n🔍 Gap Statistic 自动确定最佳簇数...")
#     print(f"   测试范围: k=2 到 k={max_k}")
#
#     for k in range(2, max_k + 1):
#         # 1. 真实数据的聚类
#         kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
#         labels = kmeans.fit_predict(embeddings)
#
#         # 计算簇内距离和 W_k
#         W_k = 0.0
#         for i in range(k):
#             cluster_points = embeddings[labels == i]
#             if len(cluster_points) > 0:
#                 center = cluster_points.mean(axis=0)
#                 W_k += np.sum((cluster_points - center) ** 2)
#
#         # 2. 参考数据的聚类（随机生成）
#         ref_W_ks = []
#         for _ in range(n_refs):
#             # 在数据边界内生成随机数据
#             ref_data = np.random.uniform(mins, maxs, size=(n_samples, n_features))
#
#             ref_kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
#             ref_labels = ref_kmeans.fit_predict(ref_data)
#
#             ref_W_k = 0.0
#             for i in range(k):
#                 ref_cluster = ref_data[ref_labels == i]
#                 if len(ref_cluster) > 0:
#                     ref_center = ref_cluster.mean(axis=0)
#                     ref_W_k += np.sum((ref_cluster - ref_center) ** 2)
#
#             ref_W_ks.append(np.log(ref_W_k + 1e-10))
#
#         # 3. 计算Gap
#         gap = np.mean(ref_W_ks) - np.log(W_k + 1e-10)
#         gaps.append(gap)
#
#         if k % 3 == 0 or k == max_k:
#             print(f"   k={k:2d}: Gap={gap:.4f}")
#
#     # 找到第一个满足 Gap(k) >= Gap(k+1) - s_{k+1} 的k
#     gaps = np.array(gaps)
#
#     # 简化版: 找到gap最大的k
#     optimal_k = np.argmax(gaps) + 2  # +2 因为从k=2开始
#
#     print(f"   ✅ 最佳簇数: k={optimal_k} (Gap={gaps[optimal_k - 2]:.4f})")
#
#     return optimal_k, gaps
#
#
# # =====================================================
# # 层次标签构建函数
# # =====================================================
# def build_hierarchical_labels_v2(entities, fine_labels):
#     """
#     改进版本 - 扩充关键词规则
#
#     原始问题：
#         规则太简单，大量样本被分到"其他"类(class 6)
#
#     改进方案：
#         每个粗类增加15+个关键词，覆盖更多样本
#     """
#     coarse_labels = []
#
#     # ✅ 扩充后的规则（每类15+关键词）
#     coarse_rules = {
#         0: [  # 电机类型
#             "电机", "电动机", "发电机", "马达", "motor",
#             "异步", "同步", "直流", "交流", "伺服",
#             "步进", "无刷", "有刷", "永磁", "感应"
#         ],
#         1: [  # 电气参数
#             "电压", "电流", "功率", "电阻", "频率",
#             "相数", "电能", "电荷", "电场", "磁场",
#             "电容", "电感", "阻抗", "导纳", "电势"
#         ],
#         2: [  # 机械参数
#             "转速", "扭矩", "负载", "转矩", "惯量",
#             "速度", "加速", "减速", "力矩", "功",
#             "能量", "动能", "势能", "机械", "运动"
#         ],
#         3: [  # 控制方式
#             "控制", "调速", "启动", "制动", "保护",
#             "调节", "开关", "继电器", "接触器", "变频",
#             "PWM", "PID", "反馈", "闭环", "开环"
#         ],
#         4: [  # 部件
#             "定子", "转子", "绕组", "轴承", "外壳",
#             "端盖", "机座", "风扇", "铁芯", "线圈",
#             "槽", "极", "轴", "联轴器", "底座"
#         ],
#         5: [  # 故障
#             "故障", "损坏", "磨损", "过热", "短路",
#             "断路", "漏电", "振动", "噪音", "异常",
#             "失效", "老化", "击穿", "烧毁", "卡死"
#         ],
#         6: []  # 其他（默认类）
#     }
#
#     # 统计匹配情况
#     match_stats = {i: 0 for i in range(7)}
#
#     for entity in entities:
#         entity_str = str(entity).lower()  # 转小写增加匹配率
#         matched = False
#
#         for coarse_id, keywords in coarse_rules.items():
#             if any(kw in entity_str for kw in keywords):
#                 coarse_labels.append(coarse_id)
#                 match_stats[coarse_id] += 1
#                 matched = True
#                 break
#
#         if not matched:
#             coarse_labels.append(6)
#             match_stats[6] += 1
#
#     coarse_labels = np.array(coarse_labels)
#
#     # 打印统计信息
#     print(f"\n✅ 层次标签统计 (改进版):")
#     print(f"   粗粒度: {len(np.unique(coarse_labels))} 类")
#     print(f"   细粒度: {len(np.unique(fine_labels))} 类")
#     print(f"\n   各粗类分布:")
#
#     class_names = ["电机类型", "电气参数", "机械参数", "控制方式", "部件", "故障", "其他"]
#     for coarse_id in range(7):
#         mask = (coarse_labels == coarse_id)
#         n_samples = mask.sum()
#         fine_in_coarse = len(np.unique(fine_labels[mask]))
#         percentage = n_samples / len(coarse_labels) * 100
#         print(f"   粗类{coarse_id} ({class_names[coarse_id]:6s}): "
#               f"{n_samples:4d} 样本 ({percentage:5.1f}%), {fine_in_coarse:3d} 细类")
#
#     # ⚠️ 警告：如果"其他"类占比超过30%，建议使用混合策略
#     other_ratio = match_stats[6] / len(entities)
#     if other_ratio > 0.3:
#         print(f"\n⚠️  警告: '其他'类占比过高 ({other_ratio:.1%})")
#         print(f"   建议使用 build_hierarchical_labels_hybrid() 混合策略")
#
#     return coarse_labels
#
#
# def build_hierarchical_labels_hybrid(entities, fine_labels, embeddings,
#                                      n_coarse_clusters=7, rule_weight=0.6):
#     """
#     混合策略 - 规则 + 聚类
#
#     适用场景：
#         - 规则匹配率 < 70%（即"其他"类 > 30%）
#         - 希望自动发现隐藏的粗类结构
#
#     原理：
#         1. 先用规则分配有把握的样本
#         2. 对"其他"类样本用聚类重新分配
#         3. 最终粗类数 = 规则类数 + 聚类发现的新类
#     """
#     from sklearn.cluster import KMeans
#
#     print(f"\n🔄 使用混合策略构建层次标签...")
#
#     # Step 1: 规则匹配
#     rule_labels = build_hierarchical_labels_v2(entities, fine_labels)
#
#     # Step 2: 对"其他"类用聚类细分
#     other_mask = (rule_labels == 6)
#     n_other = other_mask.sum()
#
#     if n_other > 100:  # 只有"其他"类样本足够多才聚类
#         print(f"\n   正在对 {n_other} 个'其他'类样本进行聚类...")
#
#         other_embeddings = embeddings[other_mask]
#
#         # 聚类数 = min(3, 其他类样本数/50)
#         n_sub_clusters = min(3, max(2, n_other // 50))
#
#         kmeans = KMeans(n_clusters=n_sub_clusters, random_state=42, n_init=20)
#         sub_labels = kmeans.fit_predict(other_embeddings)
#
#         # 重新分配"其他"类样本
#         # 原始规则类: 0-5, 新聚类类: 7, 8, 9...
#         hybrid_labels = rule_labels.copy()
#         hybrid_labels[other_mask] = 7 + sub_labels  # 从7开始编号
#
#         print(f"   ✅ '其他'类被细分为 {n_sub_clusters} 个子类")
#     else:
#         hybrid_labels = rule_labels
#         print(f"   ℹ️ '其他'类样本较少 ({n_other})，保持不变")
#
#     # 统计最终分布
#     print(f"\n✅ 混合策略最终统计:")
#     print(f"   粗粒度: {len(np.unique(hybrid_labels))} 类")
#     print(f"   细粒度: {len(np.unique(fine_labels))} 类")
#
#     for coarse_id in np.unique(hybrid_labels):
#         mask = (hybrid_labels == coarse_id)
#         n_samples = mask.sum()
#         fine_in_coarse = len(np.unique(fine_labels[mask]))
#         percentage = n_samples / len(hybrid_labels) * 100
#
#         if coarse_id < 6:
#             class_names = ["电机类型", "电气参数", "机械参数", "控制方式", "部件", "故障"]
#             name = class_names[coarse_id]
#         elif coarse_id == 6:
#             name = "其他(未分)"
#         else:
#             name = f"聚类子类{coarse_id - 6}"
#
#         print(f"   粗类{coarse_id} ({name:10s}): "
#               f"{n_samples:4d} 样本 ({percentage:5.1f}%), {fine_in_coarse:3d} 细类")
#
#     return hybrid_labels
#
#
# def build_hierarchical_labels_auto(embeddings, fine_labels, max_k=15):
#     """
#     🔥 改进版 - 使用Gap Statistic自动确定最佳粗类数量
#
#     改进点:
#         1. 自动确定最佳簇数（不再固定为7）
#         2. 避免"其他"类占比过高
#         3. 使用Gap Statistic科学选择簇数
#
#     适用场景：
#         - 完全不了解数据的语义结构
#         - 希望让模型自动发现最优层次
#         - 避免人为设定簇数带来的偏差
#     """
#     from sklearn.cluster import KMeans
#
#     print(f"\n🔄 使用自适应Gap Statistic策略构建层次标签...")
#
#     # 🔥 Step 1: 使用Gap Statistic自动确定最佳簇数
#     optimal_k, gaps = compute_gap_statistic(embeddings, max_k=max_k)
#
#     print(f"\n✅ 自动确定的最佳粗类数: {optimal_k}")
#
#     # Step 2: 使用最佳簇数进行聚类
#     kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
#     coarse_labels = kmeans.fit_predict(embeddings)
#
#     # Step 3: 统计分布
#     print(f"\n✅ 自动构建层次标签:")
#     print(f"   粗粒度: {optimal_k} 类 (自动确定)")
#     print(f"   细粒度: {len(np.unique(fine_labels))} 类")
#     print(f"\n   各粗类分布:")
#
#     for coarse_id in range(optimal_k):
#         mask = (coarse_labels == coarse_id)
#         n_samples = mask.sum()
#         fine_in_coarse = len(np.unique(fine_labels[mask]))
#         percentage = n_samples / len(coarse_labels) * 100
#         print(f"   粗类{coarse_id}: {n_samples:4d} 样本 ({percentage:5.1f}%), "
#               f"{fine_in_coarse:3d} 细类")
#
#     # Step 4: 检查平衡性
#     class_counts = [np.sum(coarse_labels == i) for i in range(optimal_k)]
#     max_ratio = max(class_counts) / len(coarse_labels)
#     min_ratio = min(class_counts) / len(coarse_labels)
#
#     print(f"\n📊 粗类平衡性:")
#     print(f"   最大类占比: {max_ratio:.1%}")
#     print(f"   最小类占比: {min_ratio:.1%}")
#
#     if max_ratio > 0.4:
#         print(f"   ⚠️  警告: 存在过大的粗类（>{max_ratio:.1%}），可能影响层次学习效果")
#     else:
#         print(f"   ✅ 粗类分布较为均衡")
#
#     return coarse_labels
#
#
# # =====================================================
# # 辅助函数：评估粗类质量
# # =====================================================
# def evaluate_coarse_quality(coarse_labels, fine_labels):
#     """
#     评估粗类标签质量
#
#     指标：
#     1. 平衡性：各粗类样本数方差（越小越好）
#     2. 纯度：每个粗类内细类数量（适中为好）
#     3. 覆盖率：被正确分配的样本比例
#     """
#     n_coarse = len(np.unique(coarse_labels))
#
#     # 1. 平衡性
#     class_counts = [np.sum(coarse_labels == i) for i in range(n_coarse)]
#     balance_score = 1.0 - (np.std(class_counts) / (np.mean(class_counts) + 1e-8))
#
#     # 2. 纯度（每个粗类平均包含多少细类）
#     fine_counts = []
#     for i in range(n_coarse):
#         mask = (coarse_labels == i)
#         fine_in_coarse = len(np.unique(fine_labels[mask]))
#         fine_counts.append(fine_in_coarse)
#     avg_fine_per_coarse = np.mean(fine_counts)
#
#     # 3. 理想状态：每个粗类包含 总细类数/粗类数 个细类
#     ideal_fine_per_coarse = len(np.unique(fine_labels)) / n_coarse
#     purity_score = 1.0 - abs(avg_fine_per_coarse - ideal_fine_per_coarse) / ideal_fine_per_coarse
#
#     print(f"\n📊 粗类质量评估:")
#     print(f"   平衡性得分: {balance_score:.3f} (越接近1越好)")
#     print(f"   每粗类平均细类数: {avg_fine_per_coarse:.1f} (理想值: {ideal_fine_per_coarse:.1f})")
#     print(f"   纯度得分: {purity_score:.3f} (越接近1越好)")
#
#     return {
#         'balance_score': balance_score,
#         'purity_score': purity_score,
#         'avg_fine_per_coarse': avg_fine_per_coarse
#     }
#
#
# # =====================================================
# # 使用示例
# # =====================================================
# if __name__ == "__main__":
#     # 模拟数据
#     entities = ["异步电机", "额定电压", "转速", "启动方式", "定子", "过热故障", "未知术语"] * 100
#     fine_labels = np.random.randint(0, 50, len(entities))
#     embeddings = np.random.randn(len(entities), 128)
#
#     # 测试数据增强
#     print("=== 数据增强测试 ===")
#     test_entities = ["异步电机", "额定电压380V", "转速1500rpm"]
#     for ent in test_entities:
#         aug_ent = augment_entity(ent, aug_prob=0.5)
#         print(f"原始: {ent:20s} -> 增强: {aug_ent}")
#
#     print("\n" + "=" * 60 + "\n")
#
#     # 方法1: 改进的规则匹配
#     coarse_v2 = build_hierarchical_labels_v2(entities, fine_labels)
#
#     # 方法2: 混合策略（推荐）
#     coarse_hybrid = build_hierarchical_labels_hybrid(entities, fine_labels, embeddings)
#
#     # 方法3: 自适应Gap Statistic（🔥 新增）
#     coarse_auto = build_hierarchical_labels_auto(embeddings, fine_labels, max_k=15)
#
#     # 评估质量
#     print("\n" + "=" * 60)
#     print("🔬 质量对比")
#     print("=" * 60)
#
#     print("\n【规则匹配方法】")
#     evaluate_coarse_quality(coarse_v2, fine_labels)
#
#     print("\n【混合策略方法】")
#     evaluate_coarse_quality(coarse_hybrid, fine_labels)
#
#     print("\n【自适应Gap Statistic方法】 🔥")
#     evaluate_coarse_quality(coarse_auto, fine_labels)


# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import normalize
#
#
# # =====================================================
# # 🔥 新增：L2归一化函数（关键修改）
# # =====================================================
# def normalize_embeddings(embeddings):
#     """
#     L2归一化embeddings，使其适用于余弦距离
#
#     原理：
#         归一化后，欧氏距离 ∝ 余弦距离
#         因此KMeans在归一化数据上相当于使用余弦距离
#
#     参数:
#         embeddings: (n_samples, n_features)
#
#     返回:
#         normalized_embeddings: L2范数=1的embeddings
#     """
#     # sklearn的normalize默认使用L2范数
#     return normalize(embeddings, norm='l2', axis=1)
#
#
# # =====================================================
# # 数据增强函数（无需修改）
# # =====================================================
# def augment_entity(entity_str, aug_prob=0.3):
#     """
#     实体文本增强（用于训练时的数据扰动）
#
#     参数:
#         entity_str: 原始实体文本
#         aug_prob: 每种增强方式的概率
#
#     增强方式:
#         1. 同义词替换
#         2. 随机插入
#         3. 随机删除（保留至少50%字符）
#
#     返回:
#         增强后的文本
#     """
#     import random
#
#     # 如果是纯数字或很短的文本，不做增强
#     if len(str(entity_str)) < 3 or str(entity_str).replace('.', '').replace('-', '').isdigit():
#         return entity_str
#
#     text = str(entity_str)
#
#     # 1. 同义词替换（针对电机领域）
#     synonyms = {
#         '电机': ['电动机', '马达', 'motor'],
#         '电压': ['电势', 'voltage'],
#         '电流': ['电荷流动', 'current'],
#         '转速': ['转动速度', '旋转速度', 'rpm'],
#         '功率': ['输出功率', 'power'],
#         '故障': ['失效', '损坏', 'fault'],
#         '控制': ['调节', '调控', 'control'],
#         '定子': ['静止部分', 'stator'],
#         '转子': ['旋转部分', 'rotor']
#     }
#
#     if random.random() < aug_prob:
#         for original, replacements in synonyms.items():
#             if original in text and random.random() < 0.5:
#                 text = text.replace(original, random.choice(replacements), 1)
#
#     # 2. 随机插入噪声（添加空格或标点）
#     if random.random() < aug_prob * 0.5:
#         insert_pos = random.randint(0, len(text))
#         noise = random.choice([' ', '-', '_'])
#         text = text[:insert_pos] + noise + text[insert_pos:]
#
#     # 3. 随机删除（但保留至少50%的字符）
#     if random.random() < aug_prob * 0.3 and len(text) > 5:
#         keep_ratio = random.uniform(0.5, 0.9)
#         keep_len = int(len(text) * keep_ratio)
#         start_pos = random.randint(0, len(text) - keep_len)
#         text = text[start_pos:start_pos + keep_len]
#
#     return text
#
#
# # =====================================================
# # Gap Statistic 计算（🔥 修改为余弦距离版本）
# # =====================================================
# def compute_gap_statistic(embeddings, max_k=15, n_refs=10, random_state=42, use_cosine=True):
#     """
#     使用Gap Statistic自动确定最佳聚类数
#
#     🔥 关键修改: 添加use_cosine参数，使用归一化后的embeddings
#
#     原理:
#         Gap(k) = E[log(W_k)] - log(W_k)
#         其中 W_k 是簇内距离和, E[log(W_k)] 是随机数据的期望
#
#     参数:
#         embeddings: 原始embeddings
#         max_k: 最大测试簇数
#         n_refs: 参考数据集数量
#         random_state: 随机种子
#         use_cosine: 是否使用余弦距离（通过归一化实现）
#
#     返回:
#         optimal_k: 最佳簇数
#         gaps: 每个k的gap值
#     """
#     from sklearn.cluster import KMeans
#
#     np.random.seed(random_state)
#
#     # 🔥 关键修改1: 归一化embeddings（使用余弦距离）
#     if use_cosine:
#         print(f"   🔥 使用余弦距离（通过L2归一化实现）")
#         embeddings = normalize_embeddings(embeddings)
#     else:
#         print(f"   使用欧氏距离")
#
#     n_samples, n_features = embeddings.shape
#     gaps = []
#
#     # 计算数据的边界
#     mins = embeddings.min(axis=0)
#     maxs = embeddings.max(axis=0)
#
#     print(f"\n🔍 Gap Statistic 自动确定最佳簇数...")
#     print(f"   测试范围: k=2 到 k={max_k}")
#
#     for k in range(2, max_k + 1):
#         # 1. 真实数据的聚类
#         kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
#         labels = kmeans.fit_predict(embeddings)
#
#         # 计算簇内距离和 W_k
#         W_k = 0.0
#         for i in range(k):
#             cluster_points = embeddings[labels == i]
#             if len(cluster_points) > 0:
#                 center = cluster_points.mean(axis=0)
#                 W_k += np.sum((cluster_points - center) ** 2)
#
#         # 2. 参考数据的聚类（随机生成）
#         ref_W_ks = []
#         for _ in range(n_refs):
#             # 在数据边界内生成随机数据
#             ref_data = np.random.uniform(mins, maxs, size=(n_samples, n_features))
#
#             # 🔥 关键修改2: 如果使用余弦距离，参考数据也要归一化
#             if use_cosine:
#                 ref_data = normalize_embeddings(ref_data)
#
#             ref_kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
#             ref_labels = ref_kmeans.fit_predict(ref_data)
#
#             ref_W_k = 0.0
#             for i in range(k):
#                 ref_cluster = ref_data[ref_labels == i]
#                 if len(ref_cluster) > 0:
#                     ref_center = ref_cluster.mean(axis=0)
#                     ref_W_k += np.sum((ref_cluster - ref_center) ** 2)
#
#             ref_W_ks.append(np.log(ref_W_k + 1e-10))
#
#         # 3. 计算Gap
#         gap = np.mean(ref_W_ks) - np.log(W_k + 1e-10)
#         gaps.append(gap)
#
#         if k % 3 == 0 or k == max_k:
#             print(f"   k={k:2d}: Gap={gap:.4f}")
#
#     # 找到第一个满足 Gap(k) >= Gap(k+1) - s_{k+1} 的k
#     gaps = np.array(gaps)
#
#     # 简化版: 找到gap最大的k
#     optimal_k = np.argmax(gaps) + 2  # +2 因为从k=2开始
#
#     print(f"   ✅ 最佳簇数: k={optimal_k} (Gap={gaps[optimal_k - 2]:.4f})")
#
#     return optimal_k, gaps
#
#
# # =====================================================
# # 层次标签构建函数（规则版本，无需修改）
# # =====================================================
# def build_hierarchical_labels_v2(entities, fine_labels):
#     """
#     改进版本 - 扩充关键词规则
#
#     原始问题：
#         规则太简单，大量样本被分到"其他"类(class 6)
#
#     改进方案：
#         每个粗类增加15+个关键词，覆盖更多样本
#     """
#     coarse_labels = []
#
#     # ✅ 扩充后的规则（每类15+关键词）
#     coarse_rules = {
#         0: [  # 电机类型
#             "电机", "电动机", "发电机", "马达", "motor",
#             "异步", "同步", "直流", "交流", "伺服",
#             "步进", "无刷", "有刷", "永磁", "感应"
#         ],
#         1: [  # 电气参数
#             "电压", "电流", "功率", "电阻", "频率",
#             "相数", "电能", "电荷", "电场", "磁场",
#             "电容", "电感", "阻抗", "导纳", "电势"
#         ],
#         2: [  # 机械参数
#             "转速", "扭矩", "负载", "转矩", "惯量",
#             "速度", "加速", "减速", "力矩", "功",
#             "能量", "动能", "势能", "机械", "运动"
#         ],
#         3: [  # 控制方式
#             "控制", "调速", "启动", "制动", "保护",
#             "调节", "开关", "继电器", "接触器", "变频",
#             "PWM", "PID", "反馈", "闭环", "开环"
#         ],
#         4: [  # 部件
#             "定子", "转子", "绕组", "轴承", "外壳",
#             "端盖", "机座", "风扇", "铁芯", "线圈",
#             "槽", "极", "轴", "联轴器", "底座"
#         ],
#         5: [  # 故障
#             "故障", "损坏", "磨损", "过热", "短路",
#             "断路", "漏电", "振动", "噪音", "异常",
#             "失效", "老化", "击穿", "烧毁", "卡死"
#         ],
#         6: []  # 其他（默认类）
#     }
#
#     # 统计匹配情况
#     match_stats = {i: 0 for i in range(7)}
#
#     for entity in entities:
#         entity_str = str(entity).lower()  # 转小写增加匹配率
#         matched = False
#
#         for coarse_id, keywords in coarse_rules.items():
#             if any(kw in entity_str for kw in keywords):
#                 coarse_labels.append(coarse_id)
#                 match_stats[coarse_id] += 1
#                 matched = True
#                 break
#
#         if not matched:
#             coarse_labels.append(6)
#             match_stats[6] += 1
#
#     coarse_labels = np.array(coarse_labels)
#
#     # 打印统计信息
#     print(f"\n✅ 层次标签统计 (改进版):")
#     print(f"   粗粒度: {len(np.unique(coarse_labels))} 类")
#     print(f"   细粒度: {len(np.unique(fine_labels))} 类")
#     print(f"\n   各粗类分布:")
#
#     class_names = ["电机类型", "电气参数", "机械参数", "控制方式", "部件", "故障", "其他"]
#     for coarse_id in range(7):
#         mask = (coarse_labels == coarse_id)
#         n_samples = mask.sum()
#         fine_in_coarse = len(np.unique(fine_labels[mask]))
#         percentage = n_samples / len(coarse_labels) * 100
#         print(f"   粗类{coarse_id} ({class_names[coarse_id]:6s}): "
#               f"{n_samples:4d} 样本 ({percentage:5.1f}%), {fine_in_coarse:3d} 细类")
#
#     # ⚠️ 警告：如果"其他"类占比超过30%，建议使用混合策略
#     other_ratio = match_stats[6] / len(entities)
#     if other_ratio > 0.3:
#         print(f"\n⚠️  警告: '其他'类占比过高 ({other_ratio:.1%})")
#         print(f"   建议使用 build_hierarchical_labels_hybrid() 混合策略")
#
#     return coarse_labels
#
#
# # =====================================================
# # 🔥 修改：混合策略（使用余弦距离）
# # =====================================================
# def build_hierarchical_labels_hybrid(entities, fine_labels, embeddings,
#                                      n_coarse_clusters=7, rule_weight=0.6,
#                                      use_cosine=True):
#     """
#     混合策略 - 规则 + 聚类
#
#     🔥 关键修改: 添加use_cosine参数，聚类时使用余弦距离
#
#     适用场景：
#         - 规则匹配率 < 70%（即"其他"类 > 30%）
#         - 希望自动发现隐藏的粗类结构
#
#     原理：
#         1. 先用规则分配有把握的样本
#         2. 对"其他"类样本用聚类重新分配
#         3. 最终粗类数 = 规则类数 + 聚类发现的新类
#
#     参数:
#         use_cosine: 是否使用余弦距离（默认True）
#     """
#     from sklearn.cluster import KMeans
#
#     print(f"\n🔄 使用混合策略构建层次标签...")
#     if use_cosine:
#         print(f"   🔥 聚类使用余弦距离（通过L2归一化实现）")
#
#     # Step 1: 规则匹配
#     rule_labels = build_hierarchical_labels_v2(entities, fine_labels)
#
#     # Step 2: 对"其他"类用聚类细分
#     other_mask = (rule_labels == 6)
#     n_other = other_mask.sum()
#
#     if n_other > 100:  # 只有"其他"类样本足够多才聚类
#         print(f"\n   正在对 {n_other} 个'其他'类样本进行聚类...")
#
#         other_embeddings = embeddings[other_mask]
#
#         # 🔥 关键修改: 归一化embeddings（使用余弦距离）
#         if use_cosine:
#             other_embeddings = normalize_embeddings(other_embeddings)
#
#         # 聚类数 = min(3, 其他类样本数/50)
#         n_sub_clusters = min(3, max(2, n_other // 50))
#
#         kmeans = KMeans(n_clusters=n_sub_clusters, random_state=42, n_init=20)
#         sub_labels = kmeans.fit_predict(other_embeddings)
#
#         # 重新分配"其他"类样本
#         # 原始规则类: 0-5, 新聚类类: 7, 8, 9...
#         hybrid_labels = rule_labels.copy()
#         hybrid_labels[other_mask] = 7 + sub_labels  # 从7开始编号
#
#         print(f"   ✅ '其他'类被细分为 {n_sub_clusters} 个子类")
#     else:
#         hybrid_labels = rule_labels
#         print(f"   ℹ️ '其他'类样本较少 ({n_other})，保持不变")
#
#     # 统计最终分布
#     print(f"\n✅ 混合策略最终统计:")
#     print(f"   粗粒度: {len(np.unique(hybrid_labels))} 类")
#     print(f"   细粒度: {len(np.unique(fine_labels))} 类")
#
#     for coarse_id in np.unique(hybrid_labels):
#         mask = (hybrid_labels == coarse_id)
#         n_samples = mask.sum()
#         fine_in_coarse = len(np.unique(fine_labels[mask]))
#         percentage = n_samples / len(hybrid_labels) * 100
#
#         if coarse_id < 6:
#             class_names = ["电机类型", "电气参数", "机械参数", "控制方式", "部件", "故障"]
#             name = class_names[coarse_id]
#         elif coarse_id == 6:
#             name = "其他(未分)"
#         else:
#             name = f"聚类子类{coarse_id - 6}"
#
#         print(f"   粗类{coarse_id} ({name:10s}): "
#               f"{n_samples:4d} 样本 ({percentage:5.1f}%), {fine_in_coarse:3d} 细类")
#
#     return hybrid_labels
#
#
# # =====================================================
# # 🔥 修改：自适应聚类（使用余弦距离）
# # =====================================================
# def build_hierarchical_labels_auto(embeddings, fine_labels, max_k=15, use_cosine=True):
#     """
#     🔥 改进版 - 使用Gap Statistic自动确定最佳粗类数量
#     🔥 关键修改: 添加use_cosine参数，使用余弦距离
#
#     改进点:
#         1. 自动确定最佳簇数（不再固定为7）
#         2. 避免"其他"类占比过高
#         3. 使用Gap Statistic科学选择簇数
#         4. 支持余弦距离（通过归一化实现）
#
#     适用场景：
#         - 完全不了解数据的语义结构
#         - 希望让模型自动发现最优层次
#         - 避免人为设定簇数带来的偏差
#
#     参数:
#         use_cosine: 是否使用余弦距离（默认True）
#     """
#     from sklearn.cluster import KMeans
#
#     print(f"\n🔄 使用自适应Gap Statistic策略构建层次标签...")
#
#     # 🔥 Step 1: 使用Gap Statistic自动确定最佳簇数（支持余弦距离）
#     optimal_k, gaps = compute_gap_statistic(embeddings, max_k=max_k, use_cosine=use_cosine)
#
#     print(f"\n✅ 自动确定的最佳粗类数: {optimal_k}")
#
#     # 🔥 关键修改: 归一化embeddings（使用余弦距离）
#     if use_cosine:
#         print(f"   🔥 聚类使用余弦距离（通过L2归一化实现）")
#         embeddings_normalized = normalize_embeddings(embeddings)
#     else:
#         print(f"   使用欧氏距离")
#         embeddings_normalized = embeddings
#
#     # Step 2: 使用最佳簇数进行聚类
#     kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
#     coarse_labels = kmeans.fit_predict(embeddings_normalized)
#
#     # Step 3: 统计分布
#     print(f"\n✅ 自动构建层次标签:")
#     print(f"   粗粒度: {optimal_k} 类 (自动确定)")
#     print(f"   细粒度: {len(np.unique(fine_labels))} 类")
#     print(f"\n   各粗类分布:")
#
#     for coarse_id in range(optimal_k):
#         mask = (coarse_labels == coarse_id)
#         n_samples = mask.sum()
#         fine_in_coarse = len(np.unique(fine_labels[mask]))
#         percentage = n_samples / len(coarse_labels) * 100
#         print(f"   粗类{coarse_id}: {n_samples:4d} 样本 ({percentage:5.1f}%), "
#               f"{fine_in_coarse:3d} 细类")
#
#     # Step 4: 检查平衡性
#     class_counts = [np.sum(coarse_labels == i) for i in range(optimal_k)]
#     max_ratio = max(class_counts) / len(coarse_labels)
#     min_ratio = min(class_counts) / len(coarse_labels)
#
#     print(f"\n📊 粗类平衡性:")
#     print(f"   最大类占比: {max_ratio:.1%}")
#     print(f"   最小类占比: {min_ratio:.1%}")
#
#     if max_ratio > 0.4:
#         print(f"   ⚠️  警告: 存在过大的粗类（>{max_ratio:.1%}），可能影响层次学习效果")
#     else:
#         print(f"   ✅ 粗类分布较为均衡")
#
#     return coarse_labels
#
#
# # =====================================================
# # 辅助函数：评估粗类质量（无需修改）
# # =====================================================
# def evaluate_coarse_quality(coarse_labels, fine_labels):
#     """
#     评估粗类标签质量
#
#     指标：
#     1. 平衡性：各粗类样本数方差（越小越好）
#     2. 纯度：每个粗类内细类数量（适中为好）
#     3. 覆盖率：被正确分配的样本比例
#     """
#     n_coarse = len(np.unique(coarse_labels))
#
#     # 1. 平衡性
#     class_counts = [np.sum(coarse_labels == i) for i in range(n_coarse)]
#     balance_score = 1.0 - (np.std(class_counts) / (np.mean(class_counts) + 1e-8))
#
#     # 2. 纯度（每个粗类平均包含多少细类）
#     fine_counts = []
#     for i in range(n_coarse):
#         mask = (coarse_labels == i)
#         fine_in_coarse = len(np.unique(fine_labels[mask]))
#         fine_counts.append(fine_in_coarse)
#     avg_fine_per_coarse = np.mean(fine_counts)
#
#     # 3. 理想状态：每个粗类包含 总细类数/粗类数 个细类
#     ideal_fine_per_coarse = len(np.unique(fine_labels)) / n_coarse
#     purity_score = 1.0 - abs(avg_fine_per_coarse - ideal_fine_per_coarse) / ideal_fine_per_coarse
#
#     print(f"\n📊 粗类质量评估:")
#     print(f"   平衡性得分: {balance_score:.3f} (越接近1越好)")
#     print(f"   每粗类平均细类数: {avg_fine_per_coarse:.1f} (理想值: {ideal_fine_per_coarse:.1f})")
#     print(f"   纯度得分: {purity_score:.3f} (越接近1越好)")
#
#     return {
#         'balance_score': balance_score,
#         'purity_score': purity_score,
#         'avg_fine_per_coarse': avg_fine_per_coarse
#     }
#
#
# # =====================================================
# # 🔥 新增：验证是否真的使用余弦距离
# # =====================================================
# def verify_distance_metric(embeddings, n_samples=5):
#     """
#     验证聚类是否真的使用了余弦距离
#
#     通过对比归一化前后的聚类结果，确认距离度量的改变
#     """
#     from sklearn.cluster import KMeans
#     from scipy.spatial.distance import cosine, euclidean
#
#     print(f"\n🔬 验证距离度量...")
#
#     # 随机选择几个样本
#     np.random.seed(42)
#     indices = np.random.choice(len(embeddings), min(n_samples, len(embeddings)), replace=False)
#     sample_embs = embeddings[indices]
#
#     # 计算第一个样本到其他样本的距离
#     print(f"\n   样本0到其他样本的距离:")
#     print(f"   {'样本对':<10} {'欧氏距离':<12} {'余弦距离':<12}")
#     print(f"   {'-' * 40}")
#
#     for i in range(1, len(sample_embs)):
#         euc_dist = euclidean(sample_embs[0], sample_embs[i])
#         cos_dist = cosine(sample_embs[0], sample_embs[i])
#         print(f"   0 <-> {i:<5} {euc_dist:<12.4f} {cos_dist:<12.4f}")
#
#     # 对比归一化前后的聚类结果
#     if len(embeddings) > 100:
#         k = 5
#
#         # 原始embeddings聚类（欧氏距离）
#         kmeans_euc = KMeans(n_clusters=k, random_state=42, n_init=10)
#         labels_euc = kmeans_euc.fit_predict(embeddings)
#
#         # 归一化后聚类（余弦距离）
#         embeddings_norm = normalize_embeddings(embeddings)
#         kmeans_cos = KMeans(n_clusters=k, random_state=42, n_init=10)
#         labels_cos = kmeans_cos.fit_predict(embeddings_norm)
#
#         # 计算标签差异
#         diff_ratio = (labels_euc != labels_cos).mean()
#
#         print(f"\n   聚类结果对比 (k={k}):")
#         print(f"   欧氏距离 vs 余弦距离的标签差异率: {diff_ratio:.2%}")
#
#         if diff_ratio > 0.05:
#             print(f"   ✅ 差异显著，确认使用了不同的距离度量")
#         else:
#             print(f"   ⚠️ 差异较小，可能数据本身对距离不敏感")
#
#
# # =====================================================
# # 使用示例
# # =====================================================
# if __name__ == "__main__":
#     # 模拟数据
#     entities = ["异步电机", "额定电压", "转速", "启动方式", "定子", "过热故障", "未知术语"] * 100
#     fine_labels = np.random.randint(0, 50, len(entities))
#     embeddings = np.random.randn(len(entities), 128)
#
#     # 🔥 新增：验证距离度量
#     print("=" * 60)
#     print("🔬 距离度量验证")
#     print("=" * 60)
#     verify_distance_metric(embeddings)
#
#     print("\n" + "=" * 60 + "\n")
#
#     # 测试数据增强
#     print("=== 数据增强测试 ===")
#     test_entities = ["异步电机", "额定电压380V", "转速1500rpm"]
#     for ent in test_entities:
#         aug_ent = augment_entity(ent, aug_prob=0.5)
#         print(f"原始: {ent:20s} -> 增强: {aug_ent}")
#
#     print("\n" + "=" * 60 + "\n")
#
#     # 方法1: 改进的规则匹配
#     coarse_v2 = build_hierarchical_labels_v2(entities, fine_labels)
#
#     # 方法2: 混合策略（🔥 使用余弦距离）
#     print("\n" + "=" * 60)
#     print("【测试混合策略 - 余弦距离】")
#     print("=" * 60)
#     coarse_hybrid_cos = build_hierarchical_labels_hybrid(
#         entities, fine_labels, embeddings,
#         use_cosine=True  # 🔥 使用余弦距离
#     )
#
#     print("\n" + "=" * 60)
#     print("【测试混合策略 - 欧氏距离（对比）】")
#     print("=" * 60)
#     coarse_hybrid_euc = build_hierarchical_labels_hybrid(
#         entities, fine_labels, embeddings,
#         use_cosine=False  # 使用欧氏距离
#     )
#
#     # 方法3: 自适应Gap Statistic（🔥 使用余弦距离）
#     print("\n" + "=" * 60)
#     print("【测试自适应策略 - 余弦距离】")
#     print("=" * 60)
#     coarse_auto_cos = build_hierarchical_labels_auto(
#         embeddings, fine_labels, max_k=15,
#         use_cosine=True  # 🔥 使用余弦距离
#     )
#
#     print("\n" + "=" * 60)
#     print("【测试自适应策略 - 欧氏距离（对比）】")
#     print("=" * 60)
#     coarse_auto_euc = build_hierarchical_labels_auto(
#         embeddings, fine_labels, max_k=15,
#         use_cosine=False  # 使用欧氏距离
#     )
#
#     # 评估质量
#     print("\n" + "=" * 60)
#     print("🔬 质量对比")
#     print("=" * 60)
#
#     print("\n【规则匹配方法】")
#     evaluate_coarse_quality(coarse_v2, fine_labels)
#
#     print("\n【混合策略 - 余弦距离】🔥")
#     evaluate_coarse_quality(coarse_hybrid_cos, fine_labels)
#
#     print("\n【混合策略 - 欧氏距离】")
#     evaluate_coarse_quality(coarse_hybrid_euc, fine_labels)
#
#     print("\n【自适应Gap Statistic - 余弦距离】🔥")
#     evaluate_coarse_quality(coarse_auto_cos, fine_labels)
#
#     print("\n【自适应Gap Statistic - 欧氏距离】")
#     evaluate_coarse_quality(coarse_auto_euc, fine_labels)
#
#     # 🔥 对比余弦 vs 欧氏的聚类差异
#     print("\n" + "=" * 60)
#     print("🔬 余弦距离 vs 欧氏距离聚类结果对比")
#     print("=" * 60)
#
#     diff_ratio_hybrid = (coarse_hybrid_cos != coarse_hybrid_euc).mean()
#     diff_ratio_auto = (coarse_auto_cos != coarse_auto_euc).mean()
#
#     print(f"\n混合策略:")
#     print(f"   标签差异率: {diff_ratio_hybrid:.2%}")
#     if diff_ratio_hybrid > 0.1:
#         print(f"   ✅ 差异显著，余弦距离确实改变了聚类结果")
#     else:
#         print(f"   ⚠️ 差异较小，可能数据对距离度量不敏感")
#
#     print(f"\n自适应策略:")
#     print(f"   标签差异率: {diff_ratio_auto:.2%}")
#     if diff_ratio_auto > 0.1:
#         print(f"   ✅ 差异显著，余弦距离确实改变了聚类结果")
#     else:
#         print(f"   ⚠️ 差异较小，可能数据对距离度量不敏感")
#
#     print("\n" + "=" * 60)
#     print("✅ 测试完成！")
#     print("=" * 60)


import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

def augment_entity(entity, noise_ratio=0.1):
    """
    对实体进行轻微扰动增强，用于实例对比学习。
    这里可以是简单的字符随机替换、插入、删除等。
    """
    import random
    entity_str = str(entity)
    n = max(1, int(len(entity_str) * noise_ratio))
    chars = list(entity_str)
    for _ in range(n):
        idx = random.randint(0, len(chars)-1)
        chars[idx] = chr(random.randint(19968, 40869))  # 随机汉字
    return ''.join(chars)


def build_hierarchical_labels(entities, fine_labels, n_coarse_clusters=5, use_cosine=False):
    """
    使用KMeans/层次聚类构建粗粒度标签
    """
    # 简单把实体embedding当作one-hot向量
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(analyzer='char')
    X = vectorizer.fit_transform(entities).toarray().astype(np.float32)

    if use_cosine:
        X = normalize(X, axis=1)

    kmeans = KMeans(n_clusters=n_coarse_clusters, random_state=42)
    coarse_labels = kmeans.fit_predict(X)
    return coarse_labels


def build_hierarchical_labels_hybrid(entities, fine_labels, embeddings, n_coarse_clusters=5, use_cosine=False):
    """
    规则 + embedding混合构建粗粒度标签
    """
    if use_cosine:
        embeddings = normalize(embeddings, axis=1)
    # 用AgglomerativeClustering做初步分簇
    clustering = AgglomerativeClustering(n_clusters=n_coarse_clusters, linkage='average', affinity='cosine' if use_cosine else 'euclidean')
    coarse_labels = clustering.fit_predict(embeddings)
    return coarse_labels


def build_coarse_labels_with_acrl(embeddings, fine_labels, n_coarse_clusters=5,
                                  alpha=0.5, n_acrl_iterations=3, use_cosine=False,
                                  verbose=False, initial_labels=None):
    """
    基于ACRL算法优化粗粒度标签
    embeddings: np.array, N x D
    fine_labels: np.array, N
    initial_labels: 初始粗粒度标签
    """
    N, D = embeddings.shape
    if use_cosine:
        embeddings = normalize(embeddings, axis=1)

    if initial_labels is None:
        kmeans = KMeans(n_clusters=n_coarse_clusters, random_state=42)
        coarse_labels = kmeans.fit_predict(embeddings)
    else:
        coarse_labels = initial_labels.copy()

    for it in range(n_acrl_iterations):
        if verbose:
            print(f"  🔹 ACRL迭代 {it+1}/{n_acrl_iterations}")
        # 计算每个粗类内细粒度纯度
        coarse_to_fine_counts = {}
        for c in np.unique(coarse_labels):
            mask = coarse_labels == c
            fine_in_cluster = fine_labels[mask]
            counts = np.bincount(fine_in_cluster)
            coarse_to_fine_counts[c] = counts

        # 调整簇中心（简单示例: 每个样本重新指派到最大相似粗类）
        for i in range(N):
            distances = []
            for c in np.unique(coarse_labels):
                mask = coarse_labels == c
                if mask.sum() == 0:
                    distances.append(np.inf)
                    continue
                center = embeddings[mask].mean(axis=0)
                if use_cosine:
                    dist = 1 - np.dot(embeddings[i], center) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(center))
                else:
                    dist = np.linalg.norm(embeddings[i] - center)
                # 融合细粒度信息
                fine_idx = fine_labels[i]
                fine_count = coarse_to_fine_counts[c][fine_idx] if fine_idx < len(coarse_to_fine_counts[c]) else 0
                dist = alpha * dist - (1 - alpha) * fine_count
                distances.append(dist)
            coarse_labels[i] = int(np.argmin(distances))

    # 最终簇中心
    acrl_centers = np.array([embeddings[coarse_labels == c].mean(axis=0) for c in np.unique(coarse_labels)])
    return coarse_labels, acrl_centers


def evaluate_coarse_quality(coarse_labels, fine_labels):
    """
    简单评估粗粒度标签质量: 计算簇内细粒度纯度
    """
    purity_list = []
    for c in np.unique(coarse_labels):
        mask = coarse_labels == c
        fine_in_cluster = fine_labels[mask]
        if len(fine_in_cluster) == 0:
            continue
        counts = np.bincount(fine_in_cluster)
        purity = counts.max() / counts.sum()
        purity_list.append(purity)
    avg_purity = np.mean(purity_list)
    print(f"  🎯 粗粒度簇平均纯度: {avg_purity:.4f}")
    return avg_purity