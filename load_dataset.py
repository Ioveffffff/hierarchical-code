import pandas as pd
import numpy as np
from tqdm import tqdm


def load_texts_from_csv(csv_path, mode='fused'):
    """Read training texts without accessing the ground-truth label column."""
    columns = pd.read_csv(csv_path, encoding='utf-8', nrows=0).columns
    has_pair = {'text0', 'text1'}.issubset(columns)
    if 'text' in columns and not has_pair:
        df = pd.read_csv(csv_path, encoding='utf-8', usecols=['text'])
        entities = df['text'].astype(str).str.strip().tolist()
    elif has_pair and mode == 'fused':
        df = pd.read_csv(csv_path, encoding='utf-8', usecols=['text0', 'text1'])
        entities = (df['text0'].astype(str).str.strip() + ' ' +
                    df['text1'].astype(str).str.strip()).tolist()
    elif has_pair and mode == 'text0_only':
        df = pd.read_csv(csv_path, encoding='utf-8', usecols=['text0'])
        entities = df['text0'].astype(str).str.strip().tolist()
    else:
        raise ValueError('Training CSV needs text or text0/text1; mode must be fused or text0_only')
    if not entities or any(not text or text.lower() == 'nan' for text in entities):
        raise ValueError('Training texts must be nonempty')
    print(f'Loaded {len(entities)} training texts without labels')
    return entities


def load_entities_from_csv(csv_path, mode='fused'):
    """
    加载CSV数据，支持多种模式处理同义词对

    参数:
        csv_path: CSV文件路径
        mode: 处理模式
            - 'fused': 拼接text0和text1为单一文本 (推荐，简单有效)
            - 'paired': 返回(text0, text1)元组，用于对比学习
            - 'text0_only': 只使用text0
            - 'average': 返回text0和text1分开，后续取平均

    返回:
        entities: 文本列表
        labels: 标签数组
    """
    df = pd.read_csv(csv_path, encoding="utf-8")

    # 检查列格式
    has_text = "text" in df.columns
    has_pair = ("text0" in df.columns) and ("text1" in df.columns)

    if not has_text and not has_pair:
        raise ValueError(
            f"CSV列不符合要求：需要 (label,text) 或 (label,text0,text1)，"
            f"实际列={list(df.columns)}"
        )

    entities = []
    labels = []

    if has_text and not has_pair:
        print("检测到单文本格式 (label, text)")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading CSV", ncols=80):
            entities.append(str(row["text"]).strip())
            labels.append(int(row["label"]))

    elif has_pair:
        print(f"检测到双文本格式 (label, text0, text1)，使用模式: {mode}")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading CSV", ncols=80):
            y = int(row["label"])
            t0 = str(row["text0"]).strip()
            t1 = str(row["text1"]).strip()

            if mode == 'fused':
                fused_text = f"{t0} {t1}"
                entities.append(fused_text)
                labels.append(y)

            elif mode == 'paired':
                entities.append((t0, t1))
                labels.append(y)

            elif mode == 'text0_only':
                entities.append(t0)
                labels.append(y)

            elif mode == 'average':
                entities.append((t0, t1))
                labels.append(y)

            else:
                raise ValueError(f"不支持的mode: {mode}")

    labels = np.array(labels, dtype=np.int64)

    # 打印统计信息
    n_samples = len(entities)
    n_classes = len(set(labels.tolist()))
    print(f"已加载 {n_samples} 条样本, 标签数: {n_classes}")

    if has_pair and mode == 'fused':
        # 检查融合后的文本长度
        avg_len = np.mean([len(text.split()) for text in entities])
        print(f"   平均文本长度: {avg_len:.1f} 词")
        if avg_len > 100:
            print(f"   警告: 文本较长，可能需要调整tokenizer的max_length")

    return entities, labels


def load_entities_with_synonyms(csv_path):
    """
    专门用于对比学习的加载方式

    返回:
        entities: text0列表 (主文本)
        synonym_pairs: [(text0, text1), ...] 同义词对
        labels: 标签数组
    """
    df = pd.read_csv(csv_path, encoding="utf-8")

    if not (("text0" in df.columns) and ("text1" in df.columns)):
        raise ValueError("此函数需要 (label, text0, text1) 格式")

    entities = []
    synonym_pairs = []
    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading Synonyms"):
        y = int(row["label"])
        t0 = str(row["text0"]).strip()
        t1 = str(row["text1"]).strip()

        entities.append(t0)
        synonym_pairs.append((t0, t1))
        labels.append(y)

    labels = np.array(labels, dtype=np.int64)
    print(f"已加载 {len(entities)} 条样本 + {len(synonym_pairs)} 对同义词")

    return entities, synonym_pairs, labels
