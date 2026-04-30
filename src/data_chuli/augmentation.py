"""
数据增强模块 - 用于对比学习的实体文本增强

针对实体匹配任务设计的数据增强策略:
1. Attribute Masking - 随机遮蔽属性值
2. Token Shuffle - 打乱词序
3. Attribute Dropout - 随机删除部分词
4. Synonym Replacement - 同义词替换

Author: PathCL-EM Project
Date: 2025-01-10
"""

import random
from typing import List, Dict, Optional


def attribute_masking(entity_text: str, mask_prob: float = 0.15, mask_token: str = '[MASK]') -> str:
    """
    随机遮蔽属性值

    Args:
        entity_text: 实体文本, 如 "Apple iPhone 8 Plus 64GB Silver"
        mask_prob: 遮蔽概率 (默认 0.15, 借鉴 BERT)
        mask_token: 遮蔽标记

    Returns:
        augmented_text: 遮蔽后的文本, 如 "Apple [MASK] 8 Plus 64GB Silver"

    Example:
        >>> attribute_masking("Apple iPhone 8 Plus 64GB Silver", mask_prob=0.2)
        "Apple [MASK] 8 Plus [MASK] Silver"
    """
    tokens = entity_text.split()

    if len(tokens) == 0:
        return entity_text

    # 计算要遮蔽的词数量
    num_mask = max(1, int(len(tokens) * mask_prob))

    # 随机选择要遮蔽的位置
    mask_indices = random.sample(range(len(tokens)), min(num_mask, len(tokens)))

    # 执行遮蔽
    for idx in mask_indices:
        tokens[idx] = mask_token

    return ' '.join(tokens)


def token_shuffle(entity_text: str, keep_first: bool = True) -> str:
    """
    打乱词序 (可选保留第一个词)

    实体匹配通常不依赖词序,但第一个词往往是重要属性(如品牌)

    Args:
        entity_text: 原始文本
        keep_first: 是否保留第一个词不动 (默认 True)

    Returns:
        shuffled_text: 打乱后的文本

    Example:
        >>> token_shuffle("Apple iPhone 8 Plus 64GB Silver", keep_first=True)
        "Apple 64GB iPhone Plus Silver 8"
    """
    tokens = entity_text.split()

    if len(tokens) <= 1:
        return entity_text

    if keep_first:
        first_token = tokens[0]
        rest_tokens = tokens[1:]
        random.shuffle(rest_tokens)
        return first_token + ' ' + ' '.join(rest_tokens)
    else:
        random.shuffle(tokens)
        return ' '.join(tokens)


def attribute_dropout(entity_text: str, dropout_prob: float = 0.3) -> str:
    """
    随机删除部分词 (模拟不同数据源缺失字段)

    Args:
        entity_text: 原始文本
        dropout_prob: 删除概率 (默认 0.3)

    Returns:
        dropped_text: 删除后的文本

    Example:
        >>> attribute_dropout("Apple iPhone 8 Plus 64GB Silver", dropout_prob=0.3)
        "Apple iPhone Plus Silver"  # 随机删除了 "8" 和 "64GB"
    """
    tokens = entity_text.split()

    if len(tokens) <= 1:
        return entity_text

    # 至少保留一个词
    keep_mask = [random.random() > dropout_prob for _ in tokens]

    # 确保至少保留一个词
    if not any(keep_mask):
        keep_mask[random.randint(0, len(keep_mask) - 1)] = True

    filtered_tokens = [token for token, keep in zip(tokens, keep_mask) if keep]

    return ' '.join(filtered_tokens)


def synonym_replacement(entity_text: str, synonym_dict: Optional[Dict[str, List[str]]] = None,
                        replace_prob: float = 0.5) -> str:
    """
    同义词替换 (针对实体匹配常见模式)

    Args:
        entity_text: 原始文本
        synonym_dict: 同义词字典, 格式 {'word': ['synonym1', 'synonym2', ...]}
        replace_prob: 替换概率 (默认 0.5)

    Returns:
        replaced_text: 替换后的文本

    Example:
        >>> synonym_replacement("Apple iPhone 8 64GB Silver")
        "Apple iPhone 8 64 Gigabyte White"  # GB→Gigabyte, Silver→White
    """
    # 默认同义词字典 (针对电商/产品实体匹配)
    if synonym_dict is None:
        synonym_dict = {
            # 存储单位
            'GB': ['Gigabyte', 'G', 'GiB'],
            'MB': ['Megabyte', 'M', 'MiB'],
            'TB': ['Terabyte', 'T', 'TiB'],

            # 颜色
            'silver': ['white', 'grey', 'gray'],
            'gold': ['golden', 'yellow'],
            'black': ['dark', 'noir'],
            'blue': ['navy', 'azure'],
            'red': ['crimson', 'scarlet'],

            # 尺寸
            'plus': ['large', 'big', '+'],
            'mini': ['small', 'compact', 'tiny'],
            'pro': ['professional', 'premium'],

            # 数字表示
            '64': ['sixty-four', '64.0'],
            '128': ['one-twenty-eight', '128.0'],
            '256': ['two-fifty-six', '256.0'],

            # 品牌简称
            'iphone': ['iPhone', 'i-phone'],
            'samsung': ['Samsung', 'SAMSUNG'],
        }

    tokens = entity_text.split()

    for i, token in enumerate(tokens):
        # 检查原词和小写版本
        token_lower = token.lower()

        if token in synonym_dict and random.random() < replace_prob:
            tokens[i] = random.choice(synonym_dict[token])
        elif token_lower in synonym_dict and random.random() < replace_prob:
            tokens[i] = random.choice(synonym_dict[token_lower])

    return ' '.join(tokens)


def augment_entity(entity_text: str,
                   methods: List[str] = ['mask', 'shuffle'],
                   mask_prob: float = 0.15,
                   dropout_prob: float = 0.3,
                   shuffle_keep_first: bool = True,
                   synonym_replace_prob: float = 0.5) -> str:
    """
    组合多种增强方法

    Args:
        entity_text: 原始文本
        methods: 增强方法列表, 可选 ['mask', 'shuffle', 'dropout', 'synonym']
        mask_prob: 遮蔽概率
        dropout_prob: 删除概率
        shuffle_keep_first: 打乱时是否保留第一个词
        synonym_replace_prob: 同义词替换概率

    Returns:
        augmented_text: 增强后的文本

    Example:
        >>> augment_entity("Apple iPhone 8 Plus 64GB Silver", methods=['mask', 'shuffle'])
        "Apple Plus [MASK] Silver iPhone 64GB"
    """
    text = entity_text

    for method in methods:
        if method == 'mask':
            text = attribute_masking(text, mask_prob=mask_prob)
        elif method == 'shuffle':
            text = token_shuffle(text, keep_first=shuffle_keep_first)
        elif method == 'dropout':
            text = attribute_dropout(text, dropout_prob=dropout_prob)
        elif method == 'synonym':
            text = synonym_replacement(text, replace_prob=synonym_replace_prob)
        else:
            raise ValueError(f"Unknown augmentation method: {method}")

    return text


def create_augmented_pair(entity_text: str,
                          method_set1: List[str] = ['mask', 'shuffle'],
                          method_set2: List[str] = ['dropout', 'synonym']) -> tuple:
    """
    为单个实体创建两个不同的增强视图 (用于对比学习)

    Args:
        entity_text: 原始实体文本
        method_set1: 第一个视图的增强方法
        method_set2: 第二个视图的增强方法

    Returns:
        (view1, view2): 两个增强后的视图

    Example:
        >>> create_augmented_pair("Apple iPhone 8 Plus 64GB Silver")
        ("Apple [MASK] Plus 64GB Silver iPhone", "Apple iPhone Plus Silver")
    """
    view1 = augment_entity(entity_text, methods=method_set1)
    view2 = augment_entity(entity_text, methods=method_set2)

    return view1, view2


def batch_augment(entity_texts: List[str],
                  num_views: int = 2,
                  method_sets: Optional[List[List[str]]] = None) -> List[List[str]]:
    """
    批量增强

    Args:
        entity_texts: 实体文本列表
        num_views: 每个实体生成的视图数量
        method_sets: 各视图的增强方法列表 (若为 None,自动分配)

    Returns:
        augmented_views: 形状 [num_views, batch_size] 的增强视图列表

    Example:
        >>> texts = ["Apple iPhone 8", "Samsung Galaxy S8"]
        >>> batch_augment(texts, num_views=2)
        [["Apple [MASK]", "Samsung [MASK] S8"],  # view 1
         ["iPhone Apple", "Galaxy Samsung"]]     # view 2
    """
    if method_sets is None:
        # 默认方法集合
        method_sets = [
            ['mask'],
            ['shuffle'],
            ['dropout'],
            ['mask', 'shuffle'],
            ['dropout', 'synonym'],
        ]

    # 确保有足够的方法集
    if len(method_sets) < num_views:
        method_sets = method_sets * ((num_views // len(method_sets)) + 1)

    augmented_views = []

    for view_idx in range(num_views):
        view_texts = []
        methods = method_sets[view_idx]

        for entity_text in entity_texts:
            augmented = augment_entity(entity_text, methods=methods)
            view_texts.append(augmented)

        augmented_views.append(view_texts)

    return augmented_views


# ==================== 单元测试 ====================
if __name__ == '__main__':
    print("=" * 60)
    print("测试数据增强模块")
    print("=" * 60)

    # 测试样本
    test_entity = "Apple iPhone 8 Plus 64GB Silver"
    print(f"\n原始实体: {test_entity}\n")

    # 测试各种增强方法
    print("1. Attribute Masking:")
    for i in range(3):
        masked = attribute_masking(test_entity, mask_prob=0.2)
        print(f"   - {masked}")

    print("\n2. Token Shuffle:")
    for i in range(3):
        shuffled = token_shuffle(test_entity, keep_first=True)
        print(f"   - {shuffled}")

    print("\n3. Attribute Dropout:")
    for i in range(3):
        dropped = attribute_dropout(test_entity, dropout_prob=0.3)
        print(f"   - {dropped}")

    print("\n4. Synonym Replacement:")
    for i in range(3):
        replaced = synonym_replacement(test_entity, replace_prob=0.5)
        print(f"   - {replaced}")

    print("\n5. 组合增强 (mask + shuffle):")
    for i in range(3):
        augmented = augment_entity(test_entity, methods=['mask', 'shuffle'])
        print(f"   - {augmented}")

    print("\n6. 创建增强对 (用于对比学习):")
    for i in range(3):
        view1, view2 = create_augmented_pair(test_entity)
        print(f"   View 1: {view1}")
        print(f"   View 2: {view2}")
        print()

    print("\n7. 批量增强:")
    test_batch = [
        "Apple iPhone 8 Plus 64GB Silver",
        "Samsung Galaxy S8 64GB Black",
        "Huawei P10 32GB Gold"
    ]
    augmented_views = batch_augment(test_batch, num_views=2)
    print(f"   原始批次: {test_batch}")
    print(f"   View 1: {augmented_views[0]}")
    print(f"   View 2: {augmented_views[1]}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
