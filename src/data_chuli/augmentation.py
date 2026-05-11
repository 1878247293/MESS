"""
对比学习用的文本增强。

针对实体匹配做了几种小手段：
- attribute masking：随机用 [MASK] 盖词
- token shuffle：打乱顺序，可选保留首词
- attribute dropout：随机扔词
- synonym replacement：常见单位/颜色/品牌的同义词替换
"""

import random
from typing import List, Dict, Optional


def attribute_masking(entity_text: str, mask_prob: float = 0.15, mask_token: str = '[MASK]') -> str:
    """
    随机把若干 token 替换成 mask_token。

    >>> attribute_masking("Apple iPhone 8 Plus 64GB Silver", mask_prob=0.2)
    "Apple [MASK] 8 Plus [MASK] Silver"
    """
    tokens = entity_text.split()

    if len(tokens) == 0:
        return entity_text

    num_mask = max(1, int(len(tokens) * mask_prob))

    mask_indices = random.sample(range(len(tokens)), min(num_mask, len(tokens)))

    for idx in mask_indices:
        tokens[idx] = mask_token

    return ' '.join(tokens)


def token_shuffle(entity_text: str, keep_first: bool = True) -> str:
    """
    打乱词序。
    实体匹配通常不依赖词序，但首词常常是品牌之类的强信号，默认留住。
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
    随机扔掉若干 token，模拟不同源的字段缺失。
    至少留一个，免得退化成空串。
    """
    tokens = entity_text.split()

    if len(tokens) <= 1:
        return entity_text

    keep_mask = [random.random() > dropout_prob for _ in tokens]

    if not any(keep_mask):
        keep_mask[random.randint(0, len(keep_mask) - 1)] = True

    filtered_tokens = [token for token, keep in zip(tokens, keep_mask) if keep]

    return ' '.join(filtered_tokens)


def synonym_replacement(entity_text: str, synonym_dict: Optional[Dict[str, List[str]]] = None,
                        replace_prob: float = 0.5) -> str:
    """
    针对常见模式做同义替换。
    没传字典就用下面这套，主要覆盖电商/产品里常见的单位、颜色、尺寸、品牌。
    """
    if synonym_dict is None:
        synonym_dict = {
            # 容量单位
            'GB': ['Gigabyte', 'G', 'GiB'],
            'MB': ['Megabyte', 'M', 'MiB'],
            'TB': ['Terabyte', 'T', 'TiB'],

            # 颜色
            'silver': ['white', 'grey', 'gray'],
            'gold': ['golden', 'yellow'],
            'black': ['dark', 'noir'],
            'blue': ['navy', 'azure'],
            'red': ['crimson', 'scarlet'],

            # 尺寸 / 档位
            'plus': ['large', 'big', '+'],
            'mini': ['small', 'compact', 'tiny'],
            'pro': ['professional', 'premium'],

            # 数字写法
            '64': ['sixty-four', '64.0'],
            '128': ['one-twenty-eight', '128.0'],
            '256': ['two-fifty-six', '256.0'],

            # 品牌
            'iphone': ['iPhone', 'i-phone'],
            'samsung': ['Samsung', 'SAMSUNG'],
        }

    tokens = entity_text.split()

    for i, token in enumerate(tokens):
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
    """串多个增强方法，按 methods 顺序作用一遍"""
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
    给一条实体生成两个不同的增强视图，用于对比学习当正样本对。
    """
    view1 = augment_entity(entity_text, methods=method_set1)
    view2 = augment_entity(entity_text, methods=method_set2)

    return view1, view2


def batch_augment(entity_texts: List[str],
                  num_views: int = 2,
                  method_sets: Optional[List[List[str]]] = None) -> List[List[str]]:
    """
    批量生成 num_views 个视图。
    method_sets 不传就用一份默认搭配；不够用就循环复用。
    返回形状是 [num_views, batch_size]。
    """
    if method_sets is None:
        method_sets = [
            ['mask'],
            ['shuffle'],
            ['dropout'],
            ['mask', 'shuffle'],
            ['dropout', 'synonym'],
        ]

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


# 自检
if __name__ == '__main__':
    print("=" * 60)
    print("augmentation self-check")
    print("=" * 60)

    test_entity = "Apple iPhone 8 Plus 64GB Silver"
    print(f"\n原始: {test_entity}\n")

    print("1. attribute_masking:")
    for i in range(3):
        masked = attribute_masking(test_entity, mask_prob=0.2)
        print(f"   - {masked}")

    print("\n2. token_shuffle:")
    for i in range(3):
        shuffled = token_shuffle(test_entity, keep_first=True)
        print(f"   - {shuffled}")

    print("\n3. attribute_dropout:")
    for i in range(3):
        dropped = attribute_dropout(test_entity, dropout_prob=0.3)
        print(f"   - {dropped}")

    print("\n4. synonym_replacement:")
    for i in range(3):
        replaced = synonym_replacement(test_entity, replace_prob=0.5)
        print(f"   - {replaced}")

    print("\n5. mask + shuffle:")
    for i in range(3):
        augmented = augment_entity(test_entity, methods=['mask', 'shuffle'])
        print(f"   - {augmented}")

    print("\n6. create_augmented_pair:")
    for i in range(3):
        view1, view2 = create_augmented_pair(test_entity)
        print(f"   View 1: {view1}")
        print(f"   View 2: {view2}")
        print()

    print("\n7. batch_augment:")
    test_batch = [
        "Apple iPhone 8 Plus 64GB Silver",
        "Samsung Galaxy S8 64GB Black",
        "Huawei P10 32GB Gold"
    ]
    augmented_views = batch_augment(test_batch, num_views=2)
    print(f"   batch: {test_batch}")
    print(f"   View 1: {augmented_views[0]}")
    print(f"   View 2: {augmented_views[1]}")

    print("\n" + "=" * 60)
    print("done")
    print("=" * 60)
