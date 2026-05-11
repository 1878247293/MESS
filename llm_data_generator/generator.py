"""Stage 3：批量调 LLM 生成 entity groups + variants。"""

import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import DatasetConfig


GENERATE_SYSTEM_PROMPT = """你是一位合成数据生成专家，负责为实体匹配任务生成训练数据。
你必须严格遵守数据集生成规范中指定的每张表的字段格式。
请始终以 JSON 格式输出。

## 绝对禁止（违反任何一条则该实体组作废）

1. **禁止占位符/模板名称**：不允许出现 "Song Name"、"Album Name"、"The Artist"、"Unknown Artist"、"Unknown Song"、"Unknown Title"、"Unknown"、"未知歌名"、"未知艺人"、"ShopeeItem"、"虚构的首都"、"Fictional Capital"、"Product Name"、"Brand Name" 等通用占位符。每个实体必须有具体的、有辨识度的名称。
2. **禁止 null 字符串**：字段值不允许出现字面文本 "null"、"None"、"N/A"、"undefined"。如果某个字段按规范应该缺失，请使用空字符串 ""。
3. **禁止重复**：每个实体组的核心名称必须与其他实体组完全不同。下方"已有实体名"列表中的名称绝对不能再次出现。
4. **禁止混入不同实体**：同一个 entity group 内的所有 variants 必须是同一个实体（同一首歌/同一个人/同一件商品）的不同格式表示。绝对不允许将不同实体放入同一组。例如不允许把"钢琴曲"和"吉他曲"放在一组，不允许把"声卡"和"服装"放在一组。
5. **非空约束**：任何表的一个实体中最多只能有 1 个空字段。
6. **属性列一致性**：所有 variant 必须包含完全相同的属性列，缺失时用空字符串 ""，绝不允许某个 variant 多出或少了某个属性列。
7. **⚠️ 表间格式差异（最重要的差异维度）⚠️**：每个实体组的**大多数列（至少一半以上）**都必须体现表间差异，差异应频繁、普遍地出现在各个字段上，而非偶尔出现在少数列。不同实体组差异出现的列和差异方式应有所不同，使整体差异分布多样化。差异类型包括：拼写变体、缩写/截断、OCR错误（如数字 0↔O、1↔l、8↔B）、格式编码不同、信息嵌入方式不同、字段合并/拆分、词序调换等。你必须按照数据集生成规范中描述的各列**可能的**差异类型来选择差异方式。
8. **禁止格式趋同**：如果生成的多个 variant 各字段内容几乎相同（仅有大小写区别），则该实体组作废。每个 variant 相比 table_0 应在多个列上有明显的内容层面差异（不只是大小写变化）。同时也禁止所有实体组在相同的列上使用相同的固定差异模式——每个实体组的差异方式应有变化。
9. **大小写差异**：同一实体的不同 variant 之间必须体现大小写的变化（如首字母大写 vs 全大写 vs 全小写 vs 混合大小写）。
10. **语言一致性**：生成数据必须使用与数据集生成规范中指定的语言相同的语言，不允许自行切换语言。
10. **虚构实体要求**：允许生成虚构实体，但名称、属性等必须符合现实世界的风格和常识，不允许出现明显不合理的内容（如虚幻的地名、不存在的商品类别、荒诞的人名等）。"""


GENERATE_USER_TEMPLATE = """请为实体匹配数据集生成合成训练数据。

## 数据集信息
- 名称: {dataset_name}
- 数据列: {columns}
- 每个实体的文本格式: `{text_format}`

## 数据集生成规范（必须严格遵守）

{dataset_instructions}

## 已有实体名（禁止重复，必须生成完全不同的新实体）

{existing_sample}

## 多样性要求（极其重要）

本批请围绕以下主题生成：**{diversity_hint}**
- 每个实体的核心名称必须彼此不同，且不能与上方已有实体名重复
- 涵盖不同的子类别/风格/地区，不要集中在同一个领域

## 生成要求（必须严格执行）

1. 生成 {batch_size} 个实体组，每个实体核心名称各不相同
2. 每个实体组包含 {num_variants} 个 variant，分别对应 table_0 到 table_{last_table_idx}
3. **⚠️ 最重要 ⚠️ 同一个 entity group 的所有 variants 必须是【同一个实体】的不同表示方式（相同主体内容，不同格式/噪声/缺失风格）。例如：同一首歌的不同格式、同一个人的不同记录、同一件商品的不同描述。绝对不允许将不同实体放入同一组！**
4. **每个表的字段必须严格按照上方生成规范中对该表的格式要求，包括ID格式、title格式、length单位、缺失字段等**
5. **⚠️ 表间格式差异（核心要求）⚠️ 每个实体组的大多数列（至少一半以上）都必须体现表间差异**，差异应频繁、普遍地出现，而非偶尔出现在少数列。差异类型包括拼写变体、缩写/截断、格式编码不同、信息嵌入方式不同、字段合并/拆分、词序调换等。**不同实体组的差异方式应有变化**——禁止所有实体组在相同列上使用相同的固定差异模式。请参考上方生成规范中"表间差异模式"章节描述的各列可能的差异类型
6. **缺失字段用空字符串 ""**，绝对不允许出现字面 "null"、"None"、"N/A"
7. **非空约束：任何表的一个实体中最多只能有 1 个空字段**
8. **属性列一致性：所有 variant 必须包含完全相同的属性列，缺失时用空字符串 ""，绝不允许某个 variant 多出或少了某个属性列**
9. **语言一致性：生成数据必须使用与上方数据集生成规范中采样数据相同的语言，不允许自行切换语言（如规范中是印尼语则生成印尼语，是英文则生成英文）**
10. **⚠️ 再次强调：差异必须频繁出现 ⚠️：每个实体组的大多数列都应体现表间差异，不能只在一两个列上有变化而其余列完全相同。每个 variant 相比 table_0 至少应在一半以上的列上有实质性内容差异（拼写变体、缩写、OCR错误、格式变化等，不只是大小写）。不同实体组的差异方式应有变化，禁止所有实体组使用完全相同的差异模式。**

请返回如下 JSON 格式：
{{
    "entities": [
        {{
            {canonical_example}
            {extra_fields_example}
            "variants": [
                {variant_example}
            ]
        }}
    ]
}}"""


# 多样性提示词池：每批轮一个，让 LLM 别老盯着同一类实体
DIVERSITY_HINTS = {
    "Geo": [
        "尽量多样化，覆盖不同类别和风格",
        "与之前的实体尽量不同，探索新的子领域",
    ],
    "Music-2000": [
        "尽量多样化，覆盖不同类别和风格",
        "与之前的实体尽量不同，探索新的子领域",
    ],
    "Shopee": [
        "尽量多样化，覆盖不同类别和风格",
        "与之前的实体尽量不同，探索新的子领域",
    ],
    "Person": [
        "尽量多样化，覆盖不同类别和风格",
        "与之前的实体尽量不同，探索新的子领域",
    ],
}

# 词典里没有的就回退到这个
_DEFAULT_HINTS = [
    "尽量多样化，覆盖不同类别和风格",
    "与之前的实体尽量不同，探索新的子领域",
]


def _get_dataset_instructions(analysis: dict) -> str:
    return analysis.get("dataset_instructions", "")


def _build_example_fields(config: DatasetConfig) -> tuple:
    """根据 config 拼 prompt 里要塞进去的字段示例"""
    # canonical
    canonical_parts = []
    for field_name in config.canonical_fields:
        canonical_parts.append(f'"{field_name}": "..."')
    canonical_example = ",\n            ".join(canonical_parts) + ","

    # 组级额外字段
    extra_parts = []
    for f in config.group_extra_fields:
        extra_parts.append(f'"{f}": "..."')
    extra_example = ",\n            ".join(extra_parts) + "," if extra_parts else ""

    # variant
    variant_fields = []
    if config.has_style:
        variant_fields.append('"style": "table_0"')
    for col in config.text_fields:
        variant_fields.append(f'"{col}": "..."')
    variant_example = "{" + ", ".join(variant_fields) + "}, ..."

    return canonical_example, extra_example, variant_example


def _get_diversity_hint(dataset_name: str, batch_idx: int) -> str:
    hints = DIVERSITY_HINTS.get(dataset_name, _DEFAULT_HINTS)
    return hints[batch_idx % len(hints)]


def generate_entity_groups(client, config: DatasetConfig,
                           analysis: dict, num_groups: int,
                           batch_size: int,
                           existing_names: set = None,
                           max_workers: int = 4) -> list:
    """
    分批跑 LLM 出 entity groups，多线程并发。
    用 lock 保护 existing_names 和 all_groups，去重在写回时做。
    """
    print(f"Stage 3: 生成 {num_groups} 个 entity groups "
          f"(batch={batch_size}, workers={max_workers})...")

    existing = existing_names or set()
    all_groups = []
    lock = threading.Lock()
    batch_idx = 0
    consecutive_failures = 0
    max_consecutive_failures = 30

    def _run_batch(bid):
        with lock:
            current_existing = set(existing)
        hint = _get_diversity_hint(config.name, bid)
        try:
            batch = _generate_batch(
                client, config, analysis,
                batch_size, current_existing, hint,
            )
            return bid, batch, hint, None
        except RuntimeError as e:
            return bid, [], hint, e

    while len(all_groups) < num_groups:
        if consecutive_failures >= max_consecutive_failures:
            print(f"  连续 {consecutive_failures} 批没产出新实体，提前停")
            break

        remaining = num_groups - len(all_groups)
        n_concurrent = min(max_workers, -(-remaining // batch_size))
        batch_ids = list(range(batch_idx, batch_idx + n_concurrent))
        batch_idx += n_concurrent

        print(f"  并发 {n_concurrent} 批 "
              f"(batch {batch_ids[0]}-{batch_ids[-1]}, "
              f"已有 {len(all_groups)}/{num_groups})...")

        round_accepted = 0
        with ThreadPoolExecutor(max_workers=n_concurrent) as executor:
            futures = {executor.submit(_run_batch, bid): bid for bid in batch_ids}
            for future in as_completed(futures):
                bid, batch, hint, error = future.result()
                if error:
                    print(f"    batch {bid} 失败: {error}")
                    continue

                with lock:
                    deduped = []
                    for group in batch:
                        name_key = _get_canonical_key(group, config)
                        if name_key and name_key not in existing:
                            deduped.append(group)
                            existing.add(name_key)
                        else:
                            print(f"    去重: 丢掉 '{name_key}'")

                    for group in deduped:
                        if len(all_groups) >= num_groups:
                            break
                        group["group_id"] = len(all_groups)
                        all_groups.append(group)

                    accepted = len(deduped)
                    total = len(batch)
                    round_accepted += accepted

                if accepted < total:
                    print(f"    batch {bid}: 去重后 {accepted}/{total} 通过")
                elif accepted > 0:
                    print(f"    batch {bid}: {accepted} 通过")

        if round_accepted > 0:
            consecutive_failures = 0
        else:
            consecutive_failures += n_concurrent

    print(f"  共 {len(all_groups)} 个 entity groups")
    return all_groups


def _get_canonical_key(group: dict, config: DatasetConfig) -> str:
    """canonical 字段拼起来当去重 key，括号内容剥掉"""
    parts = []
    for field_name in config.canonical_fields:
        val = group.get(field_name, "").strip().lower()
        # 例：Cairo (Egypt) -> cairo
        import re
        val_clean = re.sub(r'\s*\(.*?\)\s*', '', val).strip()
        parts.append(val_clean)
    return " | ".join(parts) if any(parts) else ""


def _generate_batch(client, config: DatasetConfig,
                    analysis: dict, batch_size: int,
                    existing_names: set, diversity_hint: str) -> list:
    dataset_instructions = _get_dataset_instructions(analysis)
    canonical_example, extra_example, variant_example = _build_example_fields(config)

    # 已有实体名最多塞 50 个进 prompt，让 LLM 知道哪些不能再生
    if existing_names:
        sample_size = min(50, len(existing_names))
        sampled = random.sample(sorted(existing_names), sample_size)
        existing_sample = ", ".join(sampled)
        if len(existing_names) > sample_size:
            existing_sample += f"\n...等共 {len(existing_names)} 个已有实体"
    else:
        existing_sample = "（暂无，第一批）"

    user_prompt = GENERATE_USER_TEMPLATE.format(
        dataset_name=config.name,
        columns=", ".join(config.columns),
        text_format=config.text_format,
        dataset_instructions=dataset_instructions,
        batch_size=batch_size,
        num_variants=config.default_variants,
        last_table_idx=config.default_variants - 1,
        existing_sample=existing_sample,
        diversity_hint=diversity_hint,
        canonical_example=canonical_example,
        extra_fields_example=extra_example,
        variant_example=variant_example,
    )

    messages = [
        {"role": "system", "content": GENERATE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    result = client.chat_json(messages)

    entities = result.get("entities", [])
    if not entities:
        print(f"    LLM 返回空 entities")
        return []

    valid_groups = []
    for entity in entities:
        group = _validate_entity(entity, config)
        if group:
            valid_groups.append(group)

    if len(valid_groups) < len(entities):
        print(f"    校验: {len(valid_groups)}/{len(entities)} 通过")

    return valid_groups


# 黑名单：canonical 全等于其中之一的直接丢
_PLACEHOLDER_NAMES = {
    # 通用
    "unknown", "n/a", "none", "null", "test", "example", "sample",
    # 音乐
    "song name", "album name", "the artist", "unknown artist",
    "unknown song", "unknown title", "artist name", "band name", "track name",
    # 地理
    "city name", "location name", "place name", "fictional capital",
    "虚构的首都", "虚构国家首都", "虚构的都城", "虚构城市", "虚构国家",
    "fictitious metropolis", "fictitious capital", "fictional city",
    # 电商
    "product name", "brand name", "item name", "shopee item",
}

# canonical 含其中之一的子串也丢
_PLACEHOLDER_SUBSTRINGS = [
    "shopeeitem", "shopeeoriginal", "虚构",
    "fictional", "fictitious", "未知",
]


def _validate_entity(entity: dict, config: DatasetConfig) -> dict:
    """
    校验 + 规范化单个 entity group。
    通不过返回 None，主调把它丢掉。
    """
    # canonical 字段：缺失就从第一个 variant 推一下
    group = {}
    for field_name in config.canonical_fields:
        val = entity.get(field_name, "")
        if not val:
            variants = entity.get("variants", [])
            source_col = config.canonical_fields[field_name]
            if variants and source_col in variants[0]:
                val = variants[0][source_col]
        group[field_name] = str(val) if val else ""

    # 占位符筛掉
    for field_name in config.canonical_fields:
        val = group.get(field_name, "").strip().lower()
        if val in _PLACEHOLDER_NAMES:
            return None
        for sub in _PLACEHOLDER_SUBSTRINGS:
            if sub in val:
                return None

    # 组级额外字段
    for f in config.group_extra_fields:
        group[f] = entity.get(f, "")

    # variants
    variants = entity.get("variants", [])
    if not variants:
        return None

    valid_variants = []
    for v in variants:
        # 至少有一个 text_field
        has_required = any(f in v for f in config.text_fields)
        if has_required:
            clean_v = {}
            if config.has_style and "style" in v:
                clean_v["style"] = v["style"]
            elif config.has_style:
                clean_v["style"] = f"table_{len(valid_variants)}"
            for f in config.text_fields:
                raw = v.get(f)
                # null / None / N/A 之类的字面量统一成空串
                if raw is None or str(raw).strip().lower() in ("null", "none", "n/a"):
                    clean_v[f] = ""
                else:
                    clean_v[f] = str(raw)
            # 空字段超过 2 个就丢
            empty_count = sum(1 for f in config.text_fields if not clean_v.get(f, "").strip())
            if empty_count > 2:
                continue
            valid_variants.append(clean_v)

    if not valid_variants:
        return None

    # variant 数量必须刚好等于表数量
    expected = config.default_variants
    if len(valid_variants) != expected:
        return None

    group["variants"] = valid_variants
    return group
