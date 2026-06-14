"""Stage 3: batch-call the LLM to generate entity groups + variants."""

import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import DatasetConfig


GENERATE_SYSTEM_PROMPT = """You are an expert in synthetic data generation, responsible for producing training data for entity matching tasks.
You must strictly follow the per-table field formats specified in the dataset generation specification.
Always output in JSON format.

## Absolutely forbidden (violating any single rule invalidates the entity group)

1. **No placeholder / template names**: Generic placeholders such as "Unknown Artist", "Unknown Song", "Unknown Title", "Unknown" are not allowed. Every entity must have a concrete, distinctive name.
2. **No null strings**: Field values must not contain the literal text "null", "None", "N/A", or "undefined". If a field should be missing per the specification, use an empty string "".
3. **No duplicates**: The core name of each entity group must be completely different from every other entity group. Names in the "existing entity names" list below must never appear again.
4. **No mixing of different entities**: All variants within the same entity group must be different format representations of the SAME entity (the same song / the same person / the same product). Putting different entities into one group is strictly forbidden. For example, do not put a "piano piece" and a "guitar piece" in one group, and do not put a "sound card" and "clothing" in one group.
5. **Non-empty constraint**: Any single entity in any table may have at most 1 empty field.
6. **Attribute column consistency**: All variants must contain exactly the same attribute columns; use an empty string "" when a value is missing. No variant may have an extra or a missing attribute column.
7. **Cross-table format differences (core dimension)**: Refer to the difference distribution pattern described in the dataset generation specification — typically each entity group exhibits cross-table differences on only some columns (about 2-3 columns), while the remaining columns may keep the same format across tables. Difference types usually include: different format encodings, different ways of embedding information, different field-omission strategies, etc. Choose difference methods according to the **possible** difference types described for each column in the specification, and try to vary the difference columns and methods across different entity groups.
8. **Avoid format convergence**: If the generated variants have nearly identical field contents (differing only in case), the entity group will be considered invalid. Each variant should differ from table_0 at the content level on the columns that express differences (not merely a case change). Also avoid having all entity groups use the same fixed difference pattern on the same columns — vary the difference methods across entity groups.
9. **Case differences**: Different variants of the same entity must exhibit case variation (e.g. Title Case vs ALL CAPS vs all lowercase vs MiXeD case).
10. **Language consistency**: The generated data must use the same language as specified in the dataset generation specification; do not switch languages on your own.
11. **Fictional entity requirement**: Fictional entities are allowed, but their names, attributes, etc. must conform to real-world style and common sense. Obviously unreasonable content is not allowed (e.g. fantastical place names, non-existent product categories, absurd person names)."""


GENERATE_USER_TEMPLATE = """Please generate synthetic training data for an entity matching dataset.

## Dataset information
- Name: {dataset_name}
- Columns: {columns}
- Text format of each entity: `{text_format}`

## Dataset generation specification (must be strictly followed)

{dataset_instructions}

## Existing entity names (duplicates forbidden; you must generate completely new, different entities)

{existing_sample}

## Diversity requirement (extremely important)

This batch should be generated around the following theme: **{diversity_hint}**
- The core name of each entity must be distinct from one another and must not duplicate any existing entity name above
- Cover different subcategories / styles / regions; do not concentrate on a single domain

## Generation requirements (must be strictly enforced)

1. Generate {batch_size} entity groups, each with a distinct core name
2. Each entity group contains {num_variants} variants, corresponding to table_0 through table_{last_table_idx}
3. **⚠️ MOST IMPORTANT ⚠️ All variants of the same entity group must be different representations of THE SAME entity (same underlying content, different format / noise / omission style). For example: different formats of the same song, different records of the same person, different descriptions of the same product. Putting different entities into one group is strictly forbidden!**
4. **The fields of each table must strictly follow the format requirements for that table in the generation specification above, including ID format, title format, length units, missing fields, etc.**
5. **Cross-table format differences (core dimension)**: Refer to the difference types described for each column in the "cross-table difference patterns" section of the specification above — typically each entity group exhibits differences on only some columns (about 2-3 columns), while the remaining columns may keep the same format across tables. Difference types usually include different format encodings, different ways of embedding information, different field-omission strategies, etc. Try to vary the difference columns and methods across entity groups, and avoid having all entity groups use the same fixed difference pattern on the same columns
6. **Use an empty string "" for missing fields**; the literals "null", "None", "N/A" are strictly forbidden
7. **Non-empty constraint: any single entity in any table may have at most 1 empty field**
8. **Attribute column consistency: all variants must contain exactly the same attribute columns; use an empty string "" when missing. No variant may have an extra or a missing attribute column**
9. **Language consistency: the generated data must use the same language as the sampled data in the dataset generation specification above; do not switch languages on your own (e.g. if the specification is in Indonesian, generate Indonesian; if in English, generate English)**
10. **Difference distribution note**: Refer to the difference distribution pattern described in the dataset generation specification — typically each entity group exhibits substantive content-level differences on about 2-3 columns (different format encodings, different ways of embedding information, different field-omission strategies), while the remaining columns may keep the same format across tables. Try to vary the difference columns and methods across entity groups, and avoid having all entity groups use exactly the same difference pattern.

Please return JSON in the following format:
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


# Diversity hint pool: rotate one per batch so the LLM doesn't fixate on the same kind of entity
DIVERSITY_HINTS = [
    "Be as diverse as possible, covering different categories and styles",
    "Differ as much as possible from previous entities, exploring new subdomains",
]


def _get_dataset_instructions(analysis: dict) -> str:
    return analysis.get("dataset_instructions", "")


def _build_example_fields(config: DatasetConfig) -> tuple:
    """Build the field examples to inject into the prompt, based on config"""
    # canonical
    canonical_parts = []
    for field_name in config.canonical_fields:
        canonical_parts.append(f'"{field_name}": "..."')
    canonical_example = ",\n            ".join(canonical_parts) + ","

    # group-level extra fields
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


def _get_diversity_hint(batch_idx: int) -> str:
    return DIVERSITY_HINTS[batch_idx % len(DIVERSITY_HINTS)]


def generate_entity_groups(client, config: DatasetConfig,
                           analysis: dict, num_groups: int,
                           batch_size: int,
                           existing_names: set = None,
                           max_workers: int = 4) -> list:
    """
    Run the LLM in batches to produce entity groups, concurrently across threads.
    A lock protects existing_names and all_groups; deduplication is done on write-back.
    """
    print(f"Stage 3: generating {num_groups} entity groups "
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
        hint = _get_diversity_hint(bid)
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
            print(f"  {consecutive_failures} consecutive batches produced no new entities; stopping early")
            break

        remaining = num_groups - len(all_groups)
        n_concurrent = min(max_workers, -(-remaining // batch_size))
        batch_ids = list(range(batch_idx, batch_idx + n_concurrent))
        batch_idx += n_concurrent

        print(f"  {n_concurrent} concurrent batches "
              f"(batch {batch_ids[0]}-{batch_ids[-1]}, "
              f"have {len(all_groups)}/{num_groups})...")

        round_accepted = 0
        with ThreadPoolExecutor(max_workers=n_concurrent) as executor:
            futures = {executor.submit(_run_batch, bid): bid for bid in batch_ids}
            for future in as_completed(futures):
                bid, batch, hint, error = future.result()
                if error:
                    print(f"    batch {bid} failed: {error}")
                    continue

                with lock:
                    deduped = []
                    for group in batch:
                        name_key = _get_canonical_key(group, config)
                        if name_key and name_key not in existing:
                            deduped.append(group)
                            existing.add(name_key)
                        else:
                            print(f"    dedup: dropping '{name_key}'")

                    for group in deduped:
                        if len(all_groups) >= num_groups:
                            break
                        group["group_id"] = len(all_groups)
                        all_groups.append(group)

                    accepted = len(deduped)
                    total = len(batch)
                    round_accepted += accepted

                if accepted < total:
                    print(f"    batch {bid}: {accepted}/{total} passed after dedup")
                elif accepted > 0:
                    print(f"    batch {bid}: {accepted} passed")

        if round_accepted > 0:
            consecutive_failures = 0
        else:
            consecutive_failures += n_concurrent

    print(f"  {len(all_groups)} entity groups total")
    return all_groups


def _get_canonical_key(group: dict, config: DatasetConfig) -> str:
    """Join the canonical fields into a dedup key, stripping parenthesized content"""
    parts = []
    for field_name in config.canonical_fields:
        val = group.get(field_name, "").strip().lower()
        # e.g. Cairo (Egypt) -> cairo
        import re
        val_clean = re.sub(r'\s*\(.*?\)\s*', '', val).strip()
        parts.append(val_clean)
    return " | ".join(parts) if any(parts) else ""


def _generate_batch(client, config: DatasetConfig,
                    analysis: dict, batch_size: int,
                    existing_names: set, diversity_hint: str) -> list:
    dataset_instructions = _get_dataset_instructions(analysis)
    canonical_example, extra_example, variant_example = _build_example_fields(config)

    # inject at most 50 existing entity names into the prompt so the LLM knows which not to regenerate
    if existing_names:
        sample_size = min(50, len(existing_names))
        sampled = random.sample(sorted(existing_names), sample_size)
        existing_sample = ", ".join(sampled)
        if len(existing_names) > sample_size:
            existing_sample += f"\n...and {len(existing_names)} existing entities in total"
    else:
        existing_sample = "(none yet, first batch)"

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
        print(f"    LLM returned empty entities")
        return []

    valid_groups = []
    for entity in entities:
        group = _validate_entity(entity, config)
        if group:
            valid_groups.append(group)

    if len(valid_groups) < len(entities):
        print(f"    validation: {len(valid_groups)}/{len(entities)} passed")

    return valid_groups


# denylist: drop a group whose canonical exactly equals one of these
_PLACEHOLDER_NAMES = {
    # generic
    "unknown", "n/a", "none", "null", "test", "example", "sample",
}

# also drop if the canonical contains one of these substrings
_PLACEHOLDER_SUBSTRINGS = [
    "shopeeitem", "shopeeoriginal",
    "fictional", "fictitious",
]


def _validate_entity(entity: dict, config: DatasetConfig) -> dict:
    """
    Validate + normalize a single entity group.
    Returns None if it fails validation, and the caller drops it.
    """
    # canonical fields: if missing, infer from the first variant
    group = {}
    for field_name in config.canonical_fields:
        val = entity.get(field_name, "")
        if not val:
            variants = entity.get("variants", [])
            source_col = config.canonical_fields[field_name]
            if variants and source_col in variants[0]:
                val = variants[0][source_col]
        group[field_name] = str(val) if val else ""

    # filter out placeholders
    for field_name in config.canonical_fields:
        val = group.get(field_name, "").strip().lower()
        if val in _PLACEHOLDER_NAMES:
            return None
        for sub in _PLACEHOLDER_SUBSTRINGS:
            if sub in val:
                return None

    # group-level extra fields
    for f in config.group_extra_fields:
        group[f] = entity.get(f, "")

    # variants
    variants = entity.get("variants", [])
    if not variants:
        return None

    valid_variants = []
    for v in variants:
        # must have at least one text_field
        has_required = any(f in v for f in config.text_fields)
        if has_required:
            clean_v = {}
            if config.has_style and "style" in v:
                clean_v["style"] = v["style"]
            elif config.has_style:
                clean_v["style"] = f"table_{len(valid_variants)}"
            for f in config.text_fields:
                raw = v.get(f)
                # normalize literals like null / None / N/A to an empty string
                if raw is None or str(raw).strip().lower() in ("null", "none", "n/a"):
                    clean_v[f] = ""
                else:
                    clean_v[f] = str(raw)
            # drop if more than 2 empty fields
            empty_count = sum(1 for f in config.text_fields if not clean_v.get(f, "").strip())
            if empty_count > 2:
                continue
            valid_variants.append(clean_v)

    if not valid_variants:
        return None

    # the number of variants must exactly equal the number of tables
    expected = config.default_variants
    if len(valid_variants) != expected:
        return None

    group["variants"] = valid_variants
    return group
