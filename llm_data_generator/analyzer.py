"""Stage 2: let the LLM look at the sampled data and produce a generation specification in markdown form.

The LLM is called only once. The output is read directly by the next stage, so the specification
must describe field formats, cross-table differences, noise types, etc. in sufficient detail.
"""

import json

from .config import DatasetConfig, GeneratorConfig
from .sampler import format_samples_for_prompt


ANALYSIS_SYSTEM_PROMPT = """You are a senior data engineer and synthetic-data specification designer.
Your task: carefully analyze the sampled data of each table in an entity matching dataset, and produce an extremely detailed generation specification document in Markdown format.

This document will be read directly by another LLM and used to generate synthetic training data, so you must ensure:
- **Field-level precision**: every field of every table must have its own format description, regex pattern, and concrete example values
- **Difference-driven**: focus on observing and describing the format differences that **may occur** for the same field across tables (this is the core challenge of entity matching). Cross-table differences usually include: different format encodings, different ways of embedding information, different field-omission strategies, etc. Case differences are only one relatively minor dimension; it is recommended not to use them as the sole cross-table difference
- **Difference distribution pattern**: it is recommended to first observe and summarize the actual occurrence pattern of cross-table differences in the original dataset — typically, differences do not occur fixedly on all columns simultaneously, but tend to be spread across some columns (about 2-3 columns) within each entity group, while the remaining columns may keep the same format across tables; the columns where differences occur also tend to vary across entity groups, making the overall difference distribution more diverse. In the specification, describe the difference types that **may occur** for each column accordingly, rather than mandating that each column **must** exhibit a difference
- **Same-entity constraint (most important)**: you must repeatedly emphasize in the specification that all variants of the same entity group are different format representations of the same entity (the same song / the same person / the same product), never different entities
- **Non-empty constraint**: the specification for each table must note that any single entity in any table may have at most 1 empty field
- **Attribute column consistency**: all variants must contain exactly the same attribute columns; use an empty string "" when missing. No variant may have an extra or a missing attribute column
- **Case differences**: different variants of the same entity must exhibit case variation (e.g. Title Case vs ALL CAPS vs all lowercase vs MiXeD case)
- **Language consistency**: you must identify the language used in the sampled data (e.g. English, Indonesian, Chinese, etc.) and explicitly require in the specification that the generated data use the same language as the sampled data; switching languages is not allowed
- **Cover noise**: explicitly list all noise types present in the data and where they occur
- **Rich examples**: every format description must have 2-3 realistic-style example values
- **Absolutely no vagueness**: vague statements such as "various formats" or "depends on the situation" are not allowed; everything must be concrete

Output Markdown text directly — no JSON, no code-block markers."""


ANALYSIS_USER_TEMPLATE = """Please analyze the sampled data of each table in the following entity matching dataset, then produce a complete, field-level generation specification.

## Dataset information
- Name: {dataset_name}
- Description: {description}
- Number of tables: {num_tables}
- Columns: {columns}

## Sampled records per table
{formatted_samples}

---

## Please output a Markdown specification document following the structure below

Refer to the **Hotel-300 demonstration** below, and produce a specification of equal detail for the current dataset **{dataset_name}**.
Your output must include all of the following sections:

### Required sections

1. **`## {{dataset_name}} dataset-specific requirements`** — dataset-level overall requirements (fictional entities allowed but must conform to real-world style and common sense, the meaning of variants, etc.)
2. **`### Per-table field format specification`** — one subsection per table `**Table X** (brief feature description)`; within each subsection, for **every field** give, one by one:
   - Format pattern (e.g. "prefix + N digits", "MM:SS", "decimal minutes")
   - 2-3 concrete example values
   - Missing behavior (whether this field is often an empty string in this table)
   - Key differences from the same field in other tables
   - **Non-empty constraint**: the specification for each table must note that any single entity in any table may have at most 1 (or 0) empty fields
3. **`### Cross-table difference patterns`** — list all systematic cross-table format differences (e.g. case, abbreviation, encoding, omission strategy, etc.); for each difference, indicate which tables and fields are involved
4. **`### Noise types`** — list all noise present in the data (typos, truncation, input errors, etc.); for each noise type, give specific locations of occurrence and examples
5. **`### Special notes`** — **the first item must be**: "All variants of the same entity group must be different format representations of the same entity (e.g. the same song, the same person, the same product); different entities must never be mixed into one group." Then list: the correspondence that must be maintained between variants, common mistakes, and key constraints during generation

---

=== Demonstration: Hotel-300 dataset-specific generation specification ===

## Hotel-300 dataset-specific requirements
- Fictional hotel records are allowed, but names, locations, etc. must conform to real-world style and common sense; obviously unreasonable content is not allowed (e.g. "Moon Hotel", "Quantum Space Inn")
- The 4 variants within each entity group must be different representations of the same hotel (various format variants of the same hotel name)
- **Variants across different tables must have substantive format differences** (spelling variants, abbreviations, truncation, OCR errors, different format encodings, different ways of embedding information, etc.), not merely case differences
- **Different variants of the same entity must also exhibit case differences**
- **All variants must contain exactly the same 5 attribute columns** (name, city, stars, price, country); use an empty string `""` when missing

### Per-table field format specification

**Table 0** (the most complete standard table, Title Case style)
- `name`: the full hotel name in Title Case, e.g. `"Grand Pacific Hotel"`, `"Sunset Bay Resort"`, `"Alpine Lodge & Spa"`
  - Missing behavior: always non-empty
  - Differences from other tables: Table 1 is ALL CAPS with embedded city, Table 2 is all lowercase with typos, Table 3 is mixed case with marketing words
- `city`: full city name, Title Case, e.g. `"San Francisco"`, `"New York"`, `"Buenos Aires"`
  - Missing behavior: always non-empty
  - Differences from other tables: Table 1 embeds it in the name parentheses and leaves this field empty, Table 2 uses a city abbreviation such as `"SF"`, Table 3 uses the full city name in all lowercase
- `stars`: integer star-rating string, e.g. `"5"`, `"3"`, `"4"`
  - Missing behavior: always non-empty
  - Differences from other tables: Table 1 uses English text such as `"Five Star"`, Table 2 uses symbols such as `"***"`, Table 3 uses an empty string
- `price`: US dollar price with a $ prefix, e.g. `"$350"`, `"$120"`, `"$89"`
  - Missing behavior: always non-empty
  - Differences from other tables: Table 1 uses a euro price such as `"EUR 320"`, Table 2 uses plain digits such as `"350"`, Table 3 uses an empty string
- `country`: full country name, Title Case, e.g. `"United States"`, `"France"`, `"Argentina"`
  - Missing behavior: always non-empty
  - Differences from other tables: Table 1 uses a 2-letter country code such as `"US"`, Table 2 uses a 3-letter code such as `"USA"`, Table 3 uses an empty string
- **Non-empty constraint**: all 5 fields of this table are non-empty
- **Attribute columns**: name, city, stars, price, country (identical to other tables)

**Table 1** (ALL CAPS name, city embedded in name, many format variants)
- `name`: ALL CAPS + city embedded in parentheses, e.g. `"GRAND PACIFIC HOTEL (SAN FRANCISCO)"`, `"SUNSET BAY RESORT (NEW YORK)"`, `"ALPINE LODGE & SPA (BUENOS AIRES)"`
  - Missing behavior: always non-empty
  - Differences from other tables: Table 0 is Title Case without embedded city, Table 2 is all lowercase, Table 3 is mixed case
- `city`: empty string `""` (the city information is already embedded in the name parentheses)
  - Missing behavior: occasionally an empty string (information moved into name)
  - Differences from other tables: Table 0 full city name, Table 2 abbreviation, Table 3 all-lowercase city name
- `stars`: star rating as English text, e.g. `"Five Star"`, `"Three Star"`, `"Four Star"`
  - Missing behavior: always non-empty
  - Differences from other tables: Table 0 uses a number such as `"5"`, Table 2 uses symbols such as `"***"`, Table 3 is empty
- `price`: euro price format, e.g. `"EUR 320"`, `"EUR 110"`, `"EUR 80"`
  - Missing behavior: always non-empty
  - Differences from other tables: Table 0 uses dollars with $ such as `"$350"`, Table 2 uses plain digits, Table 3 is empty
- `country`: 2-letter ISO country code, e.g. `"US"`, `"FR"`, `"AR"`
  - Missing behavior: always non-empty
  - Differences from other tables: Table 0 full country name, Table 2 three-letter code, Table 3 is empty
- **Non-empty constraint**: at most 1 field of this table is empty (only city is empty, the rest have values)
- **Attribute columns**: name, city, stars, price, country (identical to other tables)

**Table 2** (all lowercase, abbreviations, contains typos)
- `name`: all lowercase, may contain typos (missing letters / swapped letters), e.g. `"grand pacfic hotel"`, `"sunet bay resort"`, `"alpien lodge & spa"`
  - Missing behavior: always non-empty
  - Differences from other tables: Table 0 is Title Case without errors, Table 1 is ALL CAPS, Table 3 is mixed case
- `city`: city abbreviation (ALL CAPS), e.g. `"SF"`, `"NY"`, `"BA"`
  - Missing behavior: always non-empty
  - Differences from other tables: Table 0 full city name, Table 1 embedded in name with this field empty, Table 3 all-lowercase city name
- `stars`: star rating as symbols, e.g. `"***"`, `"*****"`, `"****"`
  - Missing behavior: always non-empty
  - Differences from other tables: Table 0 number such as `"5"`, Table 1 English such as `"Five Star"`, Table 3 is empty
- `price`: plain digit string (no currency prefix), e.g. `"350"`, `"120"`, `"89"`
  - Missing behavior: always non-empty
  - Differences from other tables: Table 0 has a $ prefix, Table 1 euro format, Table 3 is empty
- `country`: 3-letter ISO country code, e.g. `"USA"`, `"FRA"`, `"ARG"`
  - Missing behavior: always non-empty
  - Differences from other tables: Table 0 full country name, Table 1 two-letter code, Table 3 is empty
- **Non-empty constraint**: all 5 fields of this table are non-empty
- **Attribute columns**: name, city, stars, price, country (identical to other tables)

**Table 3** (mixed case, marketing words, only keeps name and city)
- `name`: mixed case + marketing words/symbol suffix, e.g. `"grand PACIFIC Hotel ★★★★★"`, `"SUNSET bay Resort - Best Deal"`, `"Alpine LODGE & spa [Hot Sale]"`
  - Missing behavior: always non-empty
  - Differences from other tables: Table 0 Title Case, Table 1 ALL CAPS, Table 2 all lowercase
- `city`: full city name in all lowercase, e.g. `"san francisco"`, `"new york"`, `"buenos aires"`
  - Missing behavior: always non-empty
  - Differences from other tables: Table 0 Title Case, Table 1 empty (embedded in name), Table 2 abbreviation
- `stars`: empty string `""`
  - Missing behavior: occasionally an empty string
  - Differences from other tables: Table 0 number, Table 1 English text, Table 2 symbols
- `price`: empty string `""`
  - Missing behavior: occasionally an empty string
  - Differences from other tables: Table 0 dollars, Table 1 euros, Table 2 plain digits
- `country`: empty string `""`
  - Missing behavior: occasionally an empty string
  - Differences from other tables: Table 0 full country name, Table 1 two-letter code, Table 2 three-letter code
- **Non-empty constraint**: name and city of this table are always non-empty; stars/price/country are occasionally empty
- **Attribute columns**: name, city, stars, price, country (identical to other tables)

### Cross-table difference patterns (difference distribution pattern)

**Overall idea**: the difference dimensions below are the difference types that **may occur** for each column, and can serve as a reference. It is recommended to first observe and summarize the actual difference distribution pattern in the original dataset — typically each entity group exhibits differences on only about 2-3 columns, while the remaining columns may keep the same format across tables; the columns where differences occur also tend to vary across entity groups.

- **Case differences (may occur in the name/city/country columns)**: the same content uses different case styles across tables, e.g. Title Case `"Grand Pacific Hotel"` vs ALL CAPS `"GRAND PACIFIC HOTEL"` vs all lowercase `"grand pacific hotel"` vs mixed case `"grand PACIFIC Hotel"`
- **City format (may occur in the city column)**: full city name `"San Francisco"` vs embedded in name parentheses vs abbreviation `"SF"` vs all lowercase `"san francisco"`
- **Star format (may occur in the stars column)**: number `"5"` vs English `"Five Star"` vs symbols `"*****"` vs empty
- **Price format (may occur in the price column)**: dollars `"$350"` vs euros `"EUR 320"` vs plain digits `"350"` vs empty
- **Country format (may occur in the country column)**: full country name `"United States"` vs two-letter `"US"` vs three-letter `"USA"` vs empty
- **Typos (may randomly occur in any text column)**: missing letters, swapped letters, etc.
- **Field omission (randomly distributed)**: across the different tables of each entity group, 0-1 fields may be randomly empty, but it is not the case that a fixed column of a fixed table is always empty
- **Attribute column consistency**: all 4 tables must contain exactly the same 5 attribute columns (name, city, stars, price, country); fill missing fields with an empty string `""`

**Examples**:
- Entity A: the name column shows case differences + the city column shows abbreviation differences, the remaining columns have the same format
- Entity B: the price column shows format differences + the name column contains a typo, the remaining columns have the same format
- Entity C: the country column shows abbreviation differences + the stars column shows format differences, the remaining columns have the same format

### Noise types
- **Case variation**: the same name appears as Title Case / ALL CAPS / all lowercase / mixed case across different tables, e.g. `"Grand Pacific"` vs `"GRAND PACIFIC"` vs `"grand pacific"` vs `"grand PACIFIC"`
- **Typos**: appear in the name of Table 2 — missing letters (`"Pacific"` → `"Pacfic"`), swapped letters (`"Sunset"` → `"Sunet"`), extra spaces
- **Abbreviations**: city name `"San Francisco"` → `"SF"` (Table 2), country name `"United States"` → `"US"` (Table 1)
- **Marketing words/symbol insertion**: appear in the name of Table 3, e.g. extra text such as `"★★★★★"`, `"Best Deal"`, `"[Hot Sale]"`
- **Format differences**: price `"$350"` vs `"350"` — the same information represented differently

### Special notes
- **[Most important] The variants of the same entity group must be different format representations of the same hotel; putting different hotels into one group is strictly forbidden**
- **Attribute column consistency**: all variants must contain exactly the same attribute columns (name, city, stars, price, country); use an empty string `""` when missing. No variant may have an extra or a missing attribute column
- **Cross-table format differences (core dimension)**: variants across different tables must have substantive content differences (spelling variants, abbreviations, truncation, OCR errors, different format encodings, different ways of embedding information, etc.); case differences alone are not enough
- **Difference distribution pattern (core dimension)**: typically each entity group exhibits cross-table differences on only about 2-3 columns, while the remaining columns may keep the same format; try to vary the difference columns across entity groups to make the overall difference distribution more diverse. Avoid having all entity groups use the same fixed difference pattern on the same columns
- **Case differences**: different variants of the same entity must exhibit case variation (e.g. Title Case vs ALL CAPS vs all lowercase vs mixed case)
- **Non-empty constraint**: the specification for each table must note that any single entity in any table may have at most 1 empty field
- **Noise coverage**: must include multiple noise types such as typos (missing letters / swapped letters), abbreviations, marketing-word insertion, format differences, etc.
- **Language consistency**: the generated data must use the same language as the sampled data; switching languages is not allowed (e.g. if the sampled data is in Indonesian, generate Indonesian; if in English, generate English)

Now please produce a specification of equal detail and completeness for the **{dataset_name}** dataset. Requirements:
1. List one subsection per table **Table X** (brief feature description)
2. Within each subsection, give concrete format requirements for **every field** one by one (including format pattern, 2-3 example values, missing behavior, differences from other tables)
3. Must include the **Cross-table difference patterns** section, listing all possible systematic cross-table differences. **It is recommended to first observe and summarize the actual difference distribution pattern in the original dataset** — typically differences are not concentrated on fixed columns, but spread across multiple columns; each entity group generally exhibits differences on about 2-3 columns, while the remaining columns may keep the same format across tables. The specification should describe the difference types that **may occur** for each column, rather than mandating that each column **must** exhibit a difference
4. Must include the **Noise types** section, listing all noise along with their locations and examples
5. Must include the **Special notes** section, listing the generation constraints
6. The tone must be precise and concrete; every field must have concrete example values; do not say vague things
7. Extract as much detail as possible from the sampled data; do not omit any observable pattern
8. **The first item of the "Special notes" section must be**: all variants of the same entity group must be different format representations of the same entity (different format / noise / omission variants); different entities must never be mixed into one group
9. **Non-empty constraint**: the specification for each table must note that any single entity in any table may have at most 1 empty field
10. **Attribute column consistency**: the specification must explicitly require all variants to contain exactly the same attribute columns; use an empty string "" when missing. No variant may have an extra or a missing attribute column
11. **Cross-table format differences (core requirement)**: the specification must explicitly require substantive content differences between variants across different tables (spelling variants, abbreviations, truncation, OCR errors, different format encodings, different ways of embedding information, etc.); case differences alone are not enough
12. **Difference distribution pattern (core requirement)**: it is recommended to state in the specification that each entity group typically exhibits cross-table differences on only about 2-3 columns, while the remaining columns may keep the same format; try to vary the difference columns across entity groups. Avoid having all entity groups use the same fixed difference pattern on the same columns
12. **Case differences**: the specification must explicitly require case variation between different variants of the same entity (e.g. Title Case vs ALL CAPS vs all lowercase vs mixed case)
13. **Language consistency**: you must identify the language of the sampled data and explicitly require in the specification that the generated data use the same language as the sampled data (e.g. if the sampled data is in Indonesian, generate Indonesian; if in English, generate English); switching languages is not allowed

Please output Markdown text directly — no JSON, no code-block markers."""


def analyze_dataset(client, config: DatasetConfig,
                    sampled_tables: list, num_tables: int,
                    gen_config: GeneratorConfig = None) -> dict:
    """Run the LLM once and return a dict containing dataset_instructions"""
    print("Stage 2: calling the LLM to produce the generation specification...")

    columns = config.columns
    formatted_samples = format_samples_for_prompt(sampled_tables, columns)
    analysis_temp = gen_config.analysis_temperature if gen_config else 0.3

    user_prompt = ANALYSIS_USER_TEMPLATE.format(
        dataset_name=config.name,
        description=config.description,
        num_tables=num_tables,
        columns=", ".join(columns),
        formatted_samples=formatted_samples,
    )

    messages = [
        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    dataset_instructions = client.chat(
        messages,
        temperature=analysis_temp,
        force_json=False,
    ).strip()

    # occasionally the LLM wraps the markdown in ```...```; strip that too
    if dataset_instructions.startswith("```"):
        lines = dataset_instructions.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        dataset_instructions = "\n".join(lines).strip()

    analysis = {
        "dataset_instructions": dataset_instructions,
    }

    print(f"  specification text length: {len(dataset_instructions)} characters")

    return analysis


def save_analysis(analysis: dict, output_path: str):
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"  cached to: {output_path}")


def load_analysis(cache_path: str) -> dict:
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)
