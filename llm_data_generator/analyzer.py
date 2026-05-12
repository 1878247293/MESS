"""Stage 2：让 LLM 看一眼采样数据，吐出一份 markdown 形式的生成规范。

只调一次 LLM。输出会被下一阶段直接读，所以规范要把字段格式、表间差异、
噪声类型等内容写得足够细。
"""

import json

from .config import DatasetConfig, GeneratorConfig
from .sampler import format_samples_for_prompt


ANALYSIS_SYSTEM_PROMPT = """你是一位高级数据工程师和合成数据规范设计师。
你的任务是：仔细分析实体匹配数据集中各表的采样数据，生成一份极其详尽的 Markdown 格式生成规范文档。

这份文档将直接被另一个 LLM 读取并用于生成合成训练数据，因此你必须做到：
- **字段级精确**：每张表的每个字段都必须有独立的格式说明、正则模式、具体示例值
- **差异驱动**：请重点观察并描述各表之间同一字段**可能出现的**格式差异（这是实体匹配的核心挑战）。表间差异通常包括：格式编码不同、信息嵌入方式不同、字段缺失策略不同等。大小写差异只是其中一种较为次要的维度，建议不要作为唯一的表间差异
- **差异分布规律**：建议先从原始数据集中观察并归纳表间差异的实际出现规律——通常情况下，差异并不会在所有列上同时固定出现，而是倾向于在每个实体组中分散到部分列（约 2-3 列）上体现，其余列在不同表之间可以保持相同格式；不同实体组之间差异所在的列也往往有所不同，从而让整体差异分布更加多样化。请在规范中据此说明每列**可能出现的**差异类型，而不是规定每列**必须**出现差异
- **同一实体约束（最重要）**：必须在规范中反复强调——同一 entity group 的所有 variants 是同一个实体（同一首歌/同一个人/同一件商品）的不同格式表示，绝不是不同实体
- **非空约束**：每张表的规范中必须注明任何表的一个实体中最多只能有 1 个空字段
- **属性列一致性**：所有 variant 必须包含完全相同的属性列，缺失时用空字符串 ""，绝不允许某个 variant 多出或少了某个属性列
- **大小写差异**：同一实体的不同 variant 之间必须体现大小写的变化（如首字母大写 vs 全大写 vs 全小写 vs 混合大小写）
- **语言一致性**：必须识别采样数据中使用的语言（如英文、印尼语、中文等），并在规范中明确要求生成数据使用与采样数据相同的语言，不允许自行切换语言
- **覆盖噪声**：明确列出数据中存在的所有噪声类型及其出现位置
- **示例丰富**：每个格式说明都要有 2-3 个真实风格的示例值
- **绝对不能模糊**：不允许出现"格式多样""视情况而定"等模糊表述，一切都要具体

直接输出 Markdown 文本，不要 JSON，不要代码块标记。"""


ANALYSIS_USER_TEMPLATE = """请分析以下实体匹配数据集的各表采样数据，然后生成一份字段级别的完整生成规范。

## 数据集信息
- 名称: {dataset_name}
- 描述: {description}
- 表数量: {num_tables}
- 数据列: {columns}

## 各表采样记录
{formatted_samples}

---

## 请按照以下结构输出 Markdown 规范文档

请参考下方的 **Hotel-300 示范**，为当前数据集 **{dataset_name}** 生成同等详细度的规范。
你的输出必须包含以下所有章节：

### 必须包含的章节

1. **`## {{dataset_name}} 数据集特定要求`** — 数据集级别的总体要求（允许虚构实体但必须符合现实风格和常识、variant 含义等）
2. **`### 各表字段格式规范`** — 对每张表单独一个小节 `**Table X** (简要特征描述)`，在每个小节内对**每个字段**逐一给出：
   - 格式模式（如 "前缀 + N位数字"、"MM:SS"、"小数分钟"）
   - 2-3 个具体示例值
   - 缺失情况（该字段在此表中是否常为空字符串）
   - 与其他表同字段的关键差异
   - **非空约束**：每张表的规范中必须注明任何表的一个实体中都最多只能有 1 个或者 0 个非空字段
3. **`### 表间差异模式`** — 列出所有系统性的跨表格式差异（如大小写、缩写、编码、缺失策略等），每条差异说明涉及哪些表和字段
4. **`### 噪声类型`** — 列出数据中存在的所有噪声（拼写错误、截断、输入错误等），每种噪声给出具体的出现位置和示例
5. **`### 特别注意`** — **第一条必须是**："同一 entity group 的所有 variants 必须是同一个实体的不同格式表示（例如同一首歌、同一个人、同一件商品），绝不允许不同实体混入同一组"。之后再列出：variant 之间必须保持的对应关系、容易犯的错误、生成时的关键约束

---

=== 示范：Hotel-300 数据集特定生成规范 ===

## Hotel-300 数据集特定要求
- 允许生成虚构的酒店记录，但名称、地点等必须符合现实世界的风格和常识，不允许出现明显不合理的内容（如"月球酒店""量子空间旅馆"）
- 每个实体组内的 4 个 variant 必须是同一家酒店的不同表示方式（相同酒店名的各种格式变体）
- **不同表的 variant 之间必须存在实质性的格式差异**（拼写变体、缩写、截断、OCR错误、格式编码不同、信息嵌入方式不同等），而不是仅有大小写不同
- **同一实体的不同 variant 之间也必须存在大小写差异**
- **所有 variant 必须包含完全相同的 5 个属性列**（name, city, stars, price, country），缺失时用空字符串 `""`

### 各表字段格式规范

**Table 0** (最完整的标准表，首字母大写风格)
- `name`: 格式为首字母大写的酒店全名，如 `"Grand Pacific Hotel"`、`"Sunset Bay Resort"`、`"Alpine Lodge & Spa"`
  - 缺失情况：始终非空
  - 与其他表差异：Table 1 全大写含城市嵌入，Table 2 全小写含拼写错误，Table 3 混合大小写含营销词
- `city`: 完整城市名，首字母大写，如 `"San Francisco"`、`"New York"`、`"Buenos Aires"`
  - 缺失情况：始终非空
  - 与其他表差异：Table 1 嵌入 name 括号中本字段为空，Table 2 为城市缩写如 `"SF"`，Table 3 为完整城市名全小写
- `stars`: 整数星级字符串，如 `"5"`、`"3"`、`"4"`
  - 缺失情况：始终非空
  - 与其他表差异：Table 1 为英文文本如 `"Five Star"`，Table 2 为符号如 `"***"`，Table 3 为空字符串
- `price`: 美元价格带 $ 前缀，如 `"$350"`、`"$120"`、`"$89"`
  - 缺失情况：始终非空
  - 与其他表差异：Table 1 为欧元价格如 `"EUR 320"`，Table 2 为纯数字如 `"350"`，Table 3 为空字符串
- `country`: 完整国名，首字母大写，如 `"United States"`、`"France"`、`"Argentina"`
  - 缺失情况：始终非空
  - 与其他表差异：Table 1 为 2 字母国家代码如 `"US"`，Table 2 为 3 字母代码如 `"USA"`，Table 3 为空字符串
- **非空约束**：本表所有 5 个字段均非空
- **属性列**：name, city, stars, price, country（与其他表完全一致）

**Table 1** (全大写名称，城市嵌入 name，格式变体多)
- `name`: 全大写 + 括号嵌入城市名，如 `"GRAND PACIFIC HOTEL (SAN FRANCISCO)"`、`"SUNSET BAY RESORT (NEW YORK)"`、`"ALPINE LODGE & SPA (BUENOS AIRES)"`
  - 缺失情况：始终非空
  - 与其他表差异：Table 0 首字母大写无城市嵌入，Table 2 全小写，Table 3 混合大小写
- `city`: 空字符串 `""`（城市信息已嵌入 name 括号中）
  - 缺失情况：偶尔为空字符串（信息转移至 name）
  - 与其他表差异：Table 0 完整城市名，Table 2 缩写，Table 3 全小写城市名
- `stars`: 英文文本星级，如 `"Five Star"`、`"Three Star"`、`"Four Star"`
  - 缺失情况：始终非空
  - 与其他表差异：Table 0 为数字如 `"5"`，Table 2 为符号如 `"***"`，Table 3 为空
- `price`: 欧元价格格式，如 `"EUR 320"`、`"EUR 110"`、`"EUR 80"`
  - 缺失情况：始终非空
  - 与其他表差异：Table 0 美元带 $ 如 `"$350"`，Table 2 纯数字，Table 3 为空
- `country`: 2 字母 ISO 国家代码，如 `"US"`、`"FR"`、`"AR"`
  - 缺失情况：始终非空
  - 与其他表差异：Table 0 完整国名，Table 2 三字母代码，Table 3 为空
- **非空约束**：本表最多 1 个字段为空（仅 city 为空，其余均有值）
- **属性列**：name, city, stars, price, country（与其他表完全一致）

**Table 2** (全小写，缩写，含拼写错误)
- `name`: 全小写，可能含拼写错误（漏字母/字母互换），如 `"grand pacfic hotel"`、`"sunet bay resort"`、`"alpien lodge & spa"`
  - 缺失情况：始终非空
  - 与其他表差异：Table 0 首字母大写无错误，Table 1 全大写，Table 3 混合大小写
- `city`: 城市缩写（全大写），如 `"SF"`、`"NY"`、`"BA"`
  - 缺失情况：始终非空
  - 与其他表差异：Table 0 完整城市名，Table 1 嵌入 name 本字段为空，Table 3 全小写城市名
- `stars`: 符号表示星级，如 `"***"`、`"*****"`、`"****"`
  - 缺失情况：始终非空
  - 与其他表差异：Table 0 数字如 `"5"`，Table 1 英文如 `"Five Star"`，Table 3 为空
- `price`: 纯数字字符串（无货币前缀），如 `"350"`、`"120"`、`"89"`
  - 缺失情况：始终非空
  - 与其他表差异：Table 0 带 $ 前缀，Table 1 欧元格式，Table 3 为空
- `country`: 3 字母 ISO 国家代码，如 `"USA"`、`"FRA"`、`"ARG"`
  - 缺失情况：始终非空
  - 与其他表差异：Table 0 完整国名，Table 1 两字母代码，Table 3 为空
- **非空约束**：本表所有 5 个字段均非空
- **属性列**：name, city, stars, price, country（与其他表完全一致）

**Table 3** (混合大小写，营销词，仅保留 name 和 city)
- `name`: 大小写混合 + 营销词/符号后缀，如 `"grand PACIFIC Hotel ★★★★★"`、`"SUNSET bay Resort - Best Deal"`、`"Alpine LODGE & spa【热销】"`
  - 缺失情况：始终非空
  - 与其他表差异：Table 0 首字母大写，Table 1 全大写，Table 2 全小写
- `city`: 全小写的完整城市名，如 `"san francisco"`、`"new york"`、`"buenos aires"`
  - 缺失情况：始终非空
  - 与其他表差异：Table 0 首字母大写，Table 1 为空（嵌入 name），Table 2 为缩写
- `stars`: 空字符串 `""`
  - 缺失情况：偶尔为空字符串
  - 与其他表差异：Table 0 数字，Table 1 英文文本，Table 2 符号
- `price`: 空字符串 `""`
  - 缺失情况：偶尔为空字符串
  - 与其他表差异：Table 0 美元，Table 1 欧元，Table 2 纯数字
- `country`: 空字符串 `""`
  - 缺失情况：偶尔为空字符串
  - 与其他表差异：Table 0 完整国名，Table 1 两字母代码，Table 2 三字母代码
- **非空约束**：本表 name 和 city 始终非空，stars/price/country 偶尔为空
- **属性列**：name, city, stars, price, country（与其他表完全一致）

### 表间差异模式（差异分布规律）

**总体思路**：以下差异维度是各列**可能出现的**差异类型，可作为参考。建议先从原始数据集中观察并归纳差异的实际分布规律——通常每个实体组只在约 2-3 个列上体现差异，其余列在不同表之间可以保持相同格式；不同实体组差异出现的列也往往有所不同。

- **大小写差异（可出现在 name/city/country 列）**：同一内容在不同表中使用不同大小写风格，如首字母大写 `"Grand Pacific Hotel"` vs 全大写 `"GRAND PACIFIC HOTEL"` vs 全小写 `"grand pacific hotel"` vs 混合大小写 `"grand PACIFIC Hotel"`
- **城市格式（可出现在 city 列）**：完整城市名 `"San Francisco"` vs 嵌入 name 括号 vs 缩写 `"SF"` vs 全小写 `"san francisco"`
- **星级格式（可出现在 stars 列）**：数字 `"5"` vs 英文 `"Five Star"` vs 符号 `"*****"` vs 为空
- **价格格式（可出现在 price 列）**：美元 `"$350"` vs 欧元 `"EUR 320"` vs 纯数字 `"350"` vs 为空
- **国家格式（可出现在 country 列）**：完整国名 `"United States"` vs 两字母 `"US"` vs 三字母 `"USA"` vs 为空
- **拼写错误（可随机出现在任意文本列）**：漏字母、字母互换等
- **字段缺失（随机分布）**：每个实体组的不同表之间，可随机有 0-1 个字段为空，但不是固定某张表的某列总是空
- **属性列一致性**：所有 4 张表必须包含完全相同的 5 个属性列（name, city, stars, price, country），缺失字段用空字符串 `""` 填充

**示例**：
- 实体 A：name 列体现大小写差异 + city 列体现缩写差异，其余列格式相同
- 实体 B：price 列体现格式差异 + name 列含拼写错误，其余列格式相同
- 实体 C：country 列体现缩写差异 + stars 列体现格式差异，其余列格式相同

### 噪声类型
- **大小写变化**：同一名称在不同表中分别为首字母大写 / 全大写 / 全小写 / 混合大小写，如 `"Grand Pacific"` vs `"GRAND PACIFIC"` vs `"grand pacific"` vs `"grand PACIFIC"`
- **拼写错误**：出现在 Table 2 的 name 中，漏字母（`"Pacific"` → `"Pacfic"`）、字母互换（`"Sunset"` → `"Sunet"`）、多余空格
- **缩写**：城市名 `"San Francisco"` → `"SF"`（Table 2），国家名 `"United States"` → `"US"`（Table 1）
- **营销词/符号插入**：出现在 Table 3 的 name 中，如 `"★★★★★"`、`"Best Deal"`、`"【热销】"` 等额外文本
- **格式差异**：价格 `"$350"` vs `"350"`，同一信息不同表示方式

### 特别注意
- **【最重要】同一个 entity group 的各 variant 必须是同一家酒店的不同格式表示，绝不允许将不同酒店放入同一组**
- **属性列一致性**：所有 variant 必须包含完全相同的属性列（name, city, stars, price, country），缺失时用空字符串 `""`，绝不允许某个 variant 多出或少了某个属性列
- **表间格式差异（核心维度）**：不同表的 variant 之间必须存在实质性的内容差异（拼写变体、缩写、截断、OCR错误、格式编码不同、信息嵌入方式不同等），仅有大小写不同是不够的
- **差异分布规律（核心维度）**：通常每个实体组只在约 2-3 个列上体现表间差异，其余列可以保持格式相同；不同实体组的差异列尽量有所变化，使整体差异分布更加多样化。请避免所有实体组在相同列上使用相同的固定差异模式
- **大小写差异**：同一实体的不同 variant 之间必须体现大小写的变化（如首字母大写 vs 全大写 vs 全小写 vs 混合大小写）
- **非空约束**：每张表的规范中必须注明任何表的一个实体中最多只能有 1 个空字段
- **噪声覆盖**：必须包含拼写错误（漏字母/字母互换）、缩写、营销词插入、格式差异等多种噪声类型
- **语言一致性**：生成数据必须使用与采样数据相同的语言，不允许自行切换语言（如采样数据是印尼语则生成印尼语，是英文则生成英文）

现在请为 **{dataset_name}** 数据集生成同等详细度和完整度的规范。要求：
1. 为每张表单独列一个小节 **Table X** (简要特征描述)
2. 在每个小节内，对**每个字段**逐一给出具体格式要求（包括格式模式、2-3个示例值、缺失情况、与其他表的差异）
3. 必须包含 **表间差异模式** 章节，列出所有可能的系统性跨表差异。**建议先从原始数据集中观察并归纳差异的实际分布规律**——通常差异并非集中在固定列上，而是分散到多个列上；每个实体组一般在约 2-3 个列上体现差异，其余列在不同表之间可以保持相同格式。规范中应描述每列**可能出现的**差异类型，而不是规定每列**必须**出现差异
4. 必须包含 **噪声类型** 章节，列出所有噪声及其位置和示例
5. 必须包含 **特别注意** 章节，列出生成约束
6. 语气要确切、具体，每个字段都要有具体示例值，不要说模糊的话
7. 从采样数据中提炼出尽可能多的细节，不要遗漏任何可观察到的模式
8. **"特别注意"章节的第一条必须是**：同一 entity group 的所有 variants 必须是同一个实体的不同格式表示（不同的格式/噪声/缺失变体），绝不允许不同实体混入同一组
9. **非空约束**：每张表的规范中必须注明任何表的一个实体中最多只能有 1 个空字段
10. **属性列一致性**：必须在规范中明确要求所有 variant 包含完全相同的属性列，缺失时用空字符串 ""，绝不允许某个 variant 多出或少了某个属性列
11. **表间格式差异（核心要求）**：必须在规范中明确要求不同表的 variant 之间存在实质性的内容差异（拼写变体、缩写、截断、OCR错误、格式编码不同、信息嵌入方式不同等），仅有大小写不同是不够的
12. **差异分布规律（核心要求）**：建议在规范中说明每个实体组通常只在约 2-3 个列上体现表间差异，其余列可以保持格式相同；不同实体组差异所在的列尽量有所不同。请避免所有实体组在同样的列上使用同样的固定差异模式
12. **大小写差异**：必须在规范中明确要求同一实体的不同 variant 之间体现大小写的变化（如首字母大写 vs 全大写 vs 全小写 vs 混合大小写）
13. **语言一致性**：必须识别采样数据的语言，并在规范中明确要求生成数据使用与采样数据相同的语言（如采样数据是印尼语则生成印尼语，是英文则生成英文），不允许自行切换语言

请直接输出 Markdown 文本，不要 JSON，不要代码块标记。"""


def analyze_dataset(client, config: DatasetConfig,
                    sampled_tables: list, num_tables: int,
                    gen_config: GeneratorConfig = None) -> dict:
    """跑一次 LLM，返回包含 dataset_instructions 的 dict"""
    print("Stage 2: 调 LLM 出生成规范...")

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

    # 偶尔 LLM 会把 markdown 包在 ```...``` 里，一并去掉
    if dataset_instructions.startswith("```"):
        lines = dataset_instructions.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        dataset_instructions = "\n".join(lines).strip()

    analysis = {
        "dataset_instructions": dataset_instructions,
    }

    print(f"  规范文本长度: {len(dataset_instructions)} 字符")

    return analysis


def save_analysis(analysis: dict, output_path: str):
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"  缓存到: {output_path}")


def load_analysis(cache_path: str) -> dict:
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)
