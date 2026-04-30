"""数据集配置和生成器参数。"""

from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """单个数据集的 schema 和格式描述。"""
    name: str
    columns: list  # CSV 中除 tid 外的列名
    text_fields: list  # 用于 triplet 文本的字段（可能是 columns 的子集）
    text_format: str  # triplet 文本模板，如 "name: {name}, longtitude: {longtitude}"
    description: str  # 数据集描述，用于 LLM prompt
    # entity group 中 canonical 字段名（可以有多个，如 Person 有 canonical_givenname 等）
    canonical_fields: dict = field(default_factory=dict)
    # entity group 中的额外分类字段（如 Geo 的 country, Shopee 的 category）
    group_extra_fields: list = field(default_factory=list)
    # variant 中是否包含 style 字段
    has_style: bool = True
    # 默认每个 entity 的 variant 数量（等于表数量）
    default_variants: int = 4


# 预定义数据集注册表
DATASET_REGISTRY = {
    "Geo": DatasetConfig(
        name="Geo",
        columns=["name", "longtitude", "latitude"],
        text_fields=["name", "longtitude", "latitude"],
        text_format="name: {name}, longtitude: {longtitude}, latitude: {latitude}",
        description="地理位置实体匹配：城市/地点名称，含经纬度。",
        canonical_fields={
            "canonical_name": "name",
            "canonical_longtitude": "longtitude",
            "canonical_latitude": "latitude",
        },
        group_extra_fields=["country"],
        has_style=True,
        default_variants=4,
    ),
    "Person": DatasetConfig(
        name="Person",
        columns=["recid", "givenname", "surname", "suburb", "postcode"],
        text_fields=["recid", "givenname", "surname", "suburb", "postcode"],
        text_format="recid: {recid}, givenname: {givenname}, surname: {surname}, suburb: {suburb}, postcode: {postcode}",
        description="人名实体匹配：姓名+地址记录。",
        canonical_fields={
            "canonical_recid": "recid",
            "canonical_givenname": "givenname",
            "canonical_surname": "surname",
            "canonical_suburb": "suburb",
            "canonical_postcode": "postcode",
        },
        group_extra_fields=[],
        has_style=True,
        default_variants=5,
    ),
    "Shopee": DatasetConfig(
        name="Shopee",
        columns=["title"],
        text_fields=["title"],
        text_format="title: {title}",
        description="电商产品标题匹配：印尼语Shopee商品标题。",
        canonical_fields={"canonical_title": "title"},
        group_extra_fields=["category"],
        has_style=False,
        default_variants=8,
    ),
    "Music-2000": DatasetConfig(
        name="Music-2000",
        columns=["id", "number", "title", "length", "artist", "album", "year", "language"],
        text_fields=["id", "number", "title", "length", "artist", "album", "year", "language"],
        text_format="id: {id}, number: {number}, title: {title}, length: {length}, artist: {artist}, album: {album}, year: {year}, language: {language}",
        description="音乐实体匹配：歌曲记录来自不同数据库。",
        canonical_fields={
            "canonical_id": "id",
            "canonical_number": "number",
            "canonical_title": "title",
            "canonical_length": "length",
            "canonical_artist": "artist",
            "canonical_album": "album",
            "canonical_year": "year",
            "canonical_language": "language",
        },
        group_extra_fields=[],
        has_style=True,
        default_variants=5,
    ),
}


@dataclass
class GeneratorConfig:
    """生成器运行参数（唯一参数来源）。"""
    backend: str = "api"  # LLM 后端: "ollama", "vllm" 或 "api"
    api_url: str = "https://www.qqcode.cc"  # LLM API 地址
    model: str = "gpt-5.2"
    api_key: str = "sk-CK6upo8GOzQPJX7h2Ypa0onrd00ZvtptfHpmWyv2Cur99l1v"
    sample_size: int = 100  # 每表采样记录数（用于分析）
    match_sample_size: int = 0  # 采样匹配组数
    num_entities: int = 200  # 目标生成 entity group 数
    batch_size: int = 4  # 每次 LLM 调用生成的 entity 数
    temperature: float = 0.7  # 生成阶段温度
    analysis_temperature: float = 0.3  # 分析阶段温度
    max_retries: int = 3
    seed: int = 42
    timeout: int = 4800  # LLM 请求超时秒数
    num_ctx: int = 65536  # 上下文窗口大小
    max_workers: int = 4  # 并发生成线程数
    # ── 分析阶段缓存复用开关 ──────────────────────────────────────────────
    # False (默认): 每次都重新调用 LLM 分析数据并生成新提示词
    # True: 若 analysis_cache_<model>.json 存在则直接复用,跳过 Stage 2
    # CLI 中等价于 --reuse-analysis / --skip-analysis
    reuse_analysis_cache: bool = False


def get_dataset_config(name: str) -> DatasetConfig:
    """获取数据集配置，如果不在注册表中则返回 None。"""
    return DATASET_REGISTRY.get(name)


def auto_detect_config(name: str, columns: list) -> DatasetConfig:
    """从 CSV 列名自动生成配置（用于未注册数据集）。"""
    text_fields = [c for c in columns if c != "tid"]
    text_format = ", ".join(f"{c}: {{{c}}}" for c in text_fields)
    return DatasetConfig(
        name=name,
        columns=columns,
        text_fields=text_fields,
        text_format=text_format,
        description=f"实体匹配数据集 {name}",
        canonical_fields={"canonical": text_fields[0]} if text_fields else {},
        group_extra_fields=[],
        has_style=True,
        default_variants=4,
    )
