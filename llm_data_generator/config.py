"""数据集 schema 与生成器运行参数。"""

from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """单数据集的 schema / 文本格式 / 字段约束"""
    name: str
    columns: list  # 除 tid 之外的列名
    text_fields: list  # 拼到 triplet 文本里的字段（可以是 columns 的子集）
    text_format: str  # 文本模板，比如 "name: {name}, longtitude: {longtitude}"
    description: str  # 给 prompt 用
    # entity 组里的 canonical 字段（Person 之类有多个）
    canonical_fields: dict = field(default_factory=dict)
    # 组级额外字段：Geo 的 country、Shopee 的 category 等
    group_extra_fields: list = field(default_factory=list)
    # variant 是否带 style 字段
    has_style: bool = True
    # 每个 entity 默认要几个 variant（一般 = 表数）
    default_variants: int = 4


# 已注册的数据集
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
        "Music-200": DatasetConfig(
        name="Music-200",
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
        "Music-20": DatasetConfig(
        name="Music-20",
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
    """生成器运行参数"""
    backend: str = "api"  # ollama / vllm / api
    api_url: str = "https://www.qqcode.cc"
    model: str = "gpt-5.2"
    api_key: str = "sk-CK6upo8GOzQPJX7h2Ypa0onrd00ZvtptfHpmWyv2Cur99l1v"
    sample_size: int = 100  # Stage 1 每表采样行数
    match_sample_size: int = 0
    num_entities: int = 200  # 目标 entity group 数
    batch_size: int = 4  # 每次 LLM 调用产出多少 entity
    temperature: float = 0.7  # 生成阶段温度
    analysis_temperature: float = 0.3  # 分析阶段温度
    max_retries: int = 3
    seed: int = 42
    timeout: int = 4800  # 单次请求超时
    num_ctx: int = 65536  # 上下文窗口
    max_workers: int = 4  # 并发线程
    # 分析缓存：默认每次都重新跑；置 True 时若 analysis_cache_<model>.json 存在就直接用
    # CLI 上对应 --reuse-analysis / --skip-analysis
    reuse_analysis_cache: bool = False


def get_dataset_config(name: str) -> DatasetConfig:
    return DATASET_REGISTRY.get(name)


def auto_detect_config(name: str, columns: list) -> DatasetConfig:
    """没注册过的数据集，按列名硬塞一份默认配置出来"""
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
