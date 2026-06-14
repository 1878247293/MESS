"""Dataset schema and generator runtime parameters."""

from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """Schema / text format / field constraints for a single dataset"""
    name: str
    columns: list  # column names other than tid
    text_fields: list  # fields concatenated into the triplet text (may be a subset of columns)
    text_format: str  # text template, e.g. "name: {name}, longtitude: {longtitude}"
    description: str  # used in the prompt
    # canonical fields of the entity group (Person etc. have several)
    canonical_fields: dict = field(default_factory=dict)
    # group-level extra fields: country for Geo, category for Shopee, etc.
    group_extra_fields: list = field(default_factory=list)
    # whether the variant carries a style field
    has_style: bool = True
    # default number of variants per entity (usually = number of tables)
    default_variants: int = 4


# registered datasets
DATASET_REGISTRY = {
    "Geo": DatasetConfig(
        name="Geo",
        columns=["name", "longtitude", "latitude"],
        text_fields=["name", "longtitude", "latitude"],
        text_format="name: {name}, longtitude: {longtitude}, latitude: {latitude}",
        description="Geographic location entity matching: city/place names with longitude and latitude.",
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
        description="Person-name entity matching: name + address records.",
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
        description="E-commerce product title matching: Indonesian Shopee product titles.",
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
        description="Music entity matching: song records from different databases.",
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
        description="Music entity matching: song records from different databases.",
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
        description="Music entity matching: song records from different databases.",
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
    """Generator runtime parameters"""
    backend: str = "api"  # ollama / vllm / api
    api_url: str = "https://www.qqcode.cc"
    model: str = "gpt-5.2"
    api_key: str = "sk-CK6upo8GOzQPJX7h2Ypa0onrd00ZvtptfHpmWyv2Cur99l1v"
    sample_size: int = 100  # number of rows sampled per table in Stage 1
    match_sample_size: int = 0
    num_entities: int = 200  # target number of entity groups
    batch_size: int = 4  # how many entities each LLM call produces
    temperature: float = 0.7  # temperature for the generation stage
    analysis_temperature: float = 0.3  # temperature for the analysis stage
    max_retries: int = 3
    seed: int = 42
    timeout: int = 4800  # per-request timeout
    num_ctx: int = 65536  # context window
    max_workers: int = 4  # number of concurrent threads
    # analysis cache: re-run every time by default; when True, reuse analysis_cache_<model>.json if it exists
    # corresponds to --reuse-analysis / --skip-analysis on the CLI
    reuse_analysis_cache: bool = False


def get_dataset_config(name: str) -> DatasetConfig:
    return DATASET_REGISTRY.get(name)


def auto_detect_config(name: str, columns: list) -> DatasetConfig:
    """For unregistered datasets, build a default config from the column names"""
    text_fields = [c for c in columns if c != "tid"]
    text_format = ", ".join(f"{c}: {{{c}}}" for c in text_fields)
    return DatasetConfig(
        name=name,
        columns=columns,
        text_fields=text_fields,
        text_format=text_format,
        description=f"Entity matching dataset {name}",
        canonical_fields={"canonical": text_fields[0]} if text_fields else {},
        group_extra_fields=[],
        has_style=True,
        default_variants=4,
    )
