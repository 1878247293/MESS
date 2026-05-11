"""
属性选择结果缓存。

属性选择跑一次比较慢，所以同样 (dataset, threshold, rate, model) 命中就直接复用。
缓存目录：cache/attribute_selection/
缓存文件：{dataset}_gamma{th}_rate{r}_{model}.json
"""

import json
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict

from log import log


@dataclass
class AttributeSelectionCache:
    dataset_name: str
    selected_attrs: List[str]
    col_sim_threshold: float
    selection_rate: float
    model_name: str
    max_seq_length: int
    timestamp: str
    selection_time: float


class AttributeSelectionCacheManager:
    def __init__(self, cache_dir: str = "cache/attribute_selection"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, dataset_name: str, col_sim_threshold: float,
                       selection_rate: float, model_name: str) -> str:
        # 不同模型相似度分布不一样，曾经共享缓存吃过亏，必须把模型名加进 key
        _model_snippet = model_name.split('/')[-1]
        return f"{dataset_name}_gamma{col_sim_threshold}_rate{selection_rate}_{_model_snippet}"

    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def has_cache(self, dataset_name: str, col_sim_threshold: float,
                  selection_rate: float, model_name: str) -> bool:
        cache_key = self._get_cache_key(dataset_name, col_sim_threshold, selection_rate, model_name)
        cache_path = self._get_cache_path(cache_key)
        return cache_path.exists()

    def load_cache(self, dataset_name: str, col_sim_threshold: float,
                   selection_rate: float, model_name: str) -> Optional[AttributeSelectionCache]:
        cache_key = self._get_cache_key(dataset_name, col_sim_threshold, selection_rate, model_name)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            cache = AttributeSelectionCache(**data)
            log(f"命中缓存: {cache_path.name}")
            log(f"  selected_attrs: {cache.selected_attrs}")
            log(f"  原始耗时: {cache.selection_time:.2f}s ({cache.selection_time/60:.1f}min)")
            log(f"  缓存时间: {cache.timestamp}")

            return cache
        except Exception as e:
            log(f"读缓存失败: {e}")
            return None

    def save_cache(self, dataset_name: str, selected_attrs: List[str],
                   col_sim_threshold: float, selection_rate: float,
                   model_name: str, max_seq_length: int,
                   selection_time: float) -> None:
        from datetime import datetime

        cache = AttributeSelectionCache(
            dataset_name=dataset_name,
            selected_attrs=selected_attrs,
            col_sim_threshold=col_sim_threshold,
            selection_rate=selection_rate,
            model_name=model_name,
            max_seq_length=max_seq_length,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            selection_time=selection_time
        )

        cache_key = self._get_cache_key(dataset_name, col_sim_threshold, selection_rate, model_name)
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(cache), f, indent=2, ensure_ascii=False)

            log(f"缓存已写入: {cache_path.name}")
            log(f"  selected_attrs: {selected_attrs}")
            log(f"  下次同参数运行可省 {selection_time/60:.1f} min")
        except Exception as e:
            log(f"写缓存失败: {e}")

    def clear_cache(self, dataset_name: Optional[str] = None) -> None:
        """传 dataset_name 就只清那个数据集；不传就全清"""
        if dataset_name:
            pattern = f"{dataset_name}_*.json"
            cache_files = list(self.cache_dir.glob(pattern))
        else:
            cache_files = list(self.cache_dir.glob("*.json"))

        for cache_file in cache_files:
            cache_file.unlink()
            log(f"删除: {cache_file.name}")

        log(f"共清掉 {len(cache_files)} 个缓存文件")

    def list_caches(self) -> List[str]:
        cache_files = sorted(self.cache_dir.glob("*.json"))
        return [f.stem for f in cache_files]


# 单例
_cache_manager = AttributeSelectionCacheManager()


def get_cache_manager() -> AttributeSelectionCacheManager:
    return _cache_manager


if __name__ == "__main__":
    # 跑一遍 save / load / list
    manager = AttributeSelectionCacheManager()

    manager.save_cache(
        dataset_name="Music-20",
        selected_attrs=["tid", "title", "artist", "album"],
        col_sim_threshold=0.9,
        selection_rate=0.2,
        model_name="all-MiniLM-L12-v2",
        max_seq_length=64,
        selection_time=35.5
    )

    cache = manager.load_cache("Music-20", 0.9, 0.2, "all-MiniLM-L12-v2")
    if cache:
        print(f"loaded: {cache.selected_attrs}")

    print("all caches:")
    for cache_name in manager.list_caches():
        print(f"  - {cache_name}")
