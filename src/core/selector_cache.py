"""
属性选择结果缓存模块

功能:
1. 保存每个数据集的属性选择结果
2. 加载已缓存的属性选择结果
3. 避免重复执行耗时的属性选择过程

缓存文件格式: JSON
缓存目录: cache/attribute_selection/
缓存文件名: {dataset_name}_{col_sim_threshold}_{selection_rate}.json
"""

import json
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict

from log import log


@dataclass
class AttributeSelectionCache:
    """属性选择缓存数据结构"""
    dataset_name: str
    selected_attrs: List[str]
    col_sim_threshold: float
    selection_rate: float
    model_name: str
    max_seq_length: int
    # 元数据
    timestamp: str  # 缓存创建时间
    selection_time: float  # 属性选择耗时(秒)


class AttributeSelectionCacheManager:
    """属性选择缓存管理器"""

    def __init__(self, cache_dir: str = "cache/attribute_selection"):
        """
        初始化缓存管理器

        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, dataset_name: str, col_sim_threshold: float,
                       selection_rate: float, model_name: str) -> str:
        """
        生成缓存文件名

        Args:
            dataset_name: 数据集名称
            col_sim_threshold: 列相似度阈值
            selection_rate: 采样率
            model_name: 模型名称

        Returns:
            缓存文件名(不含扩展名)
        """
        # 移除了由于不同模型相似度分布不同导致的共享缓存Bug
        _model_snippet = model_name.split('/')[-1] # 只取模型目录名 tail
        return f"{dataset_name}_gamma{col_sim_threshold}_rate{selection_rate}_{_model_snippet}"

    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件的完整路径"""
        return self.cache_dir / f"{cache_key}.json"

    def has_cache(self, dataset_name: str, col_sim_threshold: float,
                  selection_rate: float, model_name: str) -> bool:
        """
        检查是否存在缓存

        Args:
            dataset_name: 数据集名称
            col_sim_threshold: 列相似度阈值
            selection_rate: 采样率
            model_name: 模型名称

        Returns:
            True if cache exists, False otherwise
        """
        cache_key = self._get_cache_key(dataset_name, col_sim_threshold, selection_rate, model_name)
        cache_path = self._get_cache_path(cache_key)
        return cache_path.exists()

    def load_cache(self, dataset_name: str, col_sim_threshold: float,
                   selection_rate: float, model_name: str) -> Optional[AttributeSelectionCache]:
        """
        加载缓存的属性选择结果

        Args:
            dataset_name: 数据集名称
            col_sim_threshold: 列相似度阈值
            selection_rate: 采样率
            model_name: 模型名称

        Returns:
            AttributeSelectionCache 对象,如果不存在则返回 None
        """
        cache_key = self._get_cache_key(dataset_name, col_sim_threshold, selection_rate, model_name)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            cache = AttributeSelectionCache(**data)
            log(f"✓ 从缓存加载属性选择结果: {cache_path.name}")
            log(f"  选中的属性: {cache.selected_attrs}")
            log(f"  原始耗时: {cache.selection_time:.2f}秒 ({cache.selection_time/60:.1f}分钟)")
            log(f"  缓存时间: {cache.timestamp}")

            return cache
        except Exception as e:
            log(f"⚠ 加载缓存失败: {e}")
            return None

    def save_cache(self, dataset_name: str, selected_attrs: List[str],
                   col_sim_threshold: float, selection_rate: float,
                   model_name: str, max_seq_length: int,
                   selection_time: float) -> None:
        """
        保存属性选择结果到缓存

        Args:
            dataset_name: 数据集名称
            selected_attrs: 选中的属性列表
            col_sim_threshold: 列相似度阈值
            selection_rate: 采样率
            model_name: 模型名称
            max_seq_length: 最大序列长度
            selection_time: 属性选择耗时(秒)
        """
        from datetime import datetime

        # 创建缓存对象
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

        # 保存到文件
        cache_key = self._get_cache_key(dataset_name, col_sim_threshold, selection_rate, model_name)
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(cache), f, indent=2, ensure_ascii=False)

            log(f"✓ 属性选择结果已缓存: {cache_path.name}")
            log(f"  选中的属性: {selected_attrs}")
            log(f"  下次运行将直接加载缓存,节省 {selection_time/60:.1f} 分钟")
        except Exception as e:
            log(f"⚠ 保存缓存失败: {e}")

    def clear_cache(self, dataset_name: Optional[str] = None) -> None:
        """
        清除缓存

        Args:
            dataset_name: 如果指定,只清除该数据集的缓存;否则清除所有缓存
        """
        if dataset_name:
            # 清除特定数据集的所有缓存
            pattern = f"{dataset_name}_*.json"
            cache_files = list(self.cache_dir.glob(pattern))
        else:
            # 清除所有缓存
            cache_files = list(self.cache_dir.glob("*.json"))

        for cache_file in cache_files:
            cache_file.unlink()
            log(f"已删除缓存: {cache_file.name}")

        log(f"共清除 {len(cache_files)} 个缓存文件")

    def list_caches(self) -> List[str]:
        """
        列出所有缓存

        Returns:
            缓存文件名列表
        """
        cache_files = sorted(self.cache_dir.glob("*.json"))
        return [f.stem for f in cache_files]


# 全局缓存管理器实例
_cache_manager = AttributeSelectionCacheManager()


def get_cache_manager() -> AttributeSelectionCacheManager:
    """获取全局缓存管理器实例"""
    return _cache_manager


if __name__ == "__main__":
    # 测试代码
    manager = AttributeSelectionCacheManager()

    # 测试保存
    manager.save_cache(
        dataset_name="Music-20",
        selected_attrs=["tid", "title", "artist", "album"],
        col_sim_threshold=0.9,
        selection_rate=0.2,
        model_name="all-MiniLM-L12-v2",
        max_seq_length=64,
        selection_time=35.5
    )

    # 测试加载
    cache = manager.load_cache("Music-20", 0.9, 0.2, "all-MiniLM-L12-v2")
    if cache:
        print(f"加载成功: {cache.selected_attrs}")

    # 列出所有缓存
    print("所有缓存:")
    for cache_name in manager.list_caches():
        print(f"  - {cache_name}")
