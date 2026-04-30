"""Token 用量统计器 — 线程安全地累计 LLM 调用的 token 消耗。"""

import json
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path


class TokenTracker:
    """累计 LLM API 调用的 token 用量，支持按阶段分组统计。

    线程安全：内部使用 threading.Lock 保护所有写操作，
    可在 ThreadPoolExecutor 并发生成中安全使用。
    """

    def __init__(self, model: str = ""):
        self.model = model
        self._lock = threading.Lock()
        self._stages: dict[str, dict] = defaultdict(lambda: {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        })

    def record(self, prompt_tokens: int, completion_tokens: int,
               total_tokens: int = 0, stage: str = "default"):
        """记录一次 LLM 调用的 token 用量。

        Args:
            prompt_tokens: 输入 token 数
            completion_tokens: 输出 token 数
            total_tokens: 总 token 数（若为 0 则自动求和）
            stage: 阶段标签，如 "analysis" / "generation"
        """
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
        with self._lock:
            s = self._stages[stage]
            s["calls"] += 1
            s["prompt_tokens"] += prompt_tokens
            s["completion_tokens"] += completion_tokens
            s["total_tokens"] += total_tokens

    # ------ 汇总查询 ------

    @property
    def total_calls(self) -> int:
        with self._lock:
            return sum(s["calls"] for s in self._stages.values())

    @property
    def total_prompt_tokens(self) -> int:
        with self._lock:
            return sum(s["prompt_tokens"] for s in self._stages.values())

    @property
    def total_completion_tokens(self) -> int:
        with self._lock:
            return sum(s["completion_tokens"] for s in self._stages.values())

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return sum(s["total_tokens"] for s in self._stages.values())

    # ------ 输出 ------

    def report(self, input_price: float = 0.0, output_price: float = 0.0):
        """打印汇总报告。

        Args:
            input_price: 输入 token 价格（$/M tokens），0 表示不估算成本
            output_price: 输出 token 价格（$/M tokens），0 表示不估算成本
        """
        print(f"\n{'=' * 50}")
        print("LLM Token 用量统计")
        print(f"{'=' * 50}")
        if self.model:
            print(f"  模型: {self.model}")

        with self._lock:
            stages = dict(self._stages)

        for stage_name, s in stages.items():
            print(f"\n  [{stage_name}]")
            print(f"    调用次数:      {s['calls']}")
            print(f"    Prompt tokens: {s['prompt_tokens']:,}")
            print(f"    Output tokens: {s['completion_tokens']:,}")
            print(f"    Total tokens:  {s['total_tokens']:,}")

        print(f"\n  [汇总]")
        print(f"    总调用次数:    {self.total_calls}")
        print(f"    总 Prompt:     {self.total_prompt_tokens:,}")
        print(f"    总 Output:     {self.total_completion_tokens:,}")
        print(f"    总 Tokens:     {self.total_tokens:,}")

        if input_price > 0 or output_price > 0:
            cost_input = self.total_prompt_tokens / 1_000_000 * input_price
            cost_output = self.total_completion_tokens / 1_000_000 * output_price
            cost_total = cost_input + cost_output
            print(f"\n  [成本估算] (input=${input_price}/M, output=${output_price}/M)")
            print(f"    Input 成本:  ${cost_input:.4f}")
            print(f"    Output 成本: ${cost_output:.4f}")
            print(f"    总成本:      ${cost_total:.4f}")

        print(f"{'=' * 50}\n")

    def to_dict(self) -> dict:
        """导出为 dict（用于 JSON 序列化）。"""
        with self._lock:
            stages = {k: dict(v) for k, v in self._stages.items()}
        return {
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "stages": stages,
            "summary": {
                "total_calls": self.total_calls,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens,
            },
        }

    def save(self, output_path: str):
        """保存统计结果到 JSON 文件。"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"  Token 统计已保存到: {output_path}")
