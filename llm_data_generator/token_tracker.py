"""LLM token 用量计数器，线程安全（ThreadPoolExecutor 里也能用）。"""

import json
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path


class TokenTracker:
    """按阶段累加 prompt / completion token，加锁所以并发安全"""

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
        """记一次调用。total_tokens 没传就两边相加。stage 可以是 analysis / generation 之类"""
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
        with self._lock:
            s = self._stages[stage]
            s["calls"] += 1
            s["prompt_tokens"] += prompt_tokens
            s["completion_tokens"] += completion_tokens
            s["total_tokens"] += total_tokens

    # 汇总查询

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

    # 输出

    def report(self, input_price: float = 0.0, output_price: float = 0.0):
        """打印用量。
        input_price / output_price 给单位 $/M tokens；为 0 就不估成本。
        """
        print(f"\n{'=' * 50}")
        print("LLM Token 用量")
        print(f"{'=' * 50}")
        if self.model:
            print(f"  模型: {self.model}")

        with self._lock:
            stages = dict(self._stages)

        for stage_name, s in stages.items():
            print(f"\n  [{stage_name}]")
            print(f"    calls:         {s['calls']}")
            print(f"    prompt tokens: {s['prompt_tokens']:,}")
            print(f"    output tokens: {s['completion_tokens']:,}")
            print(f"    total tokens:  {s['total_tokens']:,}")

        print(f"\n  [总计]")
        print(f"    calls:        {self.total_calls}")
        print(f"    prompt:       {self.total_prompt_tokens:,}")
        print(f"    output:       {self.total_completion_tokens:,}")
        print(f"    total:        {self.total_tokens:,}")

        if input_price > 0 or output_price > 0:
            cost_input = self.total_prompt_tokens / 1_000_000 * input_price
            cost_output = self.total_completion_tokens / 1_000_000 * output_price
            cost_total = cost_input + cost_output
            print(f"\n  [成本估算] (input=${input_price}/M, output=${output_price}/M)")
            print(f"    input cost:  ${cost_input:.4f}")
            print(f"    output cost: ${cost_output:.4f}")
            print(f"    total cost:  ${cost_total:.4f}")

        print(f"{'=' * 50}\n")

    def to_dict(self) -> dict:
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
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"  token 统计 -> {output_path}")
