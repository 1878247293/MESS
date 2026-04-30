"""Ollama REST API 客户端，支持 JSON 解析容错和重试。"""

import json
import re
import time
import requests


class OllamaClient:
    def __init__(self, base_url: str, model: str, temperature: float,
                 max_retries: int, timeout: int, num_ctx: int,
                 token_tracker=None):
        """初始化 Ollama 客户端。

        Args:
            base_url: Ollama API 地址
            model: 模型名称
            temperature: 生成温度
            max_retries: 最大重试次数
            timeout: 请求超时时间（秒）
            num_ctx: 上下文窗口大小（token 数）
            token_tracker: TokenTracker 实例（可选）
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.token_tracker = token_tracker
        self._current_stage = "default"

    def chat(self, messages: list, temperature: float = None,
             force_json: bool = True) -> str:
        """发送聊天请求，返回助手回复文本。

        Args:
            messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
            temperature: 覆盖默认温度
            force_json: 若为 True，启用 Ollama JSON mode（强制 JSON 输出）；
                        若为 False，返回普通文本（用于生成 Markdown 等自由格式内容）

        Returns:
            助手回复的文本内容
        """
        temp = temperature if temperature is not None else self.temperature
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_ctx": self.num_ctx,
            },
        }
        if force_json:
            payload["format"] = "json"

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                # 防御性检查
                if "error" in data:
                    raise RuntimeError(f"Ollama 返回错误: {data['error']}")
                # 兼容两种响应格式
                content = self._extract_content(data)
                if content is None:
                    raise RuntimeError(f"响应结构异常，无法提取回复内容: {json.dumps(data, ensure_ascii=False)[:500]}")
                # 提取 token 用量
                if self.token_tracker:
                    self._record_usage(data)
                return content
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    print(f"  连接失败，{wait}s 后重试: {e}")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Ollama 连接失败 ({self.max_retries} 次重试后): {e}")
            except requests.HTTPError as e:
                raise RuntimeError(f"Ollama HTTP 错误: {e}\n响应: {resp.text}")

    @staticmethod
    def _extract_content(data: dict) -> str:
        """从响应中提取回复文本，兼容原生 Ollama 和 OpenAI 格式。

        原生 Ollama:  {"message": {"content": "..."}}
        OpenAI 兼容:  {"choices": [{"message": {"content": "..."}}]}
        """
        # 原生 Ollama 格式
        msg = data.get("message")
        if isinstance(msg, dict) and "content" in msg:
            return msg["content"]
        # OpenAI 兼容格式
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message")
            if isinstance(msg, dict) and "content" in msg:
                return msg["content"]
        return None

    def _record_usage(self, data: dict):
        """从 Ollama 响应中提取 token 用量并上报给 tracker。

        兼容两种响应格式:
        - 原生 Ollama: prompt_eval_count / eval_count
        - OpenAI 兼容: usage.prompt_tokens / usage.completion_tokens
        """
        prompt_tokens = 0
        completion_tokens = 0
        # OpenAI 兼容格式
        usage = data.get("usage")
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
        else:
            # 原生 Ollama 格式
            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)
        if prompt_tokens or completion_tokens:
            self.token_tracker.record(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                stage=self._current_stage,
            )

    def chat_json(self, messages: list, temperature: float = None) -> dict:
        """发送聊天请求，解析 JSON 响应。

        带有容错：尝试直接解析 → 提取代码块 → 提取花括号 → 重试。
        """
        current_messages = list(messages)

        for attempt in range(self.max_retries):
            text = self.chat(current_messages, temperature)
            result = self._try_parse_json(text)
            if result is not None:
                return result

            # 解析失败，追加重试提示
            print(f"  JSON 解析失败 (尝试 {attempt + 1}/{self.max_retries})，重试中...")
            current_messages = list(messages) + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": "你的回答不是有效 JSON。请只返回纯 JSON，不要任何额外文字或 markdown 代码块标记。"},
            ]

        raise RuntimeError(f"JSON 解析失败 ({self.max_retries} 次重试后)。最后的响应:\n{text[:500]}")

    @staticmethod
    def _try_parse_json(text: str):
        """尝试从文本中提取 JSON，返回 dict 或 None。"""
        # 0. 预处理：修复常见的 LLM JSON 错误
        cleaned = OllamaClient._repair_json(text)

        # 1. 直接解析（先尝试修复版，再尝试原始版）
        for candidate in [cleaned, text]:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # 2. 提取 ```json ... ``` 代码块
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            block = OllamaClient._repair_json(match.group(1))
            for candidate in [block, match.group(1)]:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

        # 3. 提取最外层 { ... }
        start = text.find("{")
        if start != -1:
            # 找到匹配的 }
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        fragment = text[start:i + 1]
                        repaired = OllamaClient._repair_json(fragment)
                        for candidate in [repaired, fragment]:
                            try:
                                return json.loads(candidate)
                            except json.JSONDecodeError:
                                pass
                        break

        # 4. 提取最外层 [ ... ]（数组情况）
        start = text.find("[")
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "[":
                    depth += 1
                elif text[i] == "]":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break

        return None

    @staticmethod
    def _repair_json(text: str) -> str:
        """尝试修复常见的 LLM JSON 错误。"""
        # 修复 "value", "", "" → 去掉无 key 的孤立空字符串
        text = re.sub(r',\s*""(?!\s*:)', '', text)

        return text

    def check_connection(self) -> bool:
        """检查 Ollama 服务是否可用。"""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def list_models(self) -> list:
        """列出可用模型。"""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []
