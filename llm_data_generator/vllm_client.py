"""vLLM OpenAI 兼容 API 客户端，与 OllamaClient 接口一致。"""

import json
import re
import time
import requests


class VllmClient:
    def __init__(self, base_url: str, model: str, temperature: float,
                 max_retries: int, timeout: int, num_ctx: int,
                 token_tracker=None):
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
        """发送聊天请求，返回助手回复文本。"""
        temp = temperature if temperature is not None else self.temperature
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": self.num_ctx,
        }
        if force_json:
            payload["response_format"] = {"type": "json_object"}

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                if "error" in data:
                    raise RuntimeError(f"vLLM 返回错误: {data['error']}")
                content = data["choices"][0]["message"]["content"]
                # 提取 token 用量
                if self.token_tracker and "usage" in data:
                    u = data["usage"]
                    self.token_tracker.record(
                        prompt_tokens=u.get("prompt_tokens", 0),
                        completion_tokens=u.get("completion_tokens", 0),
                        total_tokens=u.get("total_tokens", 0),
                        stage=self._current_stage,
                    )
                return content
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    print(f"  连接失败，{wait}s 后重试: {e}")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"vLLM 连接失败 ({self.max_retries} 次重试后): {e}")
            except requests.HTTPError as e:
                raise RuntimeError(f"vLLM HTTP 错误: {e}\n响应: {resp.text}")

    def chat_json(self, messages: list, temperature: float = None) -> dict:
        """发送聊天请求，解析 JSON 响应（带容错重试）。"""
        current_messages = list(messages)

        for attempt in range(self.max_retries):
            text = self.chat(current_messages, temperature)
            result = self._try_parse_json(text)
            if result is not None:
                return result

            print(f"  JSON 解析失败 (尝试 {attempt + 1}/{self.max_retries})，重试中...")
            current_messages = list(messages) + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": "你的回答不是有效 JSON。请只返回纯 JSON，不要任何额外文字或 markdown 代码块标记。"},
            ]

        raise RuntimeError(f"JSON 解析失败 ({self.max_retries} 次重试后)。最后的响应:\n{text[:500]}")

    @staticmethod
    def _try_parse_json(text: str):
        """尝试从文本中提取 JSON，返回 dict 或 None。"""
        cleaned = VllmClient._repair_json(text)

        for candidate in [cleaned, text]:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            block = VllmClient._repair_json(match.group(1))
            for candidate in [block, match.group(1)]:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

        start = text.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        fragment = text[start:i + 1]
                        repaired = VllmClient._repair_json(fragment)
                        for candidate in [repaired, fragment]:
                            try:
                                return json.loads(candidate)
                            except json.JSONDecodeError:
                                pass
                        break

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
        """修复常见的 LLM JSON 错误。"""
        text = re.sub(r',\s*""(?!\s*:)', '', text)
        return text

    def check_connection(self) -> bool:
        """检查 vLLM 服务是否可用。"""
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def list_models(self) -> list:
        """列出可用模型。"""
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=5)
            resp.raise_for_status()
            return [m["id"] for m in resp.json().get("data", [])]
        except Exception:
            return []
