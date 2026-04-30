"""OpenAI 兼容 API 客户端（支持 API Key 认证），适用于各类云端 LLM 服务。

兼容：OpenAI、DeepSeek、阿里通义千问、硅基流动、智谱等 OpenAI 兼容接口。
"""

import json
import re
import time
import requests


class ApiClient:
    def __init__(self, base_url: str, model: str, api_key: str,
                 temperature: float, max_retries: int, timeout: int,
                 num_ctx: int, token_tracker=None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.token_tracker = token_tracker
        self._current_stage = "default"

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(self, messages: list, temperature: float = None,
             force_json: bool = True, stream: bool = True) -> str:
        """发送聊天请求,返回助手回复文本。

        默认走 **流式**(stream=True):SSE 分片持续写入数据,保持 TCP 连接活跃,
        规避中转 / 网关的 idle 超时 (表现为 HTTP 499)。长 reasoning 模型
        (gpt-5.x / o-series) 强烈建议保持流式。

        stream=False 时走旧的非流式路径,并保留 "content 为空时自动回退流式"
        的兼容行为,用于确实需要一次性拿到完整响应的短请求。
        """
        temp = temperature if temperature is not None else self.temperature
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": self.num_ctx,
        }
        if force_json:
            payload["response_format"] = {"type": "json_object"}

        if stream:
            return self._chat_stream_with_retry(payload)

        # ---- stream=False:非流式 ----
        for attempt in range(self.max_retries):
            resp = None
            try:
                resp = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                try:
                    data = resp.json()
                except (ValueError, requests.exceptions.JSONDecodeError):
                    return self._chat_stream(payload)
                if "error" in data:
                    raise RuntimeError(f"API 返回错误: {data['error']}")
                content = data["choices"][0]["message"]["content"] or ""
                if self.token_tracker and "usage" in data:
                    u = data["usage"]
                    self.token_tracker.record(
                        prompt_tokens=u.get("prompt_tokens", 0),
                        completion_tokens=u.get("completion_tokens", 0),
                        total_tokens=u.get("total_tokens", 0),
                        stage=self._current_stage,
                    )
                if not content and data.get("usage", {}).get("completion_tokens", 0) > 0:
                    return self._chat_stream(payload)
                return content
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    print(f"  连接失败,{wait}s 后重试: {e}")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"API 连接失败 ({self.max_retries} 次重试后): {e}")
            except requests.HTTPError as e:
                error_body = resp.text[:500] if resp is not None else ""
                status = resp.status_code if resp is not None else 0
                if status in (429, 500, 502, 503, 504):
                    if attempt < self.max_retries - 1:
                        wait = 2 ** (attempt + 1)
                        print(f"  服务器错误 ({status}),{wait}s 后重试...")
                        time.sleep(wait)
                        continue
                raise RuntimeError(f"API HTTP 错误: {e}\n响应: {error_body}")

    def _chat_stream_with_retry(self, payload: dict) -> str:
        """流式 chat,带退避重试。触发 499 / 连接断开时自动再来。"""
        for attempt in range(self.max_retries):
            try:
                return self._chat_stream(payload)
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    print(f"  流式连接失败,{wait}s 后重试: {e}")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"API 连接失败 ({self.max_retries} 次重试后): {e}")
            except requests.HTTPError as e:
                resp = getattr(e, "response", None)
                status = resp.status_code if resp is not None else 0
                body = resp.text[:500] if resp is not None else ""
                # 499 / 408 / 429 / 5xx 都视为可重试
                if status in (408, 429, 499, 500, 502, 503, 504):
                    if attempt < self.max_retries - 1:
                        wait = 2 ** (attempt + 1)
                        print(f"  上游瞬时错误 ({status}),{wait}s 后重试...")
                        time.sleep(wait)
                        continue
                raise RuntimeError(f"API HTTP 错误: {e}\n响应: {body}")
        raise RuntimeError("流式请求重试全部失败")

    def _chat_stream(self, payload: dict) -> str:
        """流式接收响应内容。

        - 自动附加 stream_options.include_usage,确保最后一个 chunk 带 usage。
        - 连接超时与读超时拆分:连接 30s 即可;读端给足 self.timeout,
          但因为每个 SSE 分片都刷新 socket 读计时,不会卡死。
        """
        payload = {
            **payload,
            "stream": True,
            # OpenAI 规范:include_usage=true 时,在最后一个 (choices=[]) chunk 返回 usage
            "stream_options": {"include_usage": True},
        }
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=self._headers(),
            timeout=(30, self.timeout),
            stream=True,
        )
        resp.raise_for_status()

        chunks = []
        usage_seen = False
        for line in resp.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8", errors="ignore")
            if not text.startswith("data: "):
                continue
            if text.strip() == "data: [DONE]":
                break

            raw = text[6:]
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # usage 通常出现在最后一个 chunk(choices 为空)
            if self.token_tracker and chunk.get("usage"):
                u = chunk["usage"]
                self.token_tracker.record(
                    prompt_tokens=u.get("prompt_tokens", 0),
                    completion_tokens=u.get("completion_tokens", 0),
                    total_tokens=u.get("total_tokens", 0),
                    stage=self._current_stage,
                )
                usage_seen = True

            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content")
            if piece:
                chunks.append(piece)

        if not usage_seen and self.token_tracker:
            # 某些中转不遵循 stream_options,仅在非流式或 header 返回 usage
            # 这里不再强求,后续可从 response.headers 里捞 openai-* 计费头
            pass

        return "".join(chunks)

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
        cleaned = ApiClient._repair_json(text)

        for candidate in [cleaned, text]:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            block = ApiClient._repair_json(match.group(1))
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
                        repaired = ApiClient._repair_json(fragment)
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
        """检查 API 服务是否可用。"""
        try:
            resp = requests.get(
                f"{self.base_url}/v1/models",
                headers=self._headers(),
                timeout=10,
            )
            return resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def list_models(self) -> list:
        """列出可用模型。"""
        try:
            resp = requests.get(
                f"{self.base_url}/v1/models",
                headers=self._headers(),
                timeout=10,
            )
            resp.raise_for_status()
            return [m["id"] for m in resp.json().get("data", [])]
        except Exception:
            return []
