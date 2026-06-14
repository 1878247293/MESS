"""OpenAI-compatible API client (supports API-key authentication), for various cloud LLM services.

Compatible with: OpenAI, DeepSeek, Alibaba Tongyi Qianwen, SiliconFlow, Zhipu, and other OpenAI-compatible interfaces.
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
        """Send a chat request and return the assistant's reply text.

        Defaults to **streaming** (stream=True): SSE fragments write data continuously, keeping
        the TCP connection alive and avoiding idle timeouts of proxies / gateways (which show up as HTTP 499).
        Streaming is strongly recommended for long-reasoning models (gpt-5.x / o-series).

        When stream=False, it takes the old non-streaming path and keeps the "fall back to streaming
        automatically when content is empty" compatibility behavior, for short requests that genuinely need
        the full response in one shot.
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

        # ---- stream=False: non-streaming ----
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
                    raise RuntimeError(f"API returned an error: {data['error']}")
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
                    print(f"  connection failed, retrying in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"API connection failed (after {self.max_retries} retries): {e}")
            except requests.HTTPError as e:
                error_body = resp.text[:500] if resp is not None else ""
                status = resp.status_code if resp is not None else 0
                if status in (429, 500, 502, 503, 504):
                    if attempt < self.max_retries - 1:
                        wait = 2 ** (attempt + 1)
                        print(f"  server error ({status}), retrying in {wait}s...")
                        time.sleep(wait)
                        continue
                raise RuntimeError(f"API HTTP error: {e}\nresponse: {error_body}")

    def _chat_stream_with_retry(self, payload: dict) -> str:
        """Streaming chat with backoff retry. Automatically retries on 499 / dropped connection."""
        for attempt in range(self.max_retries):
            try:
                return self._chat_stream(payload)
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    print(f"  streaming connection failed, retrying in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"API connection failed (after {self.max_retries} retries): {e}")
            except requests.HTTPError as e:
                resp = getattr(e, "response", None)
                status = resp.status_code if resp is not None else 0
                body = resp.text[:500] if resp is not None else ""
                # 499 / 408 / 429 / 5xx are all treated as retryable
                if status in (408, 429, 499, 500, 502, 503, 504):
                    if attempt < self.max_retries - 1:
                        wait = 2 ** (attempt + 1)
                        print(f"  upstream transient error ({status}), retrying in {wait}s...")
                        time.sleep(wait)
                        continue
                raise RuntimeError(f"API HTTP error: {e}\nresponse: {body}")
        raise RuntimeError("all streaming-request retries failed")

    def _chat_stream(self, payload: dict) -> str:
        """Receive the response content via streaming.

        - Automatically appends stream_options.include_usage to ensure the last chunk carries usage.
        - Connection timeout and read timeout are split: 30s is enough for connecting; the read side
          is given the full self.timeout, but since each SSE fragment refreshes the socket read timer,
          it will not hang.
        """
        payload = {
            **payload,
            "stream": True,
            # OpenAI spec: when include_usage=true, usage is returned in the last (choices=[]) chunk
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

            # usage usually appears in the last chunk (choices empty)
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
            # some proxies do not honor stream_options and only return usage in non-streaming mode or headers
            # we no longer insist here; later we could pull openai-* billing headers from response.headers
            pass

        return "".join(chunks)

    def chat_json(self, messages: list, temperature: float = None) -> dict:
        """Send a chat request and parse the JSON response (with fault-tolerant retries)."""
        current_messages = list(messages)

        for attempt in range(self.max_retries):
            text = self.chat(current_messages, temperature)
            result = self._try_parse_json(text)
            if result is not None:
                return result

            print(f"  JSON parsing failed (attempt {attempt + 1}/{self.max_retries}), retrying...")
            current_messages = list(messages) + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": "Your reply was not valid JSON. Return pure JSON only, with no extra text or markdown code-block markers."},
            ]

        raise RuntimeError(f"JSON parsing failed (after {self.max_retries} retries). Last response:\n{text[:500]}")

    @staticmethod
    def _try_parse_json(text: str):
        """Try to extract JSON from the text, returning a dict or None."""
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
        """Repair common LLM JSON errors."""
        text = re.sub(r',\s*""(?!\s*:)', '', text)
        return text

    def check_connection(self) -> bool:
        """Check whether the API service is available."""
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
        """List available models."""
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
