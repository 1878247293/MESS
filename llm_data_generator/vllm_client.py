"""vLLM OpenAI-compatible API client, with the same interface as OllamaClient."""

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
        """Send a chat request and return the assistant's reply text."""
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
                    raise RuntimeError(f"vLLM returned an error: {data['error']}")
                content = data["choices"][0]["message"]["content"]
                # extract token usage
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
                    print(f"  connection failed, retrying in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"vLLM connection failed (after {self.max_retries} retries): {e}")
            except requests.HTTPError as e:
                raise RuntimeError(f"vLLM HTTP error: {e}\nresponse: {resp.text}")

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
        """Repair common LLM JSON errors."""
        text = re.sub(r',\s*""(?!\s*:)', '', text)
        return text

    def check_connection(self) -> bool:
        """Check whether the vLLM service is available."""
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def list_models(self) -> list:
        """List available models."""
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=5)
            resp.raise_for_status()
            return [m["id"] for m in resp.json().get("data", [])]
        except Exception:
            return []
