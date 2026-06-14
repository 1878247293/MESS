"""Ollama REST API client, with fault-tolerant JSON parsing and retries."""

import json
import re
import time
import requests


class OllamaClient:
    def __init__(self, base_url: str, model: str, temperature: float,
                 max_retries: int, timeout: int, num_ctx: int,
                 token_tracker=None):
        """Initialize the Ollama client.

        Args:
            base_url: Ollama API address
            model: model name
            temperature: generation temperature
            max_retries: maximum number of retries
            timeout: request timeout (seconds)
            num_ctx: context window size (in tokens)
            token_tracker: TokenTracker instance (optional)
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
        """Send a chat request and return the assistant's reply text.

        Args:
            messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
            temperature: override the default temperature
            force_json: if True, enable Ollama JSON mode (force JSON output);
                        if False, return plain text (for generating free-form content such as Markdown)

        Returns:
            the text content of the assistant's reply
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
                # defensive check
                if "error" in data:
                    raise RuntimeError(f"Ollama returned an error: {data['error']}")
                # support both response formats
                content = self._extract_content(data)
                if content is None:
                    raise RuntimeError(f"unexpected response structure, cannot extract reply content: {json.dumps(data, ensure_ascii=False)[:500]}")
                # extract token usage
                if self.token_tracker:
                    self._record_usage(data)
                return content
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    print(f"  connection failed, retrying in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Ollama connection failed (after {self.max_retries} retries): {e}")
            except requests.HTTPError as e:
                raise RuntimeError(f"Ollama HTTP error: {e}\nresponse: {resp.text}")

    @staticmethod
    def _extract_content(data: dict) -> str:
        """Extract the reply text from the response, supporting both native Ollama and OpenAI formats.

        Native Ollama:  {"message": {"content": "..."}}
        OpenAI-compatible:  {"choices": [{"message": {"content": "..."}}]}
        """
        # native Ollama format
        msg = data.get("message")
        if isinstance(msg, dict) and "content" in msg:
            return msg["content"]
        # OpenAI-compatible format
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message")
            if isinstance(msg, dict) and "content" in msg:
                return msg["content"]
        return None

    def _record_usage(self, data: dict):
        """Extract token usage from the Ollama response and report it to the tracker.

        Supports both response formats:
        - native Ollama: prompt_eval_count / eval_count
        - OpenAI-compatible: usage.prompt_tokens / usage.completion_tokens
        """
        prompt_tokens = 0
        completion_tokens = 0
        # OpenAI-compatible format
        usage = data.get("usage")
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
        else:
            # native Ollama format
            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)
        if prompt_tokens or completion_tokens:
            self.token_tracker.record(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                stage=self._current_stage,
            )

    def chat_json(self, messages: list, temperature: float = None) -> dict:
        """Send a chat request and parse the JSON response.

        Fault-tolerant: try direct parsing → extract code block → extract braces → retry.
        """
        current_messages = list(messages)

        for attempt in range(self.max_retries):
            text = self.chat(current_messages, temperature)
            result = self._try_parse_json(text)
            if result is not None:
                return result

            # parsing failed, append a retry hint
            print(f"  JSON parsing failed (attempt {attempt + 1}/{self.max_retries}), retrying...")
            current_messages = list(messages) + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": "Your reply was not valid JSON. Return pure JSON only, with no extra text or markdown code-block markers."},
            ]

        raise RuntimeError(f"JSON parsing failed (after {self.max_retries} retries). Last response:\n{text[:500]}")

    @staticmethod
    def _try_parse_json(text: str):
        """Try to extract JSON from the text, returning a dict or None."""
        # 0. preprocessing: repair common LLM JSON errors
        cleaned = OllamaClient._repair_json(text)

        # 1. direct parsing (try the repaired version first, then the original)
        for candidate in [cleaned, text]:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # 2. extract a ```json ... ``` code block
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            block = OllamaClient._repair_json(match.group(1))
            for candidate in [block, match.group(1)]:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

        # 3. extract the outermost { ... }
        start = text.find("{")
        if start != -1:
            # find the matching }
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

        # 4. extract the outermost [ ... ] (array case)
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
        """Try to repair common LLM JSON errors."""
        # fix "value", "", "" -> remove stray empty strings without a key
        text = re.sub(r',\s*""(?!\s*:)', '', text)

        return text

    def check_connection(self) -> bool:
        """Check whether the Ollama service is available."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def list_models(self) -> list:
        """List available models."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []
