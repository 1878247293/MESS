"""Quick test of external API connectivity + token stats."""

from .api_client import ApiClient
from .token_tracker import TokenTracker
from .config import GeneratorConfig


def main():
    cfg = GeneratorConfig()
    tracker = TokenTracker(model=cfg.model)

    client = ApiClient(
        base_url=cfg.api_url,
        model=cfg.model,
        api_key=cfg.api_key,
        temperature=0.7,
        max_retries=3,
        timeout=60,
        num_ctx=4096,
        token_tracker=tracker,
    )

    print(f"Config:")
    print(f"  API:   {cfg.api_url}")
    print(f"  model: {cfg.model}")
    print(f"  Key:   {cfg.api_key[:8]}...{cfg.api_key[-4:]}")
    print()

    # 1. connectivity
    print("1. Testing connection...")
    ok = client.check_connection()
    print(f"   result: {'OK' if ok else 'FAILED'}")
    if not ok:
        print("   Cannot connect; please check api_url and the network.")
        return

    # 2. plain chat
    print("\n2. Testing plain chat...")
    client._current_stage = "test_chat"
    resp = client.chat(
        [{"role": "user", "content": "Introduce yourself in one sentence."}],
        force_json=False,
    )
    if resp:
        print(f"   reply: {resp[:200]}")
    else:
        print(f"   empty reply (this model may place its output in reasoning_tokens)")
        print(f"   token stats are still available (see below)")

    # 3. JSON output
    print("\n3. Testing JSON output...")
    client._current_stage = "test_json"
    try:
        resp = client.chat_json(
            [{"role": "user", "content": 'Return JSON: {"status": "ok", "message": "hello"}'}],
        )
        print(f"   JSON: {resp}")
    except RuntimeError as e:
        print(f"   JSON parsing failed: {e}")
        print(f"   this model may not support JSON mode, but token stats are still available")

    # 4. token stats
    tracker.report()

    print("Test complete.")


if __name__ == "__main__":
    main()
