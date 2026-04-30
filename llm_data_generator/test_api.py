"""快速测试外部 API 连通性 + Token 统计。"""

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

    print(f"配置:")
    print(f"  API:   {cfg.api_url}")
    print(f"  模型:  {cfg.model}")
    print(f"  Key:   {cfg.api_key[:8]}...{cfg.api_key[-4:]}")
    print()

    # 1. 连通性
    print("1. 测试连接...")
    ok = client.check_connection()
    print(f"   结果: {'OK' if ok else 'FAILED'}")
    if not ok:
        print("   无法连接，请检查 api_url 和网络。")
        return

    # 2. 普通对话
    print("\n2. 测试普通对话...")
    client._current_stage = "test_chat"
    resp = client.chat(
        [{"role": "user", "content": "用一句话介绍你自己。"}],
        force_json=False,
    )
    if resp:
        print(f"   回复: {resp[:200]}")
    else:
        print(f"   回复为空（该模型可能将输出放在 reasoning_tokens 中）")
        print(f"   Token 统计仍然可用（见下方）")

    # 3. JSON 输出
    print("\n3. 测试 JSON 输出...")
    client._current_stage = "test_json"
    try:
        resp = client.chat_json(
            [{"role": "user", "content": '请返回 JSON: {"status": "ok", "message": "hello"}'}],
        )
        print(f"   JSON: {resp}")
    except RuntimeError as e:
        print(f"   JSON 解析失败: {e}")
        print(f"   该模型可能不支持 JSON mode，但 Token 统计仍然可用")

    # 4. Token 统计
    tracker.report()

    print("测试完成。")


if __name__ == "__main__":
    main()
