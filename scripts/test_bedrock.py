#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3"]
# ///
"""
Test AWS Bedrock Models
Lists all available models and tests which ones work with your credentials.

Usage:
    uv run scripts/test_bedrock.py
    uv run scripts/test_bedrock.py --region us-west-2
    uv run scripts/test_bedrock.py --test-all        # Test every model (slow)
    uv run scripts/test_bedrock.py --test-configured # Test only beepboop + titan-embed
"""

import argparse
import json
import sys
from typing import Any

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
NC = "\033[0m"  # No color


def print_header(text: str) -> None:
    print(f"\n{BLUE}{'═' * 70}")
    print(f"  {text}")
    print(f"{'═' * 70}{NC}\n")


def print_ok(text: str) -> None:
    print(f"  {GREEN}✓{NC} {text}")


def print_fail(text: str) -> None:
    print(f"  {RED}✗{NC} {text}")


def print_warn(text: str) -> None:
    print(f"  {YELLOW}⚠{NC} {text}")


def print_info(text: str) -> None:
    print(f"  {CYAN}ℹ{NC} {text}")


def check_credentials(region: str) -> tuple[bool, dict[str, Any] | None]:
    """Check if AWS credentials are available."""
    try:
        sts = boto3.client("sts", region_name=region)
        identity = sts.get_caller_identity()
        return True, identity
    except NoCredentialsError:
        return False, None
    except ClientError as e:
        return False, {"error": str(e)}


def list_foundation_models(client) -> list[dict]:
    """List all available foundation models."""
    try:
        response = client.list_foundation_models()
        return response.get("modelSummaries", [])
    except ClientError as e:
        print_fail(f"Failed to list models: {e}")
        return []


def test_text_model(client, model_id: str) -> tuple[bool, str]:
    """Test a text/chat model with a simple prompt."""

    # Different payload formats for different model providers
    if "anthropic" in model_id:
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Say 'hello' in one word."}],
        }
    elif "amazon.titan-text" in model_id:
        payload = {
            "inputText": "Say hello in one word.",
            "textGenerationConfig": {"maxTokenCount": 100},
        }
    elif "meta.llama" in model_id:
        payload = {"prompt": "Say hello in one word.", "max_gen_len": 100}
    elif "mistral" in model_id or "mixtral" in model_id:
        payload = {"prompt": "<s>[INST] Say hello in one word. [/INST]", "max_tokens": 100}
    elif "cohere.command" in model_id:
        payload = {"prompt": "Say hello in one word.", "max_tokens": 100}
    elif "ai21" in model_id:
        payload = {"prompt": "Say hello in one word.", "maxTokens": 100}
    else:
        # Generic fallback
        payload = {"prompt": "Say hello.", "max_tokens": 50}

    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        return True, f"Response received ({len(str(result))} bytes)"
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_msg = e.response.get("Error", {}).get("Message", str(e))
        if error_code == "AccessDeniedException":
            return False, "Access denied (not enabled or no permission)"
        elif error_code == "ValidationException":
            return False, f"Validation error: {error_msg[:50]}"
        else:
            return False, f"{error_code}: {error_msg[:50]}"
    except Exception as e:
        return False, str(e)[:50]


def test_embedding_model(client, model_id: str) -> tuple[bool, str]:
    """Test an embedding model."""

    if "amazon.titan-embed" in model_id:
        payload = {"inputText": "Hello world"}
    elif "cohere.embed" in model_id:
        payload = {"texts": ["Hello world"], "input_type": "search_document"}
    else:
        payload = {"inputText": "Hello world"}

    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())

        # Check for embedding in response
        embedding = result.get("embedding") or result.get("embeddings", [[]])[0]
        if embedding:
            dims = len(embedding)
            return True, f"Embedding: {dims} dimensions"
        return True, "Response received (no embedding found)"
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_msg = e.response.get("Error", {}).get("Message", str(e))
        if error_code == "AccessDeniedException":
            return False, "Access denied (not enabled)"
        else:
            return False, f"{error_code}: {error_msg[:50]}"
    except Exception as e:
        return False, str(e)[:50]


def test_configured_models(client, region: str) -> None:
    """Test the specific models configured for LightRAG."""
    print_header("Testing Configured Models")

    models_to_test = [
        {
            "name": "beepboop",
            "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "type": "llm",
            "description": "Claude 3.5 Sonnet (LLM for extraction/queries)",
        },
        {
            "name": "titan-embed",
            "model_id": "amazon.titan-embed-text-v2:0",
            "type": "embedding",
            "description": "Titan Embed Text v2 (1024 dims)",
        },
    ]

    results = []

    for model in models_to_test:
        print(f"\n  Testing {CYAN}{model['name']}{NC}")
        print(f"    Model ID: {model['model_id']}")
        print(f"    Type: {model['type']}")
        print(f"    Description: {model['description']}")

        if model["type"] == "embedding":
            success, msg = test_embedding_model(client, model["model_id"])
        else:
            success, msg = test_text_model(client, model["model_id"])

        if success:
            print_ok(f"    {msg}")
            results.append((model["name"], True))
        else:
            print_fail(f"    {msg}")
            results.append((model["name"], False))

    # Summary
    print(f"\n{BLUE}{'─' * 70}{NC}")
    passed = sum(1 for _, ok in results if ok)
    total = len(results)

    if passed == total:
        print(f"\n  {GREEN}✓ All {total} configured models working!{NC}")
        print(f"\n  LightRAG is ready to use with AWS Bedrock.")
    else:
        print(f"\n  {YELLOW}⚠ {passed}/{total} models working{NC}")
        print(f"\n  Failed models may need to be enabled in AWS Bedrock console:")
        print(f"    https://console.aws.amazon.com/bedrock/home?region={region}#/modelaccess")


def test_all_models(client, models: list[dict]) -> None:
    """Test all available models."""
    print_header("Testing All Available Models")

    # Categorize models
    text_models = []
    embedding_models = []
    image_models = []
    other_models = []

    for model in models:
        model_id = model.get("modelId", "")
        modalities = model.get("outputModalities", [])

        if "EMBEDDING" in modalities:
            embedding_models.append(model)
        elif "IMAGE" in modalities:
            image_models.append(model)
        elif "TEXT" in modalities:
            text_models.append(model)
        else:
            other_models.append(model)

    results = {"text": [], "embedding": [], "skipped": []}

    # Test text models
    if text_models:
        print(f"\n  {CYAN}Text/Chat Models ({len(text_models)}){NC}")
        for model in text_models:
            model_id = model.get("modelId", "")
            provider = model.get("providerName", "Unknown")

            # Skip on-demand throughput versions (duplicates)
            if ":0" not in model_id and model_id.count(":") > 0:
                continue

            print(f"\n    {model_id}")
            success, msg = test_text_model(client, model_id)
            if success:
                print_ok(f"      {msg}")
                results["text"].append((model_id, True, msg))
            else:
                print_fail(f"      {msg}")
                results["text"].append((model_id, False, msg))

    # Test embedding models
    if embedding_models:
        print(f"\n\n  {CYAN}Embedding Models ({len(embedding_models)}){NC}")
        for model in embedding_models:
            model_id = model.get("modelId", "")

            # Skip on-demand throughput versions
            if ":0" not in model_id and model_id.count(":") > 0:
                continue

            print(f"\n    {model_id}")
            success, msg = test_embedding_model(client, model_id)
            if success:
                print_ok(f"      {msg}")
                results["embedding"].append((model_id, True, msg))
            else:
                print_fail(f"      {msg}")
                results["embedding"].append((model_id, False, msg))

    # Skip image models
    if image_models:
        print(f"\n\n  {YELLOW}Image Models ({len(image_models)}) - Skipped{NC}")
        for model in image_models:
            results["skipped"].append(model.get("modelId", ""))

    # Summary
    print_header("Summary")

    text_ok = sum(1 for _, ok, _ in results["text"] if ok)
    embed_ok = sum(1 for _, ok, _ in results["embedding"] if ok)

    print(f"  Text Models:      {GREEN}{text_ok}{NC}/{len(results['text'])} working")
    print(f"  Embedding Models: {GREEN}{embed_ok}{NC}/{len(results['embedding'])} working")
    print(f"  Image Models:     {YELLOW}{len(results['skipped'])}{NC} skipped")

    # List working models
    if text_ok > 0:
        print(f"\n  {GREEN}Working Text Models:{NC}")
        for model_id, ok, msg in results["text"]:
            if ok:
                print(f"    • {model_id}")

    if embed_ok > 0:
        print(f"\n  {GREEN}Working Embedding Models:{NC}")
        for model_id, ok, msg in results["embedding"]:
            if ok:
                print(f"    • {model_id} ({msg})")


def main():
    parser = argparse.ArgumentParser(description="Test AWS Bedrock models")
    parser.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument("--test-all", action="store_true", help="Test all available models")
    parser.add_argument("--test-configured", action="store_true", help="Test only LightRAG configured models")
    parser.add_argument("--list", action="store_true", help="List available models without testing")
    args = parser.parse_args()

    print_header("AWS Bedrock Model Tester")

    # Check credentials
    print(f"  Region: {args.region}")
    print(f"  Checking AWS credentials...")

    creds_ok, identity = check_credentials(args.region)
    if not creds_ok:
        print_fail("No valid AWS credentials found")
        if identity and "error" in identity:
            print(f"      {identity['error']}")
        print_info("Make sure AWS_WEB_IDENTITY_TOKEN_FILE/AWS_ROLE_ARN or AWS_ACCESS_KEY_ID is set")
        sys.exit(1)

    print_ok(f"Credentials valid")
    print(f"      Account: {identity['Account']}")
    print(f"      ARN: {identity['Arn']}")

    # Create Bedrock client
    bedrock = boto3.client("bedrock", region_name=args.region)
    bedrock_runtime = boto3.client("bedrock-runtime", region_name=args.region)

    # List models
    print(f"\n  Fetching available models...")
    models = list_foundation_models(bedrock)

    if not models:
        print_fail("No models found or unable to list models")
        sys.exit(1)

    print_ok(f"Found {len(models)} models")

    # List only
    if args.list:
        print_header("Available Models")

        by_provider = {}
        for model in models:
            provider = model.get("providerName", "Unknown")
            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append(model)

        for provider, provider_models in sorted(by_provider.items()):
            print(f"\n  {CYAN}{provider}{NC} ({len(provider_models)} models)")
            for model in provider_models:
                model_id = model.get("modelId", "")
                modalities = ", ".join(model.get("outputModalities", []))
                print(f"    • {model_id}")
                print(f"      Modalities: {modalities}")
        return

    # Test configured models (default)
    if args.test_configured or not args.test_all:
        test_configured_models(bedrock_runtime, args.region)

    # Test all models
    if args.test_all:
        test_all_models(bedrock_runtime, models)

    print()


if __name__ == "__main__":
    main()
