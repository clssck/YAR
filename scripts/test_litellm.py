#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["openai"]
# ///
"""
Test LiteLLM proxy endpoints for LLM and Embedding.

Usage:
    uv run scripts/test_litellm.py
    uv run scripts/test_litellm.py --host 172.28.0.1 --port 4000
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description='Test LiteLLM proxy')
    parser.add_argument('--host', default='172.28.0.1', help='LiteLLM host')
    parser.add_argument('--port', default='4000', help='LiteLLM port')
    parser.add_argument('--key', default='sk-litellm-master-key', help='API key')
    args = parser.parse_args()

    import openai
    from openai import OpenAI

    base_url = f'http://{args.host}:{args.port}/v1'
    client = OpenAI(base_url=base_url, api_key=args.key)

    print(f'Testing LiteLLM at {base_url}\n')
    print('=' * 60)

    success = True

    # Test embedding
    print('\n1. Embedding (titan-embed):')
    try:
        resp = client.embeddings.create(model='titan-embed', input='hello world')
        dims = len(resp.data[0].embedding)
        print(f'   ✓ Success: {dims} dimensions')
        print(f'   First 5 values: {resp.data[0].embedding[:5]}')
    except openai.AuthenticationError as e:
        print(f'   ✗ Authentication failed: {e}')
        success = False
    except openai.APIConnectionError as e:
        print(f'   ✗ Connection failed: {e}')
        success = False
    except openai.APIError as e:
        print(f'   ✗ API error: {e}')
        success = False
    except Exception as e:
        print(f'   ✗ Unexpected error: {e}')
        success = False

    # Test chat
    print('\n2. Chat (beepboop / Claude 3.5 Sonnet):')
    try:
        resp = client.chat.completions.create(
            model='beepboop',
            messages=[{'role': 'user', 'content': 'Say hello in exactly 5 words.'}],
            max_tokens=50
        )
        content = resp.choices[0].message.content
        print(f'   ✓ Success: "{content}"')
        print(f'   Tokens: {resp.usage.prompt_tokens} prompt, {resp.usage.completion_tokens} completion')
    except openai.AuthenticationError as e:
        print(f'   ✗ Authentication failed: {e}')
        success = False
    except openai.APIConnectionError as e:
        print(f'   ✗ Connection failed: {e}')
        success = False
    except openai.APIError as e:
        print(f'   ✗ API error: {e}')
        success = False
    except Exception as e:
        print(f'   ✗ Unexpected error: {e}')
        success = False

    print('\n' + '=' * 60)
    if success:
        print('✓ All tests passed!')
    else:
        print('✗ Some tests failed')
        sys.exit(1)

if __name__ == '__main__':
    main()
