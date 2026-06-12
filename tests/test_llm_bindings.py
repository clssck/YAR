from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestOpenAIClientCreation:
    def test_create_openai_client_with_defaults(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            from yar.llm.openai import create_openai_async_client

            client = create_openai_async_client()
            assert client is not None
            assert client.api_key == 'test-key'

    def test_create_openai_client_with_custom_base_url(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            from yar.llm.openai import create_openai_async_client

            client = create_openai_async_client(base_url='https://custom.api.com/v1')
            assert client is not None
            assert str(client.base_url) == 'https://custom.api.com/v1/'

    def test_create_openai_client_with_explicit_api_key(self):
        from yar.llm.openai import create_openai_async_client

        client = create_openai_async_client(api_key='explicit-key')
        assert client.api_key == 'explicit-key'

    def test_create_azure_client(self):
        from yar.llm.openai import create_openai_async_client

        client = create_openai_async_client(
            api_key='azure-key',
            base_url='https://myresource.openai.azure.com/',
            use_azure=True,
            azure_deployment='gpt-4',
            api_version='2024-02-15-preview',
        )
        assert client is not None


def _create_chat_response(content: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    mock_response = MagicMock(spec=['choices', 'usage'])
    mock_choice = MagicMock(spec=['message'])
    mock_message = MagicMock(spec=['content'])
    mock_message.content = content
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return mock_response


class TestOpenAIComplete:
    @pytest.mark.asyncio
    async def test_openai_complete_basic(self):
        mock_response = _create_chat_response('Test response')

        with patch('yar.llm.openai.create_openai_async_client') as mock_client_factory:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_client_factory.return_value = mock_client

            from yar.llm.openai import openai_complete_if_cache

            result = await openai_complete_if_cache(
                model='gpt-4',
                prompt='Hello',
                api_key='test-key',
            )

            assert result == 'Test response'
            mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_openai_complete_with_system_prompt(self):
        mock_response = _create_chat_response('Response with system', 15, 5)

        with patch('yar.llm.openai.create_openai_async_client') as mock_client_factory:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_client_factory.return_value = mock_client

            from yar.llm.openai import openai_complete_if_cache

            result = await openai_complete_if_cache(
                model='gpt-4',
                prompt='Hello',
                system_prompt='You are a helpful assistant',
                api_key='test-key',
            )

            assert result == 'Response with system'
            call_kwargs = mock_client.chat.completions.create.call_args
            messages = call_kwargs[1]['messages']
            assert messages[0]['role'] == 'system'
            assert messages[0]['content'] == 'You are a helpful assistant'

    @pytest.mark.asyncio
    async def test_openai_complete_retries_transient_failure_without_error_logging(self):
        import httpx
        from openai import APIConnectionError
        from tenacity import wait_none

        from yar.llm import openai as openai_module

        mock_response = _create_chat_response('Recovered response')
        transient_error = APIConnectionError(
            request=httpx.Request('POST', 'https://example.com/v1/chat/completions')
        )

        with (
            patch('yar.llm.openai.create_openai_async_client') as mock_client_factory,
            patch.object(openai_module.logger, 'error') as mock_log_error,
        ):
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=[transient_error, mock_response]
            )
            mock_client.close = AsyncMock()
            mock_client_factory.return_value = mock_client

            retry_state = cast(Any, openai_module.openai_complete_if_cache).retry
            original_wait = retry_state.wait
            retry_state.wait = wait_none()
            try:
                result = await openai_module.openai_complete_if_cache(
                    model='gpt-4',
                    prompt='Hello',
                    api_key='test-key',
                )
            finally:
                retry_state.wait = original_wait

            assert result == 'Recovered response'
            assert mock_client.chat.completions.create.await_count == 2
            assert mock_client.close.await_count == 2
            mock_log_error.assert_not_called()


    @pytest.mark.asyncio
    async def test_openai_complete_empty_response_raises(self):
        from tenacity import RetryError

        mock_response = _create_chat_response('', 10, 0)

        with patch('yar.llm.openai.create_openai_async_client') as mock_client_factory:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_client_factory.return_value = mock_client

            from yar.llm.openai import openai_complete_if_cache

            with pytest.raises(RetryError):
                await openai_complete_if_cache(
                    model='gpt-4',
                    prompt='Hello',
                    api_key='test-key',
                )


class TestTiktokenEncoding:
    def test_get_encoding_for_known_model(self):
        from yar.llm.openai import _get_tiktoken_encoding_for_model

        encoding = _get_tiktoken_encoding_for_model('gpt-4')
        assert encoding is not None

    def test_get_encoding_for_unknown_model_falls_back(self):
        from yar.llm.openai import _get_tiktoken_encoding_for_model

        encoding = _get_tiktoken_encoding_for_model('unknown-model-xyz')
        assert encoding is not None

    def test_encoding_caching(self):
        from yar.llm.openai import _TIKTOKEN_ENCODING_CACHE, _get_tiktoken_encoding_for_model

        _TIKTOKEN_ENCODING_CACHE.clear()
        enc1 = _get_tiktoken_encoding_for_model('gpt-4')
        enc2 = _get_tiktoken_encoding_for_model('gpt-4')
        assert enc1 is enc2


class TestOpenAIEmbedding:
    @pytest.mark.asyncio
    async def test_openai_embed_basic(self):
        mock_response = MagicMock(spec=['data', 'usage'])
        mock_embedding = MagicMock(spec=['embedding'])
        mock_embedding.embedding = [0.1] * 1536
        mock_response.data = [mock_embedding]
        mock_response.usage = MagicMock(prompt_tokens=5, total_tokens=5)

        with patch('yar.llm.openai.create_openai_async_client') as mock_client_factory:
            mock_client = AsyncMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_client_factory.return_value = mock_client

            from yar.llm.openai import openai_embed

            result = await openai_embed(
                texts=['test text'],
                model='text-embedding-3-small',
                api_key='test-key',
            )

            assert result is not None
            assert len(result) == 1
            assert len(result[0]) == 1536


class TestTokenUsageCapture:
    """Fix C: the binding forwards reasoning + cached token sub-counts to the token tracker."""

    @pytest.mark.asyncio
    async def test_tracker_captures_reasoning_and_cached(self):
        from yar.llm.openai import openai_complete_if_cache
        from yar.utils import TokenTracker

        mock_response = MagicMock(spec=['choices', 'usage'])
        mock_choice = MagicMock(spec=['message'])
        mock_message = MagicMock(spec=['content'])
        mock_message.content = 'ok'
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        usage = MagicMock(
            spec=[
                'prompt_tokens', 'completion_tokens', 'total_tokens',
                'completion_tokens_details', 'prompt_tokens_details',
            ]
        )
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        usage.total_tokens = 150
        usage.completion_tokens_details = MagicMock(spec=['reasoning_tokens'])
        usage.completion_tokens_details.reasoning_tokens = 30
        usage.prompt_tokens_details = MagicMock(spec=['cached_tokens'])
        usage.prompt_tokens_details.cached_tokens = 80
        mock_response.usage = usage

        with patch('yar.llm.openai.create_openai_async_client') as mock_client_factory:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_client_factory.return_value = mock_client

            tracker = TokenTracker()
            await openai_complete_if_cache(model='gpt-4', prompt='Hi', api_key='k', token_tracker=tracker)

        usage_out = tracker.get_usage()
        assert usage_out['prompt_tokens'] == 100
        assert usage_out['completion_tokens'] == 50
        assert usage_out['reasoning_tokens'] == 30
        assert usage_out['cached_tokens'] == 80

    @pytest.mark.asyncio
    async def test_tracker_defaults_to_zero_without_details(self):
        from yar.llm.openai import openai_complete_if_cache
        from yar.utils import TokenTracker

        mock_response = MagicMock(spec=['choices', 'usage'])
        mock_choice = MagicMock(spec=['message'])
        mock_message = MagicMock(spec=['content'])
        mock_message.content = 'ok'
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        usage = MagicMock(spec=['prompt_tokens', 'completion_tokens', 'total_tokens'])
        usage.prompt_tokens = 10
        usage.completion_tokens = 4
        usage.total_tokens = 14
        mock_response.usage = usage

        with patch('yar.llm.openai.create_openai_async_client') as mock_client_factory:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_client_factory.return_value = mock_client

            tracker = TokenTracker()
            await openai_complete_if_cache(model='gpt-4', prompt='Hi', api_key='k', token_tracker=tracker)

        usage_out = tracker.get_usage()
        assert usage_out['reasoning_tokens'] == 0
        assert usage_out['cached_tokens'] == 0
