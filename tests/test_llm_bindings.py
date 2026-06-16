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


class _AsyncChunkStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    async def aclose(self):
        self.closed = True


def _create_stream_content_chunk(content: str):
    chunk = MagicMock(spec=['choices'])
    choice = MagicMock(spec=['delta'])
    delta = MagicMock(spec=['content'])
    delta.content = content
    choice.delta = delta
    chunk.choices = [choice]
    return chunk


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

    @pytest.mark.asyncio
    async def test_tracker_captures_dict_shaped_response_usage_details(self):
        from yar.llm.openai import openai_complete_if_cache
        from yar.utils import TokenTracker

        mock_response = {
            'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}],
            'usage': {
                'prompt_tokens': 100,
                'completion_tokens': 50,
                'total_tokens': 150,
                'completion_tokens_details': {'reasoning_tokens': 30},
                'prompt_tokens_details': {'cached_tokens': 80},
            },
        }

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
        assert usage_out['total_tokens'] == 150
        assert usage_out['reasoning_tokens'] == 30
        assert usage_out['cached_tokens'] == 80

    @pytest.mark.asyncio
    async def test_tracker_defaults_dict_usage_to_zero_without_details(self):
        from yar.llm.openai import openai_complete_if_cache
        from yar.utils import TokenTracker

        mock_response = {
            'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}],
            'usage': {
                'prompt_tokens': 10,
                'completion_tokens': 4,
                'total_tokens': 14,
            },
        }

        with patch('yar.llm.openai.create_openai_async_client') as mock_client_factory:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_client_factory.return_value = mock_client

            tracker = TokenTracker()
            await openai_complete_if_cache(model='gpt-4', prompt='Hi', api_key='k', token_tracker=tracker)

        usage_out = tracker.get_usage()
        assert usage_out['prompt_tokens'] == 10
        assert usage_out['completion_tokens'] == 4
        assert usage_out['total_tokens'] == 14
        assert usage_out['reasoning_tokens'] == 0
        assert usage_out['cached_tokens'] == 0

    def test_token_tracker_accumulates_reasoning_and_cached_across_calls(self):
        from yar.utils import TokenTracker

        tracker = TokenTracker()
        tracker.add_usage({
            'prompt_tokens': 10,
            'completion_tokens': 5,
            'total_tokens': 15,
            'reasoning_tokens': 2,
            'cached_tokens': 7,
        })
        tracker.add_usage({
            'prompt_tokens': 11,
            'completion_tokens': 6,
            'total_tokens': 17,
            'reasoning_tokens': 3,
            'cached_tokens': 8,
        })

        usage_out = tracker.get_usage()
        assert usage_out['prompt_tokens'] == 21
        assert usage_out['completion_tokens'] == 11
        assert usage_out['total_tokens'] == 32
        assert usage_out['reasoning_tokens'] == 5
        assert usage_out['cached_tokens'] == 15
        assert usage_out['call_count'] == 2

    @pytest.mark.asyncio
    async def test_streaming_tracker_requests_usage_and_captures_reasoning_cached(self):
        from yar.llm.openai import openai_complete_if_cache
        from yar.utils import TokenTracker

        final_chunk = MagicMock(spec=['choices', 'usage'])
        final_chunk.choices = []
        final_chunk.usage = {
            'prompt_tokens': 12,
            'completion_tokens': 8,
            'total_tokens': 20,
            'completion_tokens_details': {'reasoning_tokens': 5},
            'prompt_tokens_details': {'cached_tokens': 9},
        }
        stream = _AsyncChunkStream([
            _create_stream_content_chunk('hello '),
            _create_stream_content_chunk('world'),
            final_chunk,
        ])

        with patch('yar.llm.openai.create_openai_async_client') as mock_client_factory:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=stream)
            mock_client.close = AsyncMock()
            mock_client_factory.return_value = mock_client

            tracker = TokenTracker()
            result = await openai_complete_if_cache(
                model='gpt-4',
                prompt='Hi',
                api_key='k',
                token_tracker=tracker,
                stream=True,
            )
            chunks = []
            async for chunk in result:
                chunks.append(chunk)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        usage_out = tracker.get_usage()
        assert chunks == ['hello ', 'world']
        assert call_kwargs['stream'] is True
        assert call_kwargs['stream_options'] == {'include_usage': True}
        assert usage_out['prompt_tokens'] == 12
        assert usage_out['completion_tokens'] == 8
        assert usage_out['total_tokens'] == 20
        assert usage_out['reasoning_tokens'] == 5
        assert usage_out['cached_tokens'] == 9
        assert stream.closed is True

    @pytest.mark.asyncio
    async def test_streaming_retries_without_stream_options_on_bad_request(self):
        """Providers that reject ``stream_options`` get a transparent retry without it,
        so streaming still works (usage capture is simply skipped for that provider)."""
        import httpx
        from openai import BadRequestError

        from yar.llm.openai import openai_complete_if_cache
        from yar.utils import TokenTracker

        rejection = BadRequestError(
            "Unsupported parameter: 'stream_options'",
            response=httpx.Response(400, request=httpx.Request('POST', 'https://api.test/v1')),
            body=None,
        )
        stream = _AsyncChunkStream([
            _create_stream_content_chunk('hello '),
            _create_stream_content_chunk('world'),
        ])
        calls = []

        def create_side_effect(*args, **kwargs):
            calls.append(kwargs)
            if 'stream_options' in kwargs:
                raise rejection
            return stream

        with patch('yar.llm.openai.create_openai_async_client') as mock_client_factory:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=create_side_effect)
            mock_client.close = AsyncMock()
            mock_client_factory.return_value = mock_client

            tracker = TokenTracker()
            result = await openai_complete_if_cache(
                model='gpt-4', prompt='Hi', api_key='k', token_tracker=tracker, stream=True,
            )
            chunks = [chunk async for chunk in result]

        assert chunks == ['hello ', 'world']
        assert len(calls) == 2
        assert 'stream_options' in calls[0]
        assert 'stream_options' not in calls[1]
        assert calls[1]['stream'] is True
        assert calls[1]['model'] == 'gpt-4'
        assert tracker.get_usage()['prompt_tokens'] == 0
