from __future__ import annotations

from typing import Any, ClassVar

from yar.evaluation import phoenix_experiments
from yar.evaluation.phoenix_experiments import ExperimentConfig, _build_task


class _FakeResponse:
    status_code = 200
    text = ''

    def __init__(self) -> None:
        self.headers: dict[str, str] = {}

    def json(self) -> dict[str, object]:
        return {'response': 'ok', 'references': []}


class _FakeClient:
    requests: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, *, timeout: int) -> None:
        self.timeout = timeout

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def post(self, url: str, *, json: dict[str, object], headers: dict[str, str]) -> _FakeResponse:
        self.requests.append({'url': url, 'json': json, 'headers': headers})
        return _FakeResponse()


def test_experiment_task_uses_server_retrieval_defaults(monkeypatch) -> None:
    _FakeClient.requests = []
    monkeypatch.setattr(phoenix_experiments.httpx, 'Client', _FakeClient)

    task = _build_task(ExperimentConfig(dataset_name='dataset'))
    result = task({'input': {'query': 'What changed?', 'mode': 'mix'}})

    assert result['response'] == 'ok'
    assert _FakeClient.requests[0]['json'] == {
        'query': 'What changed?',
        'mode': 'mix',
        'disable_cache': True,
    }


def test_experiment_task_applies_explicit_query_overrides(monkeypatch) -> None:
    _FakeClient.requests = []
    monkeypatch.setattr(phoenix_experiments.httpx, 'Client', _FakeClient)

    task = _build_task(
        ExperimentConfig(
            dataset_name='dataset',
            query_overrides={'top_k': 10, 'chunk_top_k': 5, 'enable_bm25_fusion': True},
        )
    )
    task({'input': {'query': 'What changed?', 'mode': 'mix'}})

    request_body = _FakeClient.requests[0]['json']
    assert isinstance(request_body, dict)
    assert request_body['top_k'] == 10
    assert request_body['chunk_top_k'] == 5
    assert request_body['enable_bm25_fusion'] is True
