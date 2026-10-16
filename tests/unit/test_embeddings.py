"""Embedding configuration, batching, dimension discovery and cache behavior."""

from __future__ import annotations

import builtins
import re
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from longparser.server.embeddings import EmbeddingEngine


@pytest.fixture
def providers(monkeypatch):
    model = Mock()
    model.embed_documents.side_effect = lambda texts, **kwargs: [[1.0, 0.0, 0.5] for _ in texts]
    model.embed_query.return_value = [0.0, 1.0, 0.5]
    constructors = {}
    for module_name, class_name in (
        ("langchain_openai", "OpenAIEmbeddings"),
        ("langchain_google_genai", "GoogleGenerativeAIEmbeddings"),
        ("langchain_huggingface", "HuggingFaceEmbeddings"),
    ):
        module = ModuleType(module_name)
        constructor = Mock(return_value=model)
        setattr(module, class_name, constructor)
        monkeypatch.setitem(sys.modules, module_name, module)
        constructors[module_name] = constructor
    cache = Mock()
    cache.get.return_value = None
    redis = ModuleType("redis")
    redis.from_url = Mock(return_value=cache)
    monkeypatch.setitem(sys.modules, "redis", redis)
    return model, constructors, cache


@pytest.mark.parametrize("provider", ["openai", "gemini", "huggingface"])
@pytest.mark.parametrize("dimensions", [None, 128])
def test_provider_configuration_and_explicit_dimensions(providers, provider, dimensions):
    model, constructors, _ = providers
    engine = EmbeddingEngine(provider.upper(), "example-model", dimensions)
    module = {"openai": "langchain_openai", "gemini": "langchain_google_genai",
              "huggingface": "langchain_huggingface"}[provider]
    if provider == "huggingface":
        constructors[module].assert_called_once_with(
            model_name="example-model", encode_kwargs={"normalize_embeddings": True})
    else:
        option = "dimensions" if provider == "openai" else "output_dimensionality"
        constructors[module].assert_called_once_with(
            model="example-model", **({option: dimensions} if dimensions else {}))
    assert engine.embed_chunks([]) == []
    model.embed_documents.assert_not_called()
    if dimensions:
        assert engine.dim == 128
        model.embed_documents.assert_not_called()


def test_unknown_embedding_provider_rejected():
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        EmbeddingEngine("invalid")


@pytest.mark.parametrize("provider", ["openai", "gemini", "huggingface"])
def test_fingerprint_stable_and_configuration_spaces_isolated(providers, provider):
    same = EmbeddingEngine(provider, "model", 128).get_fingerprint()
    assert re.fullmatch("[0-9a-f]{10}", same)
    assert same == EmbeddingEngine(provider.upper(), "model", 128).get_fingerprint()
    assert same != EmbeddingEngine(provider, "another-model", 128).get_fingerprint()
    assert same != EmbeddingEngine(provider, "model", 64).get_fingerprint()


@pytest.mark.parametrize("provider", ["openai", "gemini", "huggingface"])
def test_embedding_queries_and_documents_use_expected_tasks(providers, provider):
    model, _, _ = providers
    engine = EmbeddingEngine(provider, "model")
    assert engine.embed_chunks(["first", "second"]) == [[1.0, 0.0, 0.5]] * 2
    assert engine.embed_query("question") == [0.0, 1.0, 0.5]
    task = {"task_type": "RETRIEVAL_DOCUMENT"} if provider == "gemini" else {}
    model.embed_documents.assert_called_once_with(["first", "second"], **task)
    query_task = {"task_type": "RETRIEVAL_QUERY"} if provider == "gemini" else {}
    model.embed_query.assert_called_once_with("question", **query_task)


def test_gemini_caps_batches_and_preserves_input_order(providers):
    model, _, _ = providers
    texts = [f"document-{index}" for index in range(205)]
    model.embed_documents.side_effect = lambda batch, **kwargs: [[int(text.split('-')[1])] for text in batch]
    assert EmbeddingEngine("gemini", "model").embed_chunks(texts, batch_size=250) == [[i] for i in range(205)]
    assert [len(call.args[0]) for call in model.embed_documents.call_args_list] == [100, 100, 5]


def test_dimensions_reuse_cross_process_cache_without_provider_call(providers):
    model, _, cache = providers
    cache.get.return_value = b"384"
    engine = EmbeddingEngine("openai", "model")
    assert engine.dim == 384
    assert engine.dim == 384
    cache.ping.assert_called_once()
    cache.get.assert_called_once_with("longparser:embed_dim:" + engine.get_fingerprint())
    model.embed_documents.assert_not_called()


@pytest.mark.parametrize("provider", ["openai", "gemini"])
@pytest.mark.parametrize("cache_failure", ["none", "ping", "get", "set"])
def test_dimension_discovery_tolerates_cache_failures(providers, provider, cache_failure):
    model, _, cache = providers
    if cache_failure != "none":
        getattr(cache, cache_failure).side_effect = OSError("cache unavailable")
    engine = EmbeddingEngine(provider, "model")
    assert engine.dim == 3
    assert engine.dim == 3
    kwargs = {"task_type": "RETRIEVAL_DOCUMENT"} if provider == "gemini" else {}
    model.embed_documents.assert_called_once_with(["test"], **kwargs)


def test_missing_redis_uses_provider_dimension(providers, monkeypatch):
    real_import = builtins.__import__

    def without_redis(name, *args, **kwargs):
        if name == "redis":
            raise ImportError("redis missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_redis)
    assert EmbeddingEngine("openai", "model").dim == 3


def test_dimension_provider_error_is_reported_instead_of_cached(providers):
    model, _, cache = providers
    model.embed_documents.side_effect = RuntimeError("provider denied request")
    engine = EmbeddingEngine("openai", "model")
    with pytest.raises(RuntimeError, match="denied request"):
        _ = engine.dim
    cache.set.assert_not_called()
