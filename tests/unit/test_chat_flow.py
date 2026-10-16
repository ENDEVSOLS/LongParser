"""Actual retrieval/chat/HITL workflows with controlled provider/storage SDKs."""

from __future__ import annotations

import logging
import socket
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver

from longparser.server.chat import checkpointer
from longparser.server.chat.callbacks import LongParserCallbackHandler
from longparser.server.chat.engine import ChatEngine
from longparser.server.chat.graph import resume_hitl_review, start_hitl_review
from longparser.server.chat.retriever import LongParserRetriever
from longparser.server.chat.schemas import ChatConfig, ChatRequest, LLMAnswer, SourceRef
from longparser.server.queue import ARQBackend


@pytest.fixture
def chat_sdks(monkeypatch):
    monkeypatch.setenv("LONGPARSER_LLM_PROVIDER", "openai")
    monkeypatch.setenv("LONGPARSER_LLM_MODEL", "test-model")

    def deny_network(*args, **kwargs):
        raise AssertionError("Chat regression tests must not contact external services")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    llm = SimpleNamespace(ainvoke=AsyncMock(return_value={"answer": "Thirty days.", "cited_chunk_ids": ["valid", "invented"]}))
    llm.with_structured_output = Mock(return_value=llm)
    provider = ModuleType("langchain_openai")
    provider.ChatOpenAI = Mock(return_value=llm)
    monkeypatch.setitem(sys.modules, "langchain_openai", provider)
    embeddings = ModuleType("langchain_huggingface")
    embedding_model = Mock()
    embedding_model.embed_query.return_value = [1.0, 0.0]
    embeddings.HuggingFaceEmbeddings = Mock(return_value=embedding_model)
    monkeypatch.setitem(sys.modules, "langchain_huggingface", embeddings)
    collection = Mock()
    collection.query.return_value = {"ids": [["vector"]], "metadatas": [[{"chunk_id": "valid", "page_numbers": "[2]", "block_ids": "[\"block\"]", "chunk_type": "section"}]], "documents": [["Return unused items within thirty days."]], "distances": [[0.1]]}
    chroma = ModuleType("chromadb")
    chroma.PersistentClient = Mock(return_value=SimpleNamespace(get_or_create_collection=Mock(return_value=collection)))
    monkeypatch.setitem(sys.modules, "chromadb", chroma)
    import tiktoken

    monkeypatch.setattr(tiktoken, "encoding_for_model", lambda model: SimpleNamespace(encode=lambda text: text.split()))
    import arq

    pool = SimpleNamespace(enqueue_job=AsyncMock(return_value=SimpleNamespace(job_id="task")))
    monkeypatch.setattr(arq, "create_pool", AsyncMock(return_value=pool))
    return llm, collection, embedding_model, pool


async def seed_index(database):
    await database.create_index_version("tenant", "job", "index", {"status": "indexed", "model": "model", "provider": "huggingface", "vector_db": "chroma", "collection": "chunks", "configured_dimensions": 2})


async def test_chat_retrieves_tenant_context_validates_citations_and_saves_turn(memory_database, chat_sdks):
    llm, collection, embeddings, pool = chat_sdks
    await seed_index(memory_database)
    await memory_database.create_chat_session("tenant", "session", "job")
    config = ChatConfig(summarize_every=1, extract_facts_every=1)
    engine = ChatEngine(memory_database, ARQBackend(), config)
    request = ChatRequest(session_id="session", job_id="job", question="When can I return it?", idempotency_key="once", top_k=99)
    response = await engine.ask("tenant", request)
    assert response.answer == "Thirty days." and response.status == "complete"
    assert [source.chunk_id for source in response.sources] == ["valid"]
    assert response.sources[0].page_numbers == [2]
    assert collection.query.call_args.kwargs["where"] == {"$and": [{"tenant_id": {"$eq": "tenant"}}, {"job_id": {"$eq": "job"}}]}
    assert collection.query.call_args.kwargs["n_results"] == config.max_top_k
    embeddings.embed_query.assert_called_once_with(request.question)
    assert any("Return unused items" in message.content for message in llm.ainvoke.call_args.args[0])
    saved = await memory_database.get_recent_turns("tenant", "session")
    assert len(saved) == 1 and saved[0]["answer"] == "Thirty days."
    assert [call.args[0] for call in pool.enqueue_job.await_args_list] == ["summarize_session", "extract_facts"]
    again = await engine.ask("tenant", request)
    assert again.turn_id == response.turn_id
    assert llm.ainvoke.await_count == 1
    await engine.close()


async def test_chat_handles_missing_session_and_all_invalid_citations(memory_database, chat_sdks):
    llm, _, _, pool = chat_sdks
    await seed_index(memory_database)
    llm.ainvoke.return_value = LLMAnswer(answer="unsupported", cited_chunk_ids=["invented"])
    response = await ChatEngine(memory_database, ARQBackend()).ask("tenant", ChatRequest(session_id="missing", job_id="job", question="policy?"))
    assert "don't have enough information" in response.answer
    assert response.sources == []
    pool.enqueue_job.assert_not_awaited()


async def test_chat_rejects_oversized_question_before_retrieval(memory_database, chat_sdks):
    llm, collection, _, _ = chat_sdks
    response = await ChatEngine(memory_database, ARQBackend(), ChatConfig(max_input_tokens=2)).ask("tenant", ChatRequest(session_id="session", job_id="job", question="one two three"))
    assert "Question too long" in response.answer and response.turn_id == ""
    llm.ainvoke.assert_not_awaited()
    collection.query.assert_not_called()


async def test_retriever_index_cache_and_missing_index_error(memory_database, chat_sdks):
    await seed_index(memory_database)
    retriever = LongParserRetriever(db=memory_database, tenant_id="tenant", job_id="job")
    assert (await retriever.ainvoke("policy"))[0].metadata["block_ids"] == ["block"]
    assert (await retriever.ainvoke("again"))[0].page_content.startswith("Return unused")
    assert sum(call[0] == "find" for call in memory_database.index_versions.calls) == 1
    with pytest.raises(ValueError, match="No embedding index"):
        await LongParserRetriever(db=memory_database, tenant_id="other", job_id="job").ainvoke("private")


@pytest.fixture
def graph_storage(monkeypatch):
    client = Mock()
    monkeypatch.setattr(checkpointer, "_mongo_client", None)
    monkeypatch.setattr(checkpointer, "_checkpointer", None)
    monkeypatch.setattr(checkpointer, "MongoClient", Mock(return_value=client))
    saver = InMemorySaver()
    monkeypatch.setattr(checkpointer, "MongoDBSaver", Mock(return_value=saver))
    return client, saver


@pytest.mark.parametrize("action,edited,status,answer", [("approve", None, "complete", "draft"), ("edit", "corrected", "complete", "corrected"), ("reject", None, "rejected", "Answer rejected by reviewer."), ("unknown", None, "complete", "draft")])
async def test_real_review_graph_interrupts_and_resumes(graph_storage, action, edited, status, answer):
    client, saver = graph_storage
    with pytest.raises(RuntimeError, match="not initialized"):
        checkpointer.get_checkpointer()
    await checkpointer.init_checkpointer("mongodb://memory.test", "reviews")
    await checkpointer.init_checkpointer("mongodb://ignored.test", "ignored")
    checkpointer.MongoClient.assert_called_once_with("mongodb://memory.test")
    assert checkpointer.get_checkpointer() is saver
    pending = await start_hitl_review("tenant", "session", "job", "question", LLMAnswer(answer="draft", cited_chunk_ids=["valid"]), [SourceRef(chunk_id="valid", score=0.9, text="context")])
    assert pending["status"] == "pending_review" and pending["draft_answer"] == "draft"
    result = await resume_hitl_review(pending["thread_id"], action, edited)
    assert result["status"] == status and result["answer"] == answer
    assert result["cited_chunk_ids"] == ([] if action == "reject" else ["valid"])
    await checkpointer.close_checkpointer()
    client.close.assert_called_once()
    await checkpointer.close_checkpointer()


def test_callbacks_log_latency_tokens_errors_and_retrieval_scores(monkeypatch, caplog):
    import longparser.server.chat.callbacks as callbacks

    monkeypatch.setattr(callbacks.time, "monotonic", Mock(side_effect=[10.0, 10.125]))
    handler = LongParserCallbackHandler("tenant", "session")
    run_id = uuid4()
    with caplog.at_level(logging.INFO):
        handler.on_llm_start({"kwargs": {"model_name": "model"}}, ["prompt"], run_id=run_id)
        handler.on_llm_end(SimpleNamespace(llm_output={"token_usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}}), run_id=run_id)
        handler.on_llm_error(RuntimeError("provider denied"), run_id=run_id)
        handler.on_retriever_end([Document(page_content="a", metadata={"score": 0.6}), Document(page_content="b", metadata={"score": 0.8})], run_id=run_id)
        LongParserCallbackHandler().on_llm_end(SimpleNamespace(llm_output=None), run_id=run_id)
        handler.on_retriever_end([], run_id=run_id)
    finished = next(record for record in caplog.records if record.message == "llm_call_end")
    assert finished.latency_ms == 125 and finished.total_tokens == 6
    results = next(record for record in caplog.records if record.message == "retriever_results")
    assert results.top_score == 0.8 and results.avg_score == 0.7
    assert any(record.message == "llm_call_error" for record in caplog.records)


@pytest.mark.parametrize("provider,sdk_name,symbol,token_option", [
    ("gemini", "langchain_google_genai", "ChatGoogleGenerativeAI", "max_output_tokens"),
    ("groq", "langchain_groq", "ChatGroq", "max_tokens"),
    ("openrouter", "langchain_openai", "ChatOpenAI", "max_tokens"),
])
def test_chat_provider_configuration_uses_correct_sdk(provider, sdk_name, symbol, token_option, monkeypatch):
    from longparser.server.chat.llm_chain import get_chat_model
    sdk = ModuleType(sdk_name)
    factory = Mock(return_value=Mock())
    setattr(sdk, symbol, factory)
    monkeypatch.setitem(sys.modules, sdk_name, sdk)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-token")
    model = get_chat_model(provider=provider, model="selected-model", max_tokens=25, temperature=.2)
    assert model is factory.return_value
    assert factory.call_args.kwargs["model"] == "selected-model"
    assert factory.call_args.kwargs[token_option] == 25
    assert factory.call_args.kwargs["temperature"] == .2
    if provider == "openrouter":
        assert factory.call_args.kwargs["base_url"] == "https://openrouter.ai/api/v1"
        assert factory.call_args.kwargs["api_key"] == "fixture-token"
