"""Unit tests for LongParser token counting and budget trimmer."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from longparser.api.chat import budget_trim, count_tokens, validate_citations
from longparser.api.chat_schemas import LLMAnswer


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------

class TestCountTokens:
    def test_empty_string_is_zero_or_near(self):
        assert count_tokens("") == 0

    def test_short_text(self):
        tokens = count_tokens("Hello world")
        assert 1 <= tokens <= 5

    def test_longer_text_more_tokens(self):
        short = count_tokens("Hi")
        long = count_tokens("This is a much longer sentence with many more words in it.")
        assert long > short

    def test_unknown_model_fallback(self):
        # Should not raise — use heuristic fallback
        tokens = count_tokens("test sentence", model="some-unknown-model-xyz")
        assert tokens > 0


# ---------------------------------------------------------------------------
# validate_citations
# ---------------------------------------------------------------------------

class TestValidateCitations:
    def _doc(self, chunk_id: str) -> Document:
        return Document(page_content="content", metadata={"chunk_id": chunk_id})

    def test_valid_citation_kept(self):
        answer = LLMAnswer(answer="Answer.", cited_chunk_ids=["chunk-1"])
        docs = [self._doc("chunk-1"), self._doc("chunk-2")]
        result = validate_citations(answer, docs)
        assert "chunk-1" in result.cited_chunk_ids

    def test_invalid_citation_stripped(self):
        answer = LLMAnswer(answer="Answer.", cited_chunk_ids=["chunk-999"])
        docs = [self._doc("chunk-1")]
        result = validate_citations(answer, docs)
        assert result.cited_chunk_ids == []

    def test_all_stripped_becomes_no_info_message(self):
        answer = LLMAnswer(answer="My answer.", cited_chunk_ids=["bad-id"])
        docs = [self._doc("chunk-1")]
        result = validate_citations(answer, docs)
        assert "I don't have enough information" in result.answer

    def test_no_documents_no_change(self):
        answer = LLMAnswer(answer="Answer.", cited_chunk_ids=[])
        result = validate_citations(answer, [])
        assert result.answer == "Answer."


# ---------------------------------------------------------------------------
# budget_trim
# ---------------------------------------------------------------------------

class TestBudgetTrim:
    def _doc(self, text: str, chunk_id: str = "c1") -> Document:
        return Document(
            page_content=text,
            metadata={"chunk_id": chunk_id, "page_numbers": [1], "score": 0.9},
        )

    def test_returns_required_keys(self):
        result = budget_trim(
            question="What is Python?",
            documents=[self._doc("Python is a programming language.")],
            recent_turns=[],
            rolling_summary="",
            long_term_facts=[],
        )
        assert set(result.keys()) == {"question", "context", "history", "summary", "facts"}

    def test_question_always_included(self):
        result = budget_trim(
            question="My question",
            documents=[],
            recent_turns=[],
            rolling_summary="",
            long_term_facts=[],
        )
        assert result["question"] == "My question"

    def test_history_is_lc_messages(self):
        turns = [{"question": "Q1", "answer": "A1"}]
        result = budget_trim("Q?", [], turns, "", [])
        for msg in result["history"]:
            assert isinstance(msg, (HumanMessage, AIMessage))

    def test_tight_budget_drops_chunks(self):
        """With a very small budget, no chunks should be included."""
        big_doc = self._doc("word " * 500)  # ~500 tokens
        result = budget_trim(
            question="Q",
            documents=[big_doc],
            recent_turns=[],
            rolling_summary="",
            long_term_facts=[],
            max_prompt_tokens=5,  # tiny budget
        )
        assert result["context"] == "" or len(result["context"]) < 50

    def test_facts_none_when_empty(self):
        result = budget_trim("Q?", [], [], "", [])
        assert result["facts"] == "None"

    def test_summary_none_when_empty(self):
        result = budget_trim("Q?", [], [], "", [])
        assert result["summary"] == "None"

    def test_summary_included_when_fits(self):
        summary = "This is a rolling summary of the conversation."
        result = budget_trim("Q?", [], [], summary, [], max_prompt_tokens=2000)
        assert result["summary"] == summary
