"""Unit tests for LongParser LLM chain factory."""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core", reason="langchain_core not installed — run: pip install longparser[server]")

from longparser.server.chat.llm_chain import DEFAULT_MODELS, SUPPORTED_PROVIDERS, get_chat_model  # noqa: E402
from longparser.server.chat.schemas import ChatConfig  # noqa: E402


class TestDefaultModels:
    """Ensure all default model names are sane strings (not speculative names)."""

    def test_all_providers_have_defaults(self):
        for provider in SUPPORTED_PROVIDERS:
            assert provider in DEFAULT_MODELS, f"No default model for {provider!r}"

    def test_openai_default_is_gpt53(self):
        assert DEFAULT_MODELS["openai"] == "gpt-5.3"

    def test_gemini_default_exists(self):
        assert "gemini" in DEFAULT_MODELS["gemini"]

    def test_groq_default_is_gpt_oss(self):
        assert "gpt-oss" in DEFAULT_MODELS["groq"].lower()


class TestGetChatModelValidation:
    """Test provider validation without instantiating real LLM clients."""

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_chat_model(provider="anthropic_fake")

    def test_unknown_provider_lists_supported(self):
        with pytest.raises(ValueError) as exc_info:
            get_chat_model(provider="nonexistent")
        assert "openai" in str(exc_info.value)

    def test_supported_providers_list_complete(self):
        assert set(SUPPORTED_PROVIDERS) == {"openai", "gemini", "groq", "openrouter"}


class TestChatConfig:
    """Config-based provider/model resolution logic."""

    def test_config_provides_defaults(self):
        cfg = ChatConfig(llm_provider="openai", llm_model="gpt-4o-mini")
        assert cfg.llm_provider == "openai"
        assert cfg.llm_model == "gpt-4o-mini"

    def test_model_fallback_chain(self):
        """Provider default is used when config has no model."""
        cfg = ChatConfig(llm_provider="openai", llm_model="")
        resolved = cfg.llm_model or DEFAULT_MODELS.get("openai", "gpt-5.3")
        assert resolved == "gpt-5.3"
