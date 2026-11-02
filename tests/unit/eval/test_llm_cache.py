"""Cache odpowiedzi LLM-a na dysku (ADR-0008 pkt 6; kontrakt F2, 8.4)."""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration


def _message(text="odpowiedź") -> AIMessage:
    return AIMessage(
        content=text,
        usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
        response_metadata={"model_name": "fake"},
    )


def test_cache_roundtrip_skips_model(tmp_path):
    from kovalo.eval.llm_cache import JsonFileLlmCache

    cache = JsonFileLlmCache(tmp_path / "llm")
    model = GenericFakeChatModel(messages=iter([_message()]), cache=cache)
    first = model.invoke("to samo pytanie")
    assert (cache.hits, cache.misses) == (0, 1)
    second = model.invoke("to samo pytanie")  # drugi raz iterator byłby pusty: StopIteration
    assert second.content == first.content == "odpowiedź"
    assert cache.hits == 1
    assert second.usage_metadata["input_tokens"] == 11
    assert second.usage_metadata["output_tokens"] == 7
    assert list((tmp_path / "llm").rglob("*.json"))  # <dir>/<key[:2]>/<key>.json

    # nowa instancja (nowy proces) czyta to samo z dysku
    fresh = JsonFileLlmCache(tmp_path / "llm")
    again = GenericFakeChatModel(messages=iter([]), cache=fresh).invoke("to samo pytanie")
    assert again.content == "odpowiedź" and fresh.hits == 1


def test_different_prompt_is_a_miss(tmp_path):
    from kovalo.eval.llm_cache import JsonFileLlmCache

    cache = JsonFileLlmCache(tmp_path / "llm")
    model = GenericFakeChatModel(messages=iter([_message("a"), _message("b")]), cache=cache)
    assert model.invoke("pytanie 1").content == "a"
    assert model.invoke("pytanie 2").content == "b"
    assert cache.hits == 0 and cache.misses == 2


def test_read_only_and_strict_modes(tmp_path):
    from kovalo.eval.llm_cache import JsonFileLlmCache, LlmCacheMiss

    directory = tmp_path / "llm"
    read_only = JsonFileLlmCache(directory, mode="read_only")
    GenericFakeChatModel(messages=iter([_message()]), cache=read_only).invoke("p")
    assert not directory.exists() or not list(directory.rglob("*.json"))

    strict = JsonFileLlmCache(directory, mode="strict")
    model = GenericFakeChatModel(messages=iter([_message()]), cache=strict)
    with pytest.raises(LlmCacheMiss):
        model.invoke("p")

    # strict czyta to, co zapisał read_write
    GenericFakeChatModel(messages=iter([_message()]), cache=JsonFileLlmCache(directory)).invoke("p")
    hit = GenericFakeChatModel(messages=iter([]), cache=JsonFileLlmCache(directory, mode="strict"))
    assert hit.invoke("p").content == "odpowiedź"


def test_miss_error_is_not_a_fallback_exception():
    from kovalo.eval.llm_cache import LlmCacheMiss
    from kovalo.llm import FALLBACK_EXCEPTIONS

    assert issubclass(LlmCacheMiss, RuntimeError)
    assert not issubclass(LlmCacheMiss, FALLBACK_EXCEPTIONS)


def test_cache_files_have_no_secret(tmp_path, tmp_config):
    from kovalo.config import load_settings
    from kovalo.eval.llm_cache import JsonFileLlmCache
    from kovalo.llm import get_chat_model

    secret = "AIza-super-tajny-klucz-123"
    real = get_chat_model(settings=load_settings(tmp_config, google_api_key=secret))
    cache = JsonFileLlmCache(tmp_path / "llm")
    cache.update("prompt", real._get_llm_string(), [ChatGeneration(message=_message())])
    files = [p for p in (tmp_path / "llm").rglob("*") if p.is_file()]
    assert files and all(secret not in p.read_text(encoding="utf-8") for p in files)
    assert not [p for p in files if p.suffix == ".tmp" or ".tmp" in p.name]  # zapis atomowy
    assert cache.lookup("prompt", real._get_llm_string()) is not None


def test_clear_removes_files(tmp_path):
    from kovalo.eval.llm_cache import JsonFileLlmCache

    cache = JsonFileLlmCache(tmp_path / "llm")
    cache.update("p", "llm", [ChatGeneration(message=_message())])
    assert cache.lookup("p", "llm") is not None
    cache.clear()
    assert cache.lookup("p", "llm") is None
