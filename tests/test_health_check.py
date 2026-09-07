import pytest

import health_check


@pytest.mark.asyncio
async def test_huggingface_health_requires_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    monkeypatch.delenv("HUGGING_FACE_TOKEN", raising=False)

    result = await health_check.HealthChecker()._check_huggingface_api()

    assert result["healthy"] is False
    assert result["details"]["token_configured"] is False
    assert "token_length" not in result["details"]


@pytest.mark.asyncio
async def test_huggingface_health_accepts_successful_authenticated_response(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")

    class FakeResponse:
        status_code = 200

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, *args, **kwargs):
            return FakeResponse()

    class FakeHttpx:
        Timeout = lambda *args, **kwargs: object()
        AsyncClient = FakeClient

    monkeypatch.setitem(__import__("sys").modules, "httpx", FakeHttpx)

    result = await health_check.HealthChecker()._check_huggingface_api()

    assert result["healthy"] is True
    assert result["details"]["authenticated"] is True
    assert result["details"]["status_code"] == 200


@pytest.mark.asyncio
async def test_huggingface_health_does_not_leak_request_exception(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "super-secret-token")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, *args, **kwargs):
            raise RuntimeError("super-secret-token should never be returned")

    class FakeHttpx:
        Timeout = lambda *args, **kwargs: object()
        AsyncClient = FakeClient

    monkeypatch.setitem(__import__("sys").modules, "httpx", FakeHttpx)

    result = await health_check.HealthChecker()._check_huggingface_api()

    assert result["healthy"] is False
    assert result["details"]["error"] == "Hugging Face API request failed"
    assert "super-secret-token" not in str(result)
