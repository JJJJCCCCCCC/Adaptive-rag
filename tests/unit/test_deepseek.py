"""DeepSeek transport configuration regression tests."""

import pytest

from app.config.settings import Settings
from app.providers.deepseek import DeepSeekProvider


async def test_deepseek_can_ignore_invalid_inherited_proxy(monkeypatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "unsupported://proxy.invalid:7892")
    settings = Settings(_env_file=None, deepseek_trust_env=False)

    provider = DeepSeekProvider(settings)
    await provider.aclose()


async def test_deepseek_respects_inherited_proxy_by_default(monkeypatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "unsupported://proxy.invalid:7892")
    settings = Settings(_env_file=None)

    with pytest.raises(ValueError, match="Unknown scheme"):
        DeepSeekProvider(settings)
