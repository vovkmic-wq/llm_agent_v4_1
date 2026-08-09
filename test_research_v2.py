import pytest

from core.research import ResearchError, ResearchManager


def test_research_requires_enable_flag() -> None:
    manager = ResearchManager(enabled=False, allowed_domains=["example.com"])
    with pytest.raises(ResearchError):
        manager.fetch_url("https://example.com")


def test_research_blocks_non_allowlisted_domain_before_network() -> None:
    manager = ResearchManager(enabled=True, allowed_domains=["docs.python.org"])
    with pytest.raises(ResearchError, match="allow-list"):
        manager.fetch_url("https://example.com")


def test_research_blocks_http() -> None:
    manager = ResearchManager(enabled=True, allowed_domains=["example.com"])
    with pytest.raises(ResearchError, match="HTTPS"):
        manager.fetch_url("http://example.com")
