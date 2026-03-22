"""
SearXNG service for web search functionality.
Handles SearXNG setup, container management, and search execution.
"""

import logging
import os
import re
import subprocess
import time
from typing import Any

import requests
from bs4 import BeautifulSoup

from .url_utils import clean_url

logger = logging.getLogger(__name__)

# Configuration
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080/search")
FETCH_BYTES = 30_000
MAX_SNIPPET_LENGTH = 3000
DEBUG = True


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
        return max(0.1, value)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_csv(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in str(raw).split(",") if item.strip()]


# Track if we've already checked/started SearchXNG
_searxng_checked = False


def _is_running_in_docker() -> bool:
    """Check if we're running inside a Docker container"""
    # Check for Docker-specific environment variables or files
    if os.path.exists("/.dockerenv"):
        return True
    if os.path.exists("/proc/self/cgroup"):
        try:
            with open("/proc/self/cgroup") as f:
                if "docker" in f.read():
                    return True
        except Exception:
            pass
    if os.getenv("container") == "docker":
        return True
    return False


def _check_searxng_accessible(url: str, timeout: int = 20) -> bool:
    """Check if SearchXNG is accessible at the given URL"""
    try:
        test_url = url.replace("/search", "")  # Try base URL first
        res = requests.get(test_url, timeout=timeout)
        return res.status_code == 200
    except Exception:
        return False


def _check_docker_running() -> bool:
    """Check if Docker daemon is running"""
    try:
        result = subprocess.run(
            ["docker", "info"], check=False, capture_output=True, timeout=50
        )
        return result.returncode == 0
    except Exception:
        return False


def _get_docker_compose_cmd():
    """Get the appropriate docker compose command (handles both 'docker compose' and 'docker-compose')"""
    # Try newer 'docker compose' first (Docker Compose V2)
    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            check=False,
            capture_output=True,
            timeout=50,
        )
        if result.returncode == 0:
            return ["docker", "compose"]
    except Exception:
        pass

    # Fall back to older 'docker-compose' (Docker Compose V1)
    try:
        result = subprocess.run(
            ["docker-compose", "version"], check=False, capture_output=True, timeout=50
        )
        if result.returncode == 0:
            return ["docker-compose"]
    except Exception:
        pass

    return None


def ensure_searxng_running():
    """Ensure SearchXNG container is running when running locally (not in Docker)"""
    global _searxng_checked
    if _searxng_checked:
        return  # Already checked

    _searxng_checked = True

    # Check if we're running in Docker - if so, SearchXNG should already be available
    if _is_running_in_docker():
        return  # Already running in Docker, no action needed

    # Also skip if SEARXNG_URL points to a Docker service name (Docker networking)
    if "searxng:" in SEARXNG_URL or not SEARXNG_URL.startswith("http://localhost"):
        return  # Not a localhost URL, assume it's configured correctly

    # Check if SearchXNG is already accessible
    if _check_searxng_accessible(SEARXNG_URL):
        return  # SearchXNG is already running

    # Check if Docker is running
    if not _check_docker_running():
        logger.warning("Docker daemon is not running, cannot auto-start SearchXNG")
        return

    # Get docker compose command
    compose_cmd = _get_docker_compose_cmd()
    if not compose_cmd:
        logger.warning("docker compose command not found, cannot auto-start SearchXNG")
        return

    # If not accessible, try to start it via docker compose
    try:
        # Find docker-compose.yml in the project root (go up from KestrelAI/agents/)
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        compose_file = os.path.join(project_root, "docker-compose.yml")

        if not os.path.exists(compose_file):
            logger.warning(
                f"docker-compose.yml not found at {compose_file}, cannot auto-start SearchXNG"
            )
            return

        logger.info(
            "Starting SearchXNG container for local development (this will also start Redis if needed)..."
        )

        # Run from the project root directory to ensure relative paths work
        cmd = compose_cmd + ["-f", compose_file, "up", "-d", "searxng"]
        result = subprocess.run(
            cmd,
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            logger.warning(f"Failed to start SearchXNG container: {result.stderr}")
            if result.stdout:
                logger.debug(f"docker compose output: {result.stdout}")
            return

        # Wait for SearchXNG to become available (with retries)
        logger.info("Waiting for SearchXNG to be ready...")
        max_retries = 300  # Wait up to 300 seconds
        for i in range(max_retries):
            if _check_searxng_accessible(SEARXNG_URL, timeout=30):
                logger.info("SearchXNG container is ready")
                return
            time.sleep(1)

        logger.warning(
            "SearchXNG container started but not yet accessible after 300 seconds"
        )
    except subprocess.TimeoutExpired:
        logger.warning("Timeout starting SearchXNG container")
    except FileNotFoundError:
        logger.warning("docker compose command not found, cannot auto-start SearchXNG")
    except Exception as e:
        logger.warning(f"Could not start SearchXNG container: {e}")


class SearXNGService:
    """Service for executing web searches via SearXNG"""

    def __init__(
        self, searxng_url: str = None, search_results: int = 4, debug: bool = True
    ):
        self.searxng_url = searxng_url or SEARXNG_URL
        self.search_results = search_results
        self.debug = debug
        self.search_connect_timeout_seconds = _env_float(
            "SEARXNG_CONNECT_TIMEOUT_SECONDS",
            5.0,
        )
        self.search_read_timeout_seconds = _env_float(
            "SEARXNG_READ_TIMEOUT_SECONDS",
            12.0,
        )
        self.fetch_connect_timeout_seconds = _env_float(
            "PAGE_FETCH_CONNECT_TIMEOUT_SECONDS",
            4.0,
        )
        self.fetch_read_timeout_seconds = _env_float(
            "PAGE_FETCH_READ_TIMEOUT_SECONDS",
            8.0,
        )
        self.search_language = os.getenv("SEARXNG_LANGUAGE", "en").strip() or "en"
        self.search_categories = (
            os.getenv("SEARXNG_CATEGORIES", "general").strip() or "general"
        )
        self.search_safesearch = max(0, min(2, _env_int("SEARXNG_SAFESEARCH", 1)))
        self.search_engines = _env_csv(
            "SEARXNG_ENGINES",
            "bing,yahoo,arxiv,pubmed,mwmbl",
        )
        self.search_retry_without_engines = os.getenv(
            "SEARXNG_RETRY_WITHOUT_ENGINES", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}
        self.min_query_term_overlap = max(
            0,
            _env_int("SEARXNG_MIN_QUERY_TERM_OVERLAP", 2),
        )

    @staticmethod
    def _search_headers() -> dict[str, str]:
        return {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {
            tok
            for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9._/-]*", (text or "").lower())
            if len(tok) >= 3 and not tok.isdigit()
        }

    def _normalize_results(
        self, data: dict[str, Any], query: str
    ) -> list[dict[str, Any]]:
        normalized = [
            {
                "title": r.get("title", "")[:100],
                "href": r.get("url", ""),
                "body": r.get("content", "")[:300],
            }
            for r in data.get("results", [])[: self.search_results * 2]
            if r.get("url")
        ]

        query_terms = self._tokenize(query)
        if not query_terms or self.min_query_term_overlap <= 0:
            return normalized[: self.search_results]

        required_overlap = min(self.min_query_term_overlap, len(query_terms))
        filtered: list[dict[str, Any]] = []
        for item in normalized:
            text_terms = self._tokenize(
                f"{item.get('title', '')} {item.get('body', '')} {item.get('href', '')}"
            )
            if len(query_terms.intersection(text_terms)) >= required_overlap:
                filtered.append(item)

        # Prefer relevance-filtered hits, but never return empty if the upstream
        # engine did provide data.
        if filtered:
            return filtered[: self.search_results]
        return normalized[: self.search_results]

    def search(self, query: str) -> list[dict[str, Any]]:
        """Execute web search via SearXNG"""
        # Ensure SearchXNG is running when running locally
        ensure_searxng_running()

        base_params = {
            "q": query,
            "format": "json",
            "language": self.search_language,
            "safesearch": self.search_safesearch,
            "categories": self.search_categories,
        }
        attempt_engines: list[str | None] = []
        configured_engines = ",".join(self.search_engines)
        if configured_engines:
            # First ask SearXNG to blend a robust mix of engines, then isolate if needed.
            attempt_engines.append(configured_engines)
            attempt_engines.extend(self.search_engines)
        if self.search_retry_without_engines:
            attempt_engines.append(None)

        seen_urls: set[str] = set()
        for engines in attempt_engines:
            params = dict(base_params)
            if engines:
                params["engines"] = engines
            try:
                res = requests.get(
                    self.searxng_url,
                    params=params,
                    headers=self._search_headers(),
                    timeout=(
                        self.search_connect_timeout_seconds,
                        self.search_read_timeout_seconds,
                    ),
                )
                res.raise_for_status()
                data = res.json()
                normalized = self._normalize_results(data, query)

                deduped: list[dict[str, Any]] = []
                for item in normalized:
                    href = item.get("href", "")
                    if href and href not in seen_urls:
                        deduped.append(item)
                        seen_urls.add(href)

                if deduped:
                    return deduped

                if self.debug:
                    logger.debug(
                        "SearXNG returned no results for query '%s' (engines=%s, unresponsive=%s)",
                        query,
                        engines or "auto",
                        data.get("unresponsive_engines", []),
                    )
            except Exception as e:
                if self.debug:
                    logger.warning(
                        "SearXNG search attempt failed for query '%s' (engines=%s): %s",
                        query,
                        engines or "auto",
                        e,
                    )
        return []

    def extract_text(self, url: str) -> str:
        """Download page & return readable text (best-effort)."""
        try:
            # Clean URL before fetching
            clean_url_str = clean_url(url)
            if clean_url_str is None:
                return ""

            resp = requests.get(
                clean_url_str,
                timeout=(
                    self.fetch_connect_timeout_seconds,
                    self.fetch_read_timeout_seconds,
                ),
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            html = resp.text[:FETCH_BYTES]
            soup = BeautifulSoup(html, "html.parser")

            # Remove non-content elements
            for t in soup(["script", "style", "noscript", "meta", "link"]):
                t.extract()

            text = " ".join(soup.get_text(" ", strip=True).split())
            return text[:MAX_SNIPPET_LENGTH]
        except Exception as e:
            if self.debug:
                print(f"Failed to extract from {url}: {e}")
            return ""
