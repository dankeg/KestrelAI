"""
SearXNG service for web search functionality.
Handles SearXNG setup, container management, and search execution.
"""

import logging
import os
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


def _check_searxng_accessible(url: str, timeout: int = 2) -> bool:
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
            ["docker", "info"], check=False, capture_output=True, timeout=5
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
            timeout=5,
        )
        if result.returncode == 0:
            return ["docker", "compose"]
    except Exception:
        pass

    # Fall back to older 'docker-compose' (Docker Compose V1)
    try:
        result = subprocess.run(
            ["docker-compose", "version"], check=False, capture_output=True, timeout=5
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
            timeout=60,
        )

        if result.returncode != 0:
            logger.warning(f"Failed to start SearchXNG container: {result.stderr}")
            if result.stdout:
                logger.debug(f"docker compose output: {result.stdout}")
            return

        # Wait for SearchXNG to become available (with retries)
        logger.info("Waiting for SearchXNG to be ready...")
        max_retries = 30  # Wait up to 30 seconds
        for i in range(max_retries):
            if _check_searxng_accessible(SEARXNG_URL, timeout=3):
                logger.info("SearchXNG container is ready")
                return
            time.sleep(1)

        logger.warning(
            "SearchXNG container started but not yet accessible after 30 seconds"
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

    def search(self, query: str) -> list[dict[str, Any]]:
        """Execute web search via SearXNG"""
        # Ensure SearchXNG is running when running locally
        ensure_searxng_running()

        params = {
            "q": query,
            "format": "json",
            "language": "en",
            "safesearch": 1,
            "engines": "google",
            "categories": "general",
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

        try:
            res = requests.get(
                self.searxng_url, params=params, headers=headers, timeout=15
            )
            res.raise_for_status()
            data = res.json()

            return [
                {
                    "title": r.get("title", "")[:100],
                    "href": r.get("url", ""),
                    "body": r.get("content", "")[:300],
                }
                for r in data.get("results", [])[: self.search_results]
                if r.get("url")
            ]
        except Exception as e:
            if self.debug:
                print(f"Search error: {e}")
            return []

    def extract_text(self, url: str) -> str:
        """Download page & return readable text (best-effort)."""
        try:
            # Clean URL before fetching
            clean_url_str = clean_url(url)
            if clean_url_str is None:
                return ""

            resp = requests.get(
                clean_url_str, timeout=10, headers={"User-Agent": "Mozilla/5.0"}
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
