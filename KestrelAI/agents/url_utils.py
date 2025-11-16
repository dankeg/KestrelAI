"""
URL utilities for web research agent.
Handles URL validation, cleaning, and flag management.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)


def is_valid_url(url: str) -> bool:
    """Check if a URL is valid and complete."""
    if not url or len(url) < 10:  # Minimum reasonable URL length
        return False

    try:
        parsed = urlparse(url)
        # Must have scheme and netloc
        if not parsed.scheme or not parsed.netloc:
            return False

        # Remove port if present for validation
        netloc_for_validation = parsed.netloc.split(":")[0]

        # Check if it's an IP address (e.g., "192.168.1.1")
        if re.match(r"^(\d{1,3}\.){3}\d{1,3}$", netloc_for_validation):
            return True

        # For domain names, must have at least domain.tld
        # Reject incomplete domains like "www", "deepmind", "grow" without TLD
        netloc_parts = netloc_for_validation.split(".")
        if len(netloc_parts) < 2:
            # No TLD found (e.g., "deepmind" without ".com")
            return False

        return True
    except Exception:
        return False


def clean_url(url: str) -> str | None:
    """Clean and validate a URL, fixing common issues.
    Returns None if URL is invalid/incomplete."""
    if not url:
        return None

    # Remove leading/trailing whitespace
    url = url.strip()

    # Remove common trailing punctuation that shouldn't be part of URLs
    # But preserve punctuation that's valid in URLs (like ? and &)
    url = re.sub(r"[.,;:!?]+$", "", url)

    # Fix URLs broken across lines (remove newlines and extra spaces)
    url = re.sub(r"\s+", "", url)

    # Ensure URL starts with http:// or https://
    if not url.startswith(("http://", "https://")):
        # Try to fix common issues
        if url.startswith("www."):
            url = "https://" + url
        elif "://" in url:
            # Has protocol but might be malformed
            parts = url.split("://", 1)
            if len(parts) == 2:
                url = parts[0].lower() + "://" + parts[1]
        else:
            # No protocol, assume https
            url = "https://" + url

    # Validate URL structure
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            # Invalid URL - no netloc
            return None

        # Check if netloc is complete (has domain and TLD)
        netloc_parts = parsed.netloc.split(".")
        # Remove port if present
        if ":" in netloc_parts[-1]:
            netloc_parts[-1] = netloc_parts[-1].split(":")[0]

        # Reject incomplete domains (must have at least domain.tld or be an IP)
        if len(netloc_parts) < 2:
            # Check if it's an IP address
            if not re.match(r"^(\d{1,3}\.){3}\d{1,3}$", parsed.netloc.split(":")[0]):
                # Not an IP and no TLD - invalid
                return None

        # Reconstruct URL with cleaned components
        cleaned = urlunparse(
            (
                parsed.scheme,
                parsed.netloc.lower(),  # Normalize domain to lowercase
                parsed.path,
                parsed.params,
                parsed.query,
                "",  # Remove fragment
            )
        )

        # Final validation
        if not is_valid_url(cleaned):
            return None

        return cleaned
    except Exception:
        # If parsing fails, return None
        return None


class URLFlagManager:
    """Manages URL flags to prevent URL corruption during LLM generation."""

    def __init__(self):
        self.url_to_flag: dict[str, str] = {}  # URL -> flag mapping
        self.flag_to_url: dict[str, str] = {}  # flag -> URL mapping
        self.counter = 0

    def get_or_create_flag(self, url: str) -> str | None:
        """Get existing flag for URL or create a new one.
        Returns None if URL is invalid/incomplete."""
        # Clean URL first
        clean_url_str = clean_url(url)

        if clean_url_str is None:
            # Invalid URL - log and return None
            logger.warning(f"Invalid/incomplete URL rejected: {url[:100]}")
            return None

        if clean_url_str in self.url_to_flag:
            return self.url_to_flag[clean_url_str]

        # Create new flag
        self.counter += 1
        flag = f"[URL_{self.counter}]"
        self.url_to_flag[clean_url_str] = flag
        self.flag_to_url[flag] = clean_url_str
        return flag

    def replace_urls_with_flags(self, text: str) -> tuple[str, dict[str, str]]:
        """
        Replace all URLs in text with flags.
        Invalid/incomplete URLs are removed rather than flagged.
        Returns: (text_with_flags, flag_to_url_mapping)
        """
        # First, handle markdown links: [text](url)
        markdown_link_pattern = r"\[([^\]]+)\]\(([^\)]+)\)"

        def replace_markdown_url(match):
            link_text = match.group(1)
            url = match.group(2)
            flag = self.get_or_create_flag(url)
            if flag is None:
                # Invalid URL - remove the link but keep the text
                return link_text
            return f"[{link_text}]({flag})"

        text = re.sub(markdown_link_pattern, replace_markdown_url, text)

        # Then handle bare URLs - use more strict pattern to avoid incomplete URLs
        # Require at least domain.tld pattern (e.g., example.com)
        # This pattern is more strict: requires at least one dot in the domain part
        bare_url_pattern = r"(https?://[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+(?:/[^\s\)\]\.,;:!?<>\n]*)?)"

        def replace_bare_url(match):
            url = match.group(1)
            # Remove trailing punctuation that's likely not part of URL
            url = url.rstrip(".,;:!?)")
            flag = self.get_or_create_flag(url)
            if flag is None:
                # Invalid URL - remove it
                return ""
            return flag

        text = re.sub(bare_url_pattern, replace_bare_url, text)

        # Fix URLs that might be broken across lines
        broken_url_pattern = r"(https?://[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+[^\s\)\]<>]*)\s*\n\s*([^\s\)\]<>]+)"

        def replace_broken_url(match):
            url_part1 = match.group(1)
            url_part2 = match.group(2)
            combined = url_part1 + url_part2
            flag = self.get_or_create_flag(combined)
            if flag is None:
                # Invalid URL - remove it
                return ""
            return flag

        text = re.sub(broken_url_pattern, replace_broken_url, text)

        # Clean up any double spaces or empty parentheses left by removed URLs
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\(\s*\)", "", text)

        return text, self.flag_to_url.copy()

    def replace_flags_with_urls(
        self, text: str, flag_to_url: dict[str, str] = None
    ) -> str:
        """
        Replace flags in text with actual URLs.
        If flag_to_url is provided, use that mapping; otherwise use internal mapping.
        Missing flags are logged and removed rather than left as-is.
        """
        mapping = flag_to_url if flag_to_url is not None else self.flag_to_url

        if not mapping:
            return text

        # Track missing flags for logging
        missing_flags = set()

        # Replace flags in markdown links: [text]([URL_N])
        markdown_flag_pattern = r"\[([^\]]+)\]\((\[URL_\d+\])\)"

        def replace_markdown_flag(match):
            link_text = match.group(1)
            flag = match.group(2)
            url = mapping.get(flag)
            if url is None:
                # Flag not found - log and remove the link but keep the text
                missing_flags.add(flag)
                return link_text
            return f"[{link_text}]({url})"

        text = re.sub(markdown_flag_pattern, replace_markdown_flag, text)

        # Replace bare flags (not in markdown links)
        # Use word boundaries to avoid matching flags that are already in markdown links
        bare_flag_pattern = r"(?<!\]\()(\[URL_\d+\])(?!\))"

        def replace_bare_flag(match):
            flag = match.group(1)
            url = mapping.get(flag)
            if url is None:
                # Flag not found - log and remove it
                missing_flags.add(flag)
                return ""
            return url

        text = re.sub(bare_flag_pattern, replace_bare_flag, text)

        # Log missing flags if any
        if missing_flags:
            logger.warning(
                f"Found {len(missing_flags)} missing URL flags in output: {sorted(missing_flags)}"
            )

        # Clean up any double spaces left by removed flags
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)  # Remove space before punctuation

        return text

    def get_url_reference_table(self) -> str:
        """Generate a reference table of flags to URLs for LLM context."""
        if not self.flag_to_url:
            return ""

        lines = ["\nURL Reference Table (use these flags when referencing URLs):"]
        # Sort by flag number
        sorted_flags = sorted(
            self.flag_to_url.items(),
            key=lambda x: (
                int(re.search(r"\d+", x[0]).group()) if re.search(r"\d+", x[0]) else 0
            ),
        )
        for flag, url in sorted_flags:
            lines.append(f"  {flag} = {url}")
        lines.append(
            "IMPORTANT: Use the flags (e.g., [URL_1]) in your response, NOT the full URLs."
        )
        return "\n".join(lines)
