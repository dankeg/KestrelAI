"""
LangChain structured planning chains for WebResearchAgent actions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from KestrelAI.agents.structured_parsing import parse_to_schema
from KestrelAI.graphs.schemas import ResearchActionPlan

try:
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:  # pragma: no cover - dependency-gated path
    ChatPromptTemplate = None


def _require_langchain() -> None:
    if ChatPromptTemplate is None:
        raise ImportError("langchain_core is required for action chains")


@dataclass
class WebResearchActionChains:
    """Structured chain wrappers for next-action planning."""

    model: Any

    def __post_init__(self) -> None:
        _require_langchain()
        self._next_action_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", "{system_prompt}"),
                    ("human", "{context}"),
                ]
            )
            | self.model
        )

    def next_action(self, *, system_prompt: str, context: str) -> ResearchActionPlan:
        result = self._next_action_chain.invoke(
            {"system_prompt": system_prompt, "context": context}
        )
        return self._parse_action_plan(result)

    @staticmethod
    def _parse_action_plan(result: Any) -> ResearchActionPlan:
        """
        Parse model output into ResearchActionPlan without provider-side
        pydantic parse hooks (avoids serializer warning noise).
        """
        try:
            return parse_to_schema(result, ResearchActionPlan)
        except Exception as primary_error:
            content = getattr(result, "content", result)
            if isinstance(content, list):
                text_parts: list[str] = []
                for item in content:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, dict):
                        text_parts.append(str(item.get("text", "")))
                content = "\n".join(part for part in text_parts if part).strip()
            if not isinstance(content, str):
                raise RuntimeError(
                    "LangChain action chain returned invalid schema"
                ) from primary_error
            raw = content.strip()
            try:
                return ResearchActionPlan.model_validate_json(raw)
            except Exception:
                match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
                if not match:
                    raise RuntimeError(
                        "LangChain action chain returned invalid schema"
                    ) from primary_error
                data = json.loads(match.group(0))
                return ResearchActionPlan.model_validate(data)
