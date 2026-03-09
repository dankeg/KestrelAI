"""
Resilient structured-output parsing helpers.

These utilities recover partially malformed JSON model outputs and map them
into Pydantic schemas without requiring a full model retry.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, get_args, get_origin

from pydantic import BaseModel

_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_coerce_text(item) for item in value)
    if isinstance(value, dict):
        if "text" in value:
            return _coerce_text(value.get("text"))
        return json.dumps(value, ensure_ascii=True)
    content = getattr(value, "content", None)
    if content is not None:
        return _coerce_text(content)
    return str(value)


def _extract_json_like_chunks(text: str) -> list[str]:
    chunks: list[str] = []

    # Prefer explicit json code blocks first.
    for match in _CODE_BLOCK_RE.finditer(text):
        candidate = (match.group(1) or "").strip()
        if candidate:
            chunks.append(candidate)

    # Then include balanced JSON object/array chunks from free text.
    in_string = False
    escape = False
    depth = 0
    start_idx: int | None = None
    for idx, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            if depth == 0:
                start_idx = idx
            depth += 1
            continue
        if ch in "}]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start_idx is not None:
                    chunks.append(text[start_idx : idx + 1].strip())
                    start_idx = None

    # If JSON was truncated, keep trailing chunk for repair attempts.
    if depth > 0 and start_idx is not None:
        chunks.append(text[start_idx:].strip())

    # Finally try the entire response text.
    if text.strip():
        chunks.append(text.strip())
    return chunks


def _normalize_json_text(fragment: str) -> str:
    normalized = (fragment or "").strip()
    normalized = (
        normalized.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    normalized = _TRAILING_COMMA_RE.sub(r"\1", normalized)
    return normalized


def _close_json_fragment(fragment: str) -> str:
    """Best-effort fix for truncated JSON: close open strings and braces."""
    text = fragment
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                opener = stack[-1]
                if (opener == "{" and ch == "}") or (opener == "[" and ch == "]"):
                    stack.pop()

    out = text
    if in_string:
        out += '"'
    for opener in reversed(stack):
        out += "}" if opener == "{" else "]"
    return out


def _attempt_json_load(fragment: str) -> Any:
    candidates = [
        fragment,
        _normalize_json_text(fragment),
        _close_json_fragment(_normalize_json_text(fragment)),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            continue

    # Last-resort: parse python-literal-like dict/list.
    py_like = _normalize_json_text(fragment)
    py_like = re.sub(r"\btrue\b", "True", py_like, flags=re.IGNORECASE)
    py_like = re.sub(r"\bfalse\b", "False", py_like, flags=re.IGNORECASE)
    py_like = re.sub(r"\bnull\b", "None", py_like, flags=re.IGNORECASE)
    return ast.literal_eval(py_like)


def _parse_key_value_fallback(text: str, schema: type[BaseModel]) -> dict[str, Any]:
    """Parse simple 'field: value' lines as a final schema-aligned fallback."""
    schema_fields = getattr(schema, "model_fields", {})
    if not schema_fields:
        return {}

    field_map = {name.lower(): name for name in schema_fields.keys()}
    parsed: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-* ")
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = re.sub(r"[^a-z0-9_]+", "_", key.strip().lower()).strip("_")
        target_key = field_map.get(normalized_key)
        if not target_key:
            continue
        cleaned_value = re.sub(r"^[`*_#\-\s]+|[`*_#\s]+$", "", value.strip().strip('"'))
        parsed[target_key] = cleaned_value
    return parsed


def _schema_field_names(schema: type[BaseModel]) -> set[str]:
    return set(getattr(schema, "model_fields", {}).keys())


def _extract_named_section(text: str, headings: tuple[str, ...]) -> str:
    heading_pattern = "|".join(re.escape(item) for item in headings)
    match = re.search(
        rf"(?ims)^\s*(?:\*\*)?(?:\d+\.\s*)?(?:{heading_pattern})(?:\*\*)?\s*:?\s*(.*?)"
        rf"(?=^\s*(?:\*\*)?(?:\d+\.\s*)?(?:[A-Z][^\n]{{0,80}}|subtask\s+\d+|success criteria)(?:\*\*)?\s*:|\Z)",
        text,
    )
    return (match.group(1) or "").strip() if match else ""


def _extract_subtasks_from_markdown(text: str) -> list[dict[str, Any]]:
    subtasks: list[dict[str, Any]] = []
    lines = text.splitlines()
    current: dict[str, Any] | None = None

    def flush_current() -> None:
        nonlocal current
        if not current:
            return
        description = str(current.get("description", "")).strip()
        if not description:
            current = None
            return
        success = str(current.get("success_criteria", "")).strip()
        if not success:
            body_lines = [
                line.strip() for line in current.get("body", []) if line.strip()
            ]
            if body_lines:
                success = " ".join(body_lines[:3])
        subtasks.append(
            {
                "order": len(subtasks) + 1,
                "description": description,
                "success_criteria": success
                or "Collect concrete findings for this subtask.",
            }
        )
        current = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        subtask_match = re.match(
            r"^(?:[-*]\s*)?(?:subtask|task|step)\s*(\d+)?\s*[:.-]?\s*(.+)$",
            line,
            re.IGNORECASE,
        )
        numbered_match = re.match(r"^(?:[-*]\s*)?(\d+)[.)]\s+(.+)$", line)
        success_match = re.match(
            r"^(?:[-*]\s*)?success criteria\s*[:.-]?\s*(.+)$",
            line,
            re.IGNORECASE,
        )

        if success_match and current is not None:
            current["success_criteria"] = success_match.group(1).strip()
            continue

        if subtask_match:
            flush_current()
            current = {
                "description": subtask_match.group(2).strip("* ").strip(),
                "body": [],
                "success_criteria": "",
            }
            continue

        if numbered_match:
            candidate = numbered_match.group(2).strip("* ").strip()
            lowered = candidate.lower()
            if any(
                token in lowered
                for token in (
                    "restatement",
                    "objective",
                    "constraint",
                    "deliverable",
                    "success criteria",
                )
            ):
                continue
            flush_current()
            current = {
                "description": candidate,
                "body": [],
                "success_criteria": "",
            }
            continue

        if current is not None:
            current.setdefault("body", []).append(line)

    flush_current()
    return subtasks


def _extract_review_decision_fallback(text: str) -> dict[str, Any]:
    normalized = _normalize_json_text(text)
    lowered = normalized.lower()

    decision = "continue"
    if re.search(r"\b(done|complete)\b", lowered):
        decision = "done"
    elif re.search(r"\b(switch|proceed|advance|move to next)\b", lowered):
        decision = "switch"
    elif re.search(r"\bcontinue\b", lowered):
        decision = "continue"

    subtask = "proceed" if decision in {"switch", "done"} else "stay"

    action_match = re.search(
        r"(?im)^\s*(?:decision|action)\s*:\s*(.+)$",
        normalized,
    )
    feedback = normalized.strip()
    if action_match:
        feedback = action_match.group(1).strip()
    feedback = re.sub(
        r"(?im)^\s*(?:decision|action)\s*:\s*",
        "",
        feedback,
    ).strip()
    feedback = re.sub(r"(?im)^\s*\*+\s*", "", feedback).strip()
    feedback = feedback[:600] if feedback else "Continue gathering evidence."

    reasoning = normalized.strip()
    reasoning = reasoning[:600] if reasoning else feedback

    return {
        "reasoning": reasoning or feedback,
        "decision": decision,
        "feedback": feedback or "Continue gathering evidence.",
        "subtask": subtask,
        "next_task": "current_task",
    }


def _repair_review_decision_dict(data: dict[str, Any]) -> dict[str, Any]:
    normalized = {str(key): value for key, value in (data or {}).items()}
    text_hint = " ".join(
        str(normalized.get(field, "") or "")
        for field in ("reasoning", "feedback", "decision", "subtask", "next_task")
    ).strip()
    fallback = _extract_review_decision_fallback(text_hint or json.dumps(normalized))

    repaired = dict(normalized)
    repaired["decision"] = _normalize_control_literal(
        "decision",
        repaired.get("decision") or fallback["decision"],
    )
    repaired["subtask"] = _normalize_control_literal(
        "subtask",
        repaired.get("subtask")
        or ("proceed" if repaired["decision"] in {"switch", "done"} else "stay"),
    )
    repaired["feedback"] = str(
        repaired.get("feedback")
        or repaired.get("reasoning")
        or fallback["feedback"]
        or "Continue gathering evidence."
    ).strip()
    repaired["reasoning"] = str(
        repaired.get("reasoning")
        or repaired.get("feedback")
        or fallback["reasoning"]
        or repaired["feedback"]
    ).strip()
    repaired["next_task"] = str(
        repaired.get("next_task") or fallback["next_task"] or "current_task"
    ).strip()
    if not repaired["feedback"]:
        repaired["feedback"] = fallback["feedback"]
    if not repaired["reasoning"]:
        repaired["reasoning"] = repaired["feedback"]
    return repaired


def _coerce_field_value(value: Any, annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin in {list, set, tuple}:
        if value is None:
            return []
        if isinstance(value, str):
            pieces = [
                item.strip(" -*\t")
                for item in re.split(r"[\n;]+", value)
                if item.strip(" -*\t")
            ]
            return pieces or [value.strip()]
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]
    if annotation is str and isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return value


def _normalize_control_literal(field_name: str, value: Any) -> Any:
    if not isinstance(value, str):
        return value
    normalized = value.strip()
    lowered = normalized.lower().replace("-", "_").replace(" ", "_")
    if field_name == "decision":
        mapping = {
            "continue": "continue",
            "switch": "switch",
            "done": "done",
            "complete": "done",
            "completed": "done",
            "proceed": "switch",
            "advance": "switch",
        }
        return mapping.get(lowered, lowered)
    if field_name == "subtask":
        mapping = {
            "stay": "stay",
            "continue": "stay",
            "remain": "stay",
            "proceed": "proceed",
            "advance": "proceed",
            "next": "proceed",
        }
        return mapping.get(lowered, lowered)
    if field_name == "action":
        mapping = {
            "think": "think",
            "search": "search",
            "mcp_tool": "mcp_tool",
            "mcp": "mcp_tool",
            "done": "done",
        }
        return mapping.get(lowered, lowered)
    return normalized


def _coerce_dict_for_schema(
    data: dict[str, Any], schema: type[BaseModel]
) -> dict[str, Any]:
    schema_fields = getattr(schema, "model_fields", {})
    if not schema_fields:
        return data
    field_names = set(schema_fields.keys())
    working = dict(data)
    if {"decision", "feedback", "subtask", "next_task", "reasoning"}.issubset(
        field_names
    ):
        working = _repair_review_decision_dict(working)
    coerced: dict[str, Any] = {}
    for key, value in working.items():
        field = schema_fields.get(key)
        if field is None:
            coerced[key] = value
            continue
        normalized_value = _normalize_control_literal(key, value)
        coerced[key] = _coerce_field_value(normalized_value, field.annotation)
    return coerced


def _schema_specific_fallback(text: str, schema: type[BaseModel]) -> dict[str, Any]:
    field_names = _schema_field_names(schema)

    if {"action", "reasoning", "query", "thought"}.issubset(field_names):
        action_match = re.search(
            r"(?im)^\s*(?:action\s*:\s*)?(think|search|mcp_tool|done)\s*:\s*(.+)$",
            text,
        )
        if action_match:
            action = action_match.group(1).lower()
            tail = action_match.group(2).strip()
            parsed = {
                "reasoning": tail,
                "action": action,
                "query": "",
                "thought": "",
                "tool_name": "",
                "tool_parameters": {},
            }
            if action == "search":
                parsed["query"] = tail
            elif action == "think":
                parsed["thought"] = tail
            return parsed

    if {"decision", "feedback", "subtask", "next_task", "reasoning"}.issubset(
        field_names
    ):
        return _extract_review_decision_fallback(text)

    if {"restated_task", "subtasks"}.issubset(field_names):
        restated_task = _extract_named_section(
            text,
            (
                "Restatement of Task",
                "Restated Task",
                "Task Restatement",
                "Task Summary",
            ),
        )
        subtasks = _extract_subtasks_from_markdown(text)
        if restated_task or subtasks:
            return {
                "restated_task": restated_task or "Research task decomposition",
                "subtasks": subtasks,
            }

    return {}


def parse_to_schema(
    raw_output: Any,
    schema: type[BaseModel],
) -> BaseModel:
    """Convert provider output into a concrete Pydantic schema instance."""
    if isinstance(raw_output, schema):
        return raw_output
    if isinstance(raw_output, BaseModel):
        return schema.model_validate(
            _coerce_dict_for_schema(raw_output.model_dump(), schema)
        )
    if isinstance(raw_output, dict):
        return schema.model_validate(_coerce_dict_for_schema(raw_output, schema))

    text = _coerce_text(raw_output).strip()
    if not text:
        raise ValueError("empty model output")

    parse_errors: list[str] = []
    for chunk in _extract_json_like_chunks(text):
        try:
            obj = _attempt_json_load(chunk)
            if isinstance(obj, dict):
                return schema.model_validate(_coerce_dict_for_schema(obj, schema))
        except Exception as e:
            parse_errors.append(str(e))

    kv_fallback = _parse_key_value_fallback(text, schema)
    if kv_fallback:
        return schema.model_validate(_coerce_dict_for_schema(kv_fallback, schema))

    schema_fallback = _schema_specific_fallback(text, schema)
    if schema_fallback:
        return schema.model_validate(_coerce_dict_for_schema(schema_fallback, schema))

    raise ValueError(
        "unable to parse model output into schema; sample="
        + text[:300]
        + (f" | parse_errors={'; '.join(parse_errors[:3])}" if parse_errors else "")
    )
