from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from requests import Response

LOW_SIGNAL_DOMAINS = (
    "reddit.com",
    "quora.com",
    "medium.com",
    "substack.com",
    "shopify.com",
    "linkedin.com",
)


def load_prompts(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    json_payload: dict[str, Any] | None = None,
    timeout: int = 30,
    attempts: int = 5,
) -> Response:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = session.request(method, url, json=json_payload, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(min(2 * attempt, 10))
    raise last_error or RuntimeError(
        f"{method} {url} failed without a captured exception"
    )


def get_json(session: requests.Session, url: str) -> Any:
    response = _request_with_retry(session, "GET", url)
    return response.json()


def post_json(
    session: requests.Session, url: str, payload: dict[str, Any] | None = None
) -> Any:
    response = _request_with_retry(session, "POST", url, json_payload=payload)
    if response.content:
        return response.json()
    return None


def contains_low_signal_plan(plan: dict[str, Any]) -> bool:
    joined = json.dumps(plan).lower()
    return any(domain in joined for domain in LOW_SIGNAL_DOMAINS)


def count_authoritative_links(text: str) -> int:
    urls = re.findall(r"https?://[^\s)]+", text or "")
    return sum(
        1
        for url in urls
        if any(token in url.lower() for token in (".gov", ".edu", "nsf.gov", "nih.gov"))
    )


def extract_quality_signals(
    plan: dict[str, Any], reports: list[dict[str, Any]], export_payload: dict[str, Any]
) -> dict[str, Any]:
    final_report = ""
    if reports:
        preferred = next(
            (
                report
                for report in reports
                if str(report.get("title", "")).startswith("Final Report")
            ),
            reports[0],
        )
        final_report = str(preferred.get("content", "") or "")

    return {
        "plan_subtask_count": len(plan.get("subtasks", [])),
        "plan_has_low_signal_domains": contains_low_signal_plan(plan),
        "report_has_verified_section": "## Verified Findings" in final_report,
        "report_has_tentative_section": "## Tentative Findings" in final_report,
        "report_has_uncertainty_section": "## Open Uncertainties" in final_report,
        "report_has_next_steps_section": "## Next Verification Steps" in final_report,
        "authoritative_link_count": count_authoritative_links(final_report),
        "report_count": len(reports),
        "export_has_searches": bool(export_payload.get("searches")),
        "export_has_reports": bool(export_payload.get("reports")),
    }


def run_prompt(
    session: requests.Session,
    base_url: str,
    prompt: dict[str, Any],
    max_wait_seconds: int,
) -> dict[str, Any]:
    payload = {
        "name": prompt["name"],
        "description": prompt["description"],
        "budgetMinutes": int(prompt.get("budget_minutes", 30)),
    }
    task = post_json(session, f"{base_url}/api/v1/tasks", payload)
    task_id = task["id"]
    post_json(session, f"{base_url}/api/v1/tasks/{task_id}/start")

    start = time.time()
    milestones: dict[str, float] = {}
    final_task: dict[str, Any] | None = None
    final_plan: dict[str, Any] = {}

    while time.time() - start <= max_wait_seconds:
        time.sleep(5)
        final_task = get_json(session, f"{base_url}/api/v1/tasks/{task_id}")
        final_plan = get_json(
            session, f"{base_url}/api/v1/tasks/{task_id}/research-plan"
        )

        if "subtasks" in final_plan and "plan_generated_seconds" not in milestones:
            milestones["plan_generated_seconds"] = round(time.time() - start, 2)
        if final_task.get("status") == "complete":
            milestones["completed_seconds"] = round(time.time() - start, 2)
            break

    reports = get_json(session, f"{base_url}/api/v1/tasks/{task_id}/reports")
    export_payload = get_json(session, f"{base_url}/api/v1/tasks/{task_id}/export")

    return {
        "task_id": task_id,
        "prompt": prompt,
        "timings": milestones,
        "task": final_task,
        "plan": final_plan,
        "reports": reports,
        "export": export_payload,
        "quality_signals": extract_quality_signals(final_plan, reports, export_payload),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run lightweight live API benchmarks for Kestrel."
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("KESTREL_BENCHMARK_BASE_URL", "http://localhost:8000"),
    )
    parser.add_argument("--prompts-file", default="benchmarks/prompts.json")
    parser.add_argument("--prompt-id")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-wait-seconds", type=int, default=1800)
    args = parser.parse_args()

    prompts = load_prompts(Path(args.prompts_file))
    if args.prompt_id:
        prompts = [prompt for prompt in prompts if prompt["id"] == args.prompt_id]
    elif not args.all:
        prompts = prompts[:1]

    if not prompts:
        raise SystemExit("No benchmark prompts selected.")

    session = requests.Session()
    out_dir = Path("notes/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for prompt in prompts:
        result = run_prompt(session, args.base_url, prompt, args.max_wait_seconds)
        results.append(result)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            out_dir / f"{timestamp}_{slugify(prompt['id'])}_{result['task_id']}.json"
        )
        filename.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "prompt_id": prompt["id"],
                    "task_id": result["task_id"],
                    "status": (result.get("task") or {}).get("status"),
                    "timings": result["timings"],
                    "quality_signals": result["quality_signals"],
                    "artifact": str(filename),
                },
                indent=2,
            )
        )

    summary = {
        "ran": len(results),
        "results": [
            {
                "prompt_id": item["prompt"]["id"],
                "task_id": item["task_id"],
                "status": (item.get("task") or {}).get("status"),
                "quality_signals": item["quality_signals"],
            }
            for item in results
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
