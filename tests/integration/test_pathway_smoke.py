import asyncio
import os

import pytest

try:
    from KestrelAI.agents.base import LlmWrapper
    from KestrelAI.agents.research_orchestrator import (
        OrchestratorDecision,
        ResearchOrchestrator,
    )
    from KestrelAI.memory.vector_store import MemoryStore
    from KestrelAI.shared.models import ResearchPlan, Subtask, SubtaskType, Task
except ImportError:
    from agents.base import LlmWrapper
    from agents.research_orchestrator import OrchestratorDecision, ResearchOrchestrator
    from memory.vector_store import MemoryStore
    from shared.models import ResearchPlan, Subtask, SubtaskType, Task


MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "ministral-8b-latest")
MISTRAL_HOST = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")

CASES = [
    {
        "name": "RAG Eval Papers",
        "description": (
            "Find strong papers and conference proceedings about evaluation methods "
            "for retrieval-augmented generation agents."
        ),
        "subtask_description": (
            "Discover papers, proceedings, and lab or publisher sources across "
            "different source classes."
        ),
        "success_criteria": (
            "Cover multiple source classes and find authoritative benchmark or "
            "evaluation papers."
        ),
        "good": ("conference", "proceedings", "publisher", "benchmark", "paper"),
        "bad": ("opportunity", "fellowship", "grant"),
    },
    {
        "name": "Local Agent Ecosystem",
        "description": (
            "Map the current open-source local-first AI agent ecosystem, including "
            "frameworks, repositories, and documentation hubs."
        ),
        "subtask_description": (
            "Discover repository, documentation, and ecosystem index sources across "
            "different source classes."
        ),
        "success_criteria": (
            "Cover multiple source classes and identify authoritative framework or "
            "repository sources."
        ),
        "good": ("repository", "documentation", "index", "framework"),
        "bad": ("opportunity", "fellowship", "grant", "conference"),
    },
]


def _build_task(payload: dict[str, str]) -> Task:
    task = Task(
        name=payload["name"],
        description=payload["description"],
        budgetMinutes=20,
    )
    task.research_plan = ResearchPlan(
        restated_task=payload["description"],
        subtasks=[
            Subtask(
                order=1,
                description=payload["subtask_description"],
                success_criteria=payload["success_criteria"],
                subtask_type=SubtaskType.DISCOVERY,
            )
        ],
        current_subtask_index=0,
    )
    return task


async def _run_case(payload: dict[str, str]) -> dict[str, object]:
    llm = LlmWrapper(
        model=MISTRAL_MODEL,
        host=MISTRAL_HOST,
        provider="openai_compatible",
        api_key=MISTRAL_API_KEY,
    )
    memory = MemoryStore()
    task = _build_task(payload)

    orchestrator = ResearchOrchestrator(
        [task], llm, profile="kestrel", max_context_tokens=32768
    )
    task_state = orchestrator.task_states[task.name]
    task_state.research_plan = task.research_plan
    task_state.subtask_index = 0

    agent = task_state.create_subtask_agent(0, llm, memory)

    await agent.run_step(task)
    decision = OrchestratorDecision(
        reasoning="Need more breadth across discovery pathways.",
        decision="continue",
        feedback=(
            "Continue gathering evidence for the current subtask using uncovered "
            "discovery pathways and source classes."
        ),
        subtask="stay",
        next_task=task.name,
    )
    guidance = orchestrator._build_guidance_from_decision(task, task_state, decision)
    orchestrator._set_subtask_guidance(task, task_state, 0, guidance)

    await agent.run_step(task)
    state = agent._state[task.name]
    latest_history = list(getattr(state, "search_history", []) or [])
    latest = latest_history[-1] if latest_history else {}
    query = str(latest.get("query", "") or "")
    pathways = list(getattr(state, "search_pathways", []) or [])
    return {
        "query": query,
        "pathways": pathways,
        "metrics": {
            "attempted": sum(
                1
                for pathway in pathways
                if int(pathway.get("attempt_count", 0) or 0) > 0
            ),
            "hit": sum(
                1 for pathway in pathways if int(pathway.get("hit_count", 0) or 0) > 0
            ),
        },
    }


@pytest.mark.integration
@pytest.mark.skipif(
    not MISTRAL_API_KEY, reason="MISTRAL_API_KEY is required for live pathway probes"
)
@pytest.mark.asyncio
async def test_pathway_smoke_with_mistral():
    for payload in CASES:
        result = await _run_case(payload)
        lowered = result["query"].lower()
        assert result["query"], payload["name"]
        assert result["metrics"]["attempted"] >= 1, payload["name"]
        assert result["metrics"]["hit"] >= 1, payload["name"]
        assert result["pathways"], payload["name"]
        assert any(term in lowered for term in payload["good"]), payload["name"]
        assert not any(term in lowered for term in payload["bad"]), payload["name"]
