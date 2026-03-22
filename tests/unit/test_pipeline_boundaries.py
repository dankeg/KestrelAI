from unittest.mock import AsyncMock, Mock

import pytest

try:
    from KestrelAI.agents.base_agent import AgentState
    from KestrelAI.agents.context_builder import ContextBuilder
    from KestrelAI.agents.prompt_builder import PromptBuilder
    from KestrelAI.agents.research_config import ResearchConfig
    from KestrelAI.agents.research_orchestrator import OrchestratorDecision
    from KestrelAI.agents.url_utils import URLFlagManager
    from KestrelAI.graphs.orchestrator_runner import LangGraphOrchestratorRunner
    from KestrelAI.graphs.worker_step_runner import LangGraphWorkerStepRunner
except ImportError:
    from agents.base_agent import AgentState
    from agents.context_builder import ContextBuilder
    from agents.prompt_builder import PromptBuilder
    from agents.research_config import ResearchConfig
    from agents.research_orchestrator import OrchestratorDecision
    from agents.url_utils import URLFlagManager
    from graphs.orchestrator_runner import LangGraphOrchestratorRunner
    from graphs.worker_step_runner import LangGraphWorkerStepRunner


@pytest.mark.unit
def test_context_builder_basic_includes_orchestrator_guidance_and_url_flags(mock_task):
    config = ResearchConfig(
        is_subtask_agent=True,
        subtask_description="Investigate strong fellowship leads.",
        success_criteria="Find official opportunity pages.",
        previous_findings="Earlier note: https://example.com/findings",
        orchestrator_guidance="Pivot to official program pages and organization directories.",
    )
    builder = ContextBuilder(config, URLFlagManager())

    state = AgentState(task_id=mock_task.name)
    state.last_checkpoint = "Checkpoint from https://example.org/checkpoint"
    state.history.append("Searched https://example.net/listing")

    context = builder.build_context(mock_task, state)

    assert "Orchestrator guidance: Pivot to official program pages" in context
    assert "[URL_" in context


@pytest.mark.unit
def test_context_builder_token_aware_passes_guidance_component(mock_task):
    config = ResearchConfig(
        is_subtask_agent=True,
        subtask_description="Investigate strong fellowship leads.",
        success_criteria="Find official opportunity pages.",
        orchestrator_guidance="Use official sources and directory pages.",
    )
    context_manager = Mock()
    context_manager.build_context.return_value = ("token-aware-context", {"total": 12})
    token_budget = Mock(rag_content=256)
    builder = ContextBuilder(
        config,
        URLFlagManager(),
        context_manager=context_manager,
        token_budget=token_budget,
    )

    state = AgentState(task_id=mock_task.name)
    context = builder.build_context(mock_task, state)

    assert context == "token-aware-context"
    components = context_manager.build_context.call_args.args[0]
    assert components["orchestrator_guidance"] == (
        "Use official sources and directory pages."
    )


@pytest.mark.unit
def test_prompt_builder_subtask_prompt_surfaces_orchestrator_guidance():
    config = ResearchConfig(
        is_subtask_agent=True,
        subtask_description="Investigate strong fellowship leads.",
        success_criteria="Find official opportunity pages.",
        orchestrator_guidance="Use official sources and directory pages.",
    )

    prompt = PromptBuilder(config).get_system_prompt()

    assert "Orchestrator Guidance: Use official sources and directory pages." in prompt
    assert "SEARCH QUERY RULES:" in prompt


@pytest.mark.unit
@pytest.mark.asyncio
async def test_orchestrator_runner_review_decision_records_feedback(mock_task):
    runner = LangGraphOrchestratorRunner.__new__(LangGraphOrchestratorRunner)
    decision = OrchestratorDecision(
        reasoning="Need more evidence",
        decision="continue",
        feedback="Pivot to official sources.",
        subtask="stay",
        next_task=mock_task.name,
    )
    task_state = Mock()
    task_state.record_decision = Mock()
    runner.orchestrator = Mock(
        _review=AsyncMock(return_value=decision),
        task_states={mock_task.name: task_state},
    )

    result = await LangGraphOrchestratorRunner._review_decision(
        runner,
        {"task": mock_task, "latest_notes": "latest"},
    )

    assert result["decision"] == decision
    task_state.record_decision.assert_called_once_with(
        decision.decision, decision.feedback
    )


@pytest.mark.unit
def test_orchestrator_runner_routes_no_more_subtasks_to_finalize():
    runner = LangGraphOrchestratorRunner.__new__(LangGraphOrchestratorRunner)
    route = LangGraphOrchestratorRunner._route_after_subtask(
        runner,
        {"status": "no_more_subtasks"},
    )
    assert route == "finalize"


@pytest.mark.unit
def test_worker_step_runner_aggregate_activity_keeps_true_action_count(mock_task):
    runner = LangGraphWorkerStepRunner.__new__(LangGraphWorkerStepRunner)
    task_id = "task-123"

    agent_state = AgentState(task_id=mock_task.name)
    agent_state.action_count = 5
    agent_state.search_count = 2
    agent_state.last_step_feedback = "[SEARCH] Found new official results"
    agent_state.last_step_activity = "search"

    current_agent = Mock()
    current_agent._state = {mock_task.name: agent_state}

    task_state = Mock()
    task_state.subtask_index = 0
    task_state.subtask_agents = {0: current_agent}

    worker = Mock()
    worker.task_metrics = {
        task_id: {
            "action_count": 0,
            "last_agent_action_count": 4,
            "last_search_count": 1,
            "last_think_count": 0,
            "last_summary_count": 0,
            "last_checkpoint_count": 0,
        }
    }
    worker.redis_client = Mock()
    worker.orchestrator = Mock(task_states={mock_task.name: task_state})
    runner.worker = worker

    result = LangGraphWorkerStepRunner._aggregate_activity(
        runner,
        {
            "task_id": task_id,
            "task": mock_task,
            "current_subtask": "Current subtask",
            "progress_info": {
                "subtasks": [
                    {
                        "agent_metrics": {
                            "total_searches": 2,
                            "total_thoughts": 0,
                            "total_summaries": 0,
                            "total_checkpoints": 0,
                            "action_count": 5,
                        }
                    }
                ]
            },
        },
    )

    assert result["has_meaningful_activity"] is True
    assert worker.task_metrics[task_id]["action_count"] == 5
    worker.redis_client.send_activity.assert_called_once()


@pytest.mark.unit
def test_worker_step_runner_aggregate_activity_uses_get_metrics_fallback(mock_task):
    runner = LangGraphWorkerStepRunner.__new__(LangGraphWorkerStepRunner)
    task_id = "task-456"

    agent_state = AgentState(task_id=mock_task.name)
    agent_state.action_count = 1
    agent_state.search_count = 1
    agent_state.last_step_feedback = "[SEARCH] Found official result"
    agent_state.last_step_activity = "search"

    current_agent = Mock()
    current_agent._state = {mock_task.name: agent_state}
    current_agent.get_global_metrics = None
    current_agent.get_metrics = Mock(
        return_value={
            "total_web_fetches": 3,
            "total_llm_calls": 7,
        }
    )
    current_agent.get_task_metrics = Mock(return_value={"searches": []})

    task_state = Mock()
    task_state.subtask_index = 0
    task_state.subtask_agents = {0: current_agent}

    worker = Mock()
    worker.task_metrics = {
        task_id: {
            "action_count": 0,
            "last_agent_action_count": 0,
            "last_search_count": 0,
            "last_think_count": 0,
            "last_summary_count": 0,
            "last_checkpoint_count": 0,
            "search_count": 1,
            "think_count": 0,
            "summary_count": 0,
            "checkpoint_count": 0,
        }
    }
    worker.redis_client = Mock()
    worker.orchestrator = Mock(task_states={mock_task.name: task_state})
    worker.agent = current_agent
    worker.global_metrics = {}
    worker._compute_elapsed_and_progress = Mock(return_value=(30, 25.0))
    runner.worker = worker

    result = LangGraphWorkerStepRunner._emit_outputs(
        runner,
        {
            "task_id": task_id,
            "task": mock_task,
            "notes": "",
            "progress_info": {"subtasks": []},
            "has_meaningful_activity": True,
            "activity_type": "search",
            "step_timed_out": False,
        },
    )

    assert result["should_stop"] is False
    update_payload = worker.redis_client.send_update.call_args.kwargs
    assert update_payload["metrics"]["webFetchCount"] == 3
    assert update_payload["metrics"]["llmTokensUsed"] == 7000


@pytest.mark.unit
def test_url_flag_manager_replace_known_urls_with_flags_drops_unknown_urls():
    manager = URLFlagManager()
    _, mapping = manager.replace_urls_with_flags(
        "Known source: https://example.edu/program"
    )

    text = (
        "Use [Official Program](https://example.edu/program) and "
        "ignore https://invented.example.com/page."
    )
    rewritten = manager.replace_known_urls_with_flags(text, mapping)

    assert "[Official Program]([URL_1])" in rewritten
    assert "invented.example.com" not in rewritten


@pytest.mark.unit
def test_url_flag_manager_restores_bare_flags_inside_parentheses():
    manager = URLFlagManager()
    _, mapping = manager.replace_urls_with_flags(
        "Known source: https://example.edu/program"
    )

    restored = manager.replace_flags_with_urls(
        "Primary source ([URL_1])",
        mapping,
    )

    assert "https://example.edu/program" in restored
    assert "[URL_1]" not in restored


@pytest.mark.unit
def test_url_flag_manager_strip_unknown_urls_keeps_only_evidenced_links():
    manager = URLFlagManager()
    _, mapping = manager.replace_urls_with_flags(
        "Known source: https://example.edu/program"
    )

    restored = manager.replace_flags_with_urls(
        "[https://example.edu/program]([URL_1]) and https://invented.example.com/page",
        mapping,
    )
    filtered = manager.strip_unknown_urls(restored, list(mapping.values()))

    assert "https://example.edu/program" in filtered
    assert "invented.example.com" not in filtered


@pytest.mark.unit
def test_url_flag_manager_strip_unknown_urls_preserves_markdown_links():
    manager = URLFlagManager()
    known = ["https://example.edu/program"]
    text = "[Program Website](https://example.edu/program) and https://invented.example.com/page"

    filtered = manager.strip_unknown_urls(text, known)

    assert "[Program Website](https://example.edu/program)" in filtered
    assert "invented.example.com" not in filtered


@pytest.mark.unit
def test_url_flag_manager_restores_grouped_flag_lists():
    manager = URLFlagManager()
    _, mapping = manager.replace_urls_with_flags(
        "Known sources: https://example.edu/program and https://example.org/opportunity"
    )

    restored = manager.replace_flags_with_urls("[URL_1, URL_2]", mapping)

    assert "https://example.edu/program" in restored
    assert "https://example.org/opportunity" in restored
    assert "URL_1" not in restored
