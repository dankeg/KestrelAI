"""
Kestrel Research Agent Runner (v4)

Complete rewrite focusing on:
- Intelligent, adaptive research strategies
- Critical analysis and evidence evaluation
- Professional, objective report generation
- Deep integration with orchestrator v3 context
"""

from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Literal, Optional, Set, Tuple
from collections import deque, defaultdict
from uuid import uuid4

import requests
from bs4 import BeautifulSoup

from .base import LlmWrapper
from memory.vector_store import MemoryStore

# Import enhanced orchestrator types
try:
    from agents.orchestrator import (
        TaskSpec, Subtask, AgentUpdate, StepBudget,
        ResearchStrategy, ExecutionContext
    )
except Exception:  # pragma: no cover
    # Fallback definitions
    from pydantic import BaseModel
    from enum import Enum
    
    class ResearchStrategy(str, Enum):
        BROAD_SURVEY = "broad_survey"
        DEEP_DIVE = "deep_dive"
        COMPARATIVE = "comparative"
        FACT_FINDING = "fact_finding"
        EXPLORATORY = "exploratory"
        SYSTEMATIC_REVIEW = "systematic_review"

    class TaskSpec(BaseModel):
        id: str
        name: str
        description: str
        objectives: List[str] = []
        priority: int = 3

    class Subtask(BaseModel):
        id: str
        task_id: str
        title: str
        description: str = ""
        research_questions: List[str] = []
        acceptance_criteria: List[str] = []
        strategy: ResearchStrategy = ResearchStrategy.BROAD_SURVEY
        metadata: Dict[str, Any] = {}

    class ExecutionContext(BaseModel):
        task: TaskSpec
        subtask: Subtask
        related_findings: List[str] = []
        prior_attempts: List[Dict[str, Any]] = []
        global_context: str = ""
        quality_feedback: Optional[str] = None
        suggested_approaches: List[str] = []
        avoided_queries: List[str] = []

    class AgentUpdate(BaseModel):
        timestamp: datetime
        task_id: str
        subtask_id: str
        agent_id: str
        progress: Literal["none", "partial", "good", "complete"]
        findings: str = ""
        sources: List[str] = []
        blockers: List[str] = []
        proposed_next: List[str] = []
        confidence: float = 0.7
        key_insights: List[str] = []
        methodology_notes: str = ""

    class StepBudget(BaseModel):
        max_actions: int = 6
        max_seconds: Optional[int] = None
        max_searches: Optional[int] = None
        max_summaries: Optional[int] = None
        min_quality_score: float = 0.7
        require_evidence: bool = True

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# Enhanced Prompts - Critical and Professional
# --------------------------------------------------------------------------- #

ADAPTIVE_PLANNER_PROMPT = """You are a strategic research planner. Your goal is to gather high-quality evidence efficiently.

CURRENT CONTEXT:
{context}

ADAPTIVE STRATEGIES by research type:
- BROAD_SURVEY: Start wide, then narrow based on findings
- DEEP_DIVE: Focus intensely on specific aspects with follow-up queries
- COMPARATIVE: Alternate between different perspectives/sources
- FACT_FINDING: Target authoritative sources with specific queries
- EXPLORATORY: Follow interesting leads while maintaining focus
- SYSTEMATIC_REVIEW: Methodical coverage of all aspects

DECISION FACTORS:
1. What specific evidence is still needed?
2. Which sources have proven most reliable?
3. What gaps remain in the current understanding?
4. Have we exhausted useful variations of current approach?

OUTPUT FORMAT (JSON):
{{
  "analysis": "Brief assessment of current state and needs",
  "action": "search" | "analyze" | "synthesize" | "evaluate",
  "query": "Specific search query if action=search",
  "focus": "What specific aspect to analyze if action=analyze",
  "rationale": "Why this action best serves the research goals"
}}

ACTIONS:
- search: Gather new evidence with targeted queries
- analyze: Deep examination of existing evidence
- synthesize: Combine findings into coherent insights
- evaluate: Assess if acceptance criteria are met"""

EVIDENCE_EVALUATOR_PROMPT = """Evaluate this evidence for research use.

Evidence:
{evidence}

Research Context:
{context}

Rate on these dimensions (0-10):
1. RELEVANCE: How well does it address the research questions?
2. CREDIBILITY: Source authority and reliability
3. SPECIFICITY: Concrete data, facts, examples vs. generalities
4. RECENCY: How current is the information?
5. UNIQUENESS: New information vs. redundancy

Also identify:
- Key facts or claims worth including
- Potential biases or limitations
- Follow-up questions raised

Be critical and objective."""

CRITICAL_ANALYZER_PROMPT = """Analyze the collected evidence to extract key insights.

Research Questions:
{questions}

Evidence Summary:
{evidence}

Provide:
1. PATTERN ANALYSIS: What patterns or themes emerge?
2. CONTRADICTIONS: What conflicting information exists?
3. EVIDENCE GAPS: What's missing or unclear?
4. CONFIDENCE LEVELS: How certain can we be about each finding?
5. CAUSAL RELATIONSHIPS: What causes or connections are supported?

Be analytical and objective. Avoid speculation beyond what evidence supports."""

PROFESSIONAL_SYNTHESIS_PROMPT = """Create a professional research memorandum.

REQUIREMENTS:
- Professional, objective tone - no self-praise or meta-commentary
- Every claim must be supported by specific evidence
- Use concrete data, quotes, and examples
- Acknowledge limitations and uncertainties
- Clear, scannable structure with bold key points

TASK: {task_name}
SUBTASK: {subtask_title}
RESEARCH QUESTIONS: {questions}
KEY FINDINGS: {findings}
EVIDENCE MAPPING: {evidence_map}

Structure your response as a formal memorandum with:
1. Executive Summary (3-5 bullet points)
2. Detailed Findings (organized by theme)
3. Evidence Assessment
4. Limitations & Gaps
5. Recommendations

Use [^n] citations for all substantive claims."""

QUERY_OPTIMIZER_PROMPT = """Generate an optimized search query.

GOAL: {goal}
PREVIOUS QUERIES: {previous}
CURRENT KNOWLEDGE: {knowledge}
FAILED APPROACHES: {failed}

Create a search query that:
1. Avoids repetition of previous searches
2. Targets the specific missing information
3. Uses precise terms likely to yield authoritative results
4. Includes relevant qualifiers (dates, locations, types)
5. Balances specificity with result availability

Return only the query string, no explanation."""

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888/search")
SEARCH_RESULTS = 10
FETCH_BYTES = 50_000
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
CONTEXT_WINDOW = 50  # Increased for better memory
CHECKPOINT_FREQ = 8
MAX_SNIPPET_LENGTH = 3000
MIN_EVIDENCE_SCORE = 4.0  # Minimum quality threshold

# --------------------------------------------------------------------------- #
# Enhanced Data Structures
# --------------------------------------------------------------------------- #

@dataclass
class Evidence:
    """Rich evidence representation"""
    url: str
    title: str
    content: str
    summary: str
    extracted_facts: List[str]
    relevance_score: float
    credibility_score: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResearchThread:
    """Track a line of investigation"""
    theme: str
    queries: List[str]
    findings: List[str]
    evidence_ids: List[str]
    status: Literal["active", "exhausted", "concluded"]
    confidence: float

@dataclass
class SubtaskState:
    """Enhanced state tracking"""
    # Search tracking
    queries_attempted: Set[str] = field(default_factory=set)
    successful_queries: Dict[str, int] = field(default_factory=dict)  # query -> result count
    failed_queries: Set[str] = field(default_factory=set)
    
    # Evidence management
    evidence_store: Dict[str, Evidence] = field(default_factory=dict)
    evidence_scores: Dict[str, float] = field(default_factory=dict)
    evidence_graph: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))  # evidence connections
    
    # Research threads
    threads: Dict[str, ResearchThread] = field(default_factory=dict)
    active_thread: Optional[str] = None
    
    # Analysis results
    key_findings: List[Dict[str, Any]] = field(default_factory=list)
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    
    # Progress tracking
    action_count: int = 0
    search_count: int = 0
    analysis_count: int = 0
    synthesis_count: int = 0
    last_checkpoint: str = ""
    checkpoints: List[str] = field(default_factory=list)
    
    # Context and history
    working_context: str = ""
    decision_history: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=20))
    quality_history: Deque[float] = field(default_factory=lambda: deque(maxlen=10))

# --------------------------------------------------------------------------- #
# Enhanced Action Types
# --------------------------------------------------------------------------- #

class ResearchAction(BaseModel):
    """Structured research action"""
    analysis: str
    action: Literal["search", "analyze", "synthesize", "evaluate"]
    query: Optional[str] = None
    focus: Optional[str] = None
    rationale: str

class EvidenceEvaluation(BaseModel):
    """Structured evidence evaluation"""
    relevance: float
    credibility: float
    specificity: float
    recency: float
    uniqueness: float
    key_facts: List[str]
    limitations: List[str]
    follow_up_questions: List[str]

# --------------------------------------------------------------------------- #
# Research Agent Runner
# --------------------------------------------------------------------------- #

class ResearchAgentRunner:
    """Enhanced research agent with adaptive intelligence"""
    
    def __init__(
        self,
        memory: MemoryStore,
        llm: LlmWrapper,
        agent_id: str,
        *,
        planner_model: Any,
        evaluator_model: Optional[Any] = None
    ):
        self.memory = memory
        self.llm = llm
        self.agent_id = agent_id
        self.planner_model = planner_model
        self.evaluator_model = evaluator_model or planner_model
        
        self._state: Dict[str, SubtaskState] = {}
        self._execution_contexts: Dict[str, ExecutionContext] = {}
        
        self.metrics = defaultdict(int)
        
    def run_step(
        self,
        task: TaskSpec,
        subtask: Subtask,
        timebox: timedelta,
        budget: Optional[StepBudget] = None,
    ) -> AgentUpdate:
        """Execute one research step with enhanced intelligence"""
        key = f"{task.id}:{subtask.id}"
        state = self._state.setdefault(key, SubtaskState())
        
        # Extract execution context if provided
        exec_context = self._extract_execution_context(subtask)
        if exec_context:
            self._execution_contexts[key] = exec_context
            
        # Adapt budget based on strategy and context
        budget = self._adapt_budget(budget, subtask.strategy, state)
        
        # Initialize tracking
        start_time = datetime.utcnow()
        actions_taken = 0
        findings_batch = []
        sources_batch = []
        blockers = []
        insights = []
        
        # Main execution loop
        while actions_taken < budget.max_actions:
            # Check time limit
            if budget.max_seconds and (datetime.utcnow() - start_time).seconds > budget.max_seconds:
                blockers.append("Time limit reached")
                break
                
            # Get next action
            action = self._plan_next_action(task, subtask, state, budget)
            state.decision_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "action": action.action,
                "rationale": action.rationale
            })
            
            # Execute action
            if action.action == "search":
                if state.search_count >= (budget.max_searches or 15):
                    blockers.append("Search limit reached")
                    action.action = "analyze"  # Fallback to analysis
                else:
                    results = self._execute_search(action.query, state)
                    if results:
                        findings_batch.append(f"Found {len(results)} sources for: {action.query}")
                        sources_batch.extend([e.url for e in results])
                    else:
                        blockers.append(f"No results for: {action.query}")
                        
            elif action.action == "analyze":
                analysis = self._deep_analyze(state, action.focus)
                if analysis:
                    findings_batch.extend(analysis["findings"])
                    insights.extend(analysis["insights"])
                    
            elif action.action == "synthesize":
                synthesis = self._synthesize_findings(task, subtask, state)
                findings_batch.append(synthesis["summary"])
                state.synthesis_count += 1
                
            elif action.action == "evaluate":
                evaluation = self._evaluate_completeness(subtask, state)
                if evaluation["complete"]:
                    findings_batch.append("Acceptance criteria satisfied")
                else:
                    state.gaps = evaluation["gaps"]
                    
            actions_taken += 1
            state.action_count += 1
            
            # Checkpoint periodically
            if state.action_count % CHECKPOINT_FREQ == 0:
                self._create_checkpoint(task, subtask, state)
                
        # Assess overall progress
        progress = self._assess_progress(state, subtask)
        confidence = self._calculate_confidence(state)
        
        # Generate methodology notes
        methodology = self._generate_methodology_notes(state, actions_taken)
        
        # Propose next steps
        next_steps = self._propose_next_steps(state, subtask)
        
        # Create update
        update = AgentUpdate(
            timestamp=datetime.utcnow(),
            task_id=task.id,
            subtask_id=subtask.id,
            agent_id=self.agent_id,
            progress=progress,
            findings="\n\n".join(findings_batch)[:2000],
            sources=list(dict.fromkeys(sources_batch))[:20],
            blockers=blockers,
            proposed_next=next_steps,
            confidence=confidence,
            key_insights=insights[:5],
            methodology_notes=methodology
        )
        
        self._update_metrics(state)
        return update
        
    def build_markdown_report(self, task: TaskSpec, subtask: Subtask) -> str:
        """Generate professional research memorandum"""
        key = f"{task.id}:{subtask.id}"
        state = self._state.get(key)
        if not state or not state.evidence_store:
            return f"# Research Memorandum: {task.name}\n\n## {subtask.title}\n\n*Insufficient data for comprehensive report.*"
            
        # Prepare evidence for synthesis
        evidence_map = self._prepare_evidence_map(state)
        key_findings = self._extract_key_findings(state)
        
        # Generate professional synthesis
        synthesis_prompt = PROFESSIONAL_SYNTHESIS_PROMPT.format(
            task_name=task.name,
            subtask_title=subtask.title,
            questions="\n".join(subtask.research_questions) if hasattr(subtask, 'research_questions') else subtask.title,
            findings=json.dumps(key_findings, indent=2),
            evidence_map=json.dumps(evidence_map, indent=2)
        )
        
        report = self.llm.chat([
            {"role": "system", "content": "You are a professional research analyst. Be objective, critical, and evidence-based."},
            {"role": "user", "content": synthesis_prompt}
        ])
        
        # Add footnotes
        footnotes = self._generate_footnotes(state)
        
        # Add metadata section
        metadata = self._generate_report_metadata(state)
        
        return f"{report}\n\n---\n\n{metadata}\n\n{footnotes}"
        
    # ---------- Core Intelligence Methods ----------
    
    def _plan_next_action(
        self,
        task: TaskSpec,
        subtask: Subtask,
        state: SubtaskState,
        budget: StepBudget
    ) -> ResearchAction:
        """Intelligently plan next research action"""
        # Build comprehensive context
        context = {
            "task": task.name,
            "subtask": subtask.title,
            "strategy": subtask.strategy,
            "research_questions": getattr(subtask, 'research_questions', []),
            "acceptance_criteria": subtask.acceptance_criteria,
            "evidence_count": len(state.evidence_store),
            "quality_scores": list(state.quality_history)[-5:],
            "recent_actions": list(state.decision_history)[-5:],
            "gaps": state.gaps,
            "active_thread": state.threads.get(state.active_thread) if state.active_thread else None,
            "failed_queries": list(state.failed_queries)[-5:],
            "execution_context": self._execution_contexts.get(f"{task.id}:{subtask.id}", {})
        }
        
        planner_prompt = ADAPTIVE_PLANNER_PROMPT.format(
            context=json.dumps(context, indent=2, default=str)
        )
        
        structured_planner = self.planner_model.with_structured_output(ResearchAction)
        action = structured_planner.invoke([
            SystemMessage(content=planner_prompt),
            HumanMessage(content="Plan the next research action.")
        ])
        
        self.metrics["llm_calls"] += 1
        return action
        
    def _execute_search(self, query: str, state: SubtaskState) -> List[Evidence]:
        """Execute search with enhanced processing"""
        if not query or query in state.queries_attempted:
            return []
            
        state.queries_attempted.add(query)
        state.search_count += 1
        self.metrics["searches"] += 1
        
        # Optimize query if needed
        optimized_query = self._optimize_query(query, state)
        
        # Execute search
        results = self._search_web(optimized_query)
        if not results:
            state.failed_queries.add(query)
            return []
            
        state.successful_queries[query] = len(results)
        
        # Process and evaluate each result
        evidence_list = []
        for result in results:
            evidence = self._process_search_result(result, query)
            if evidence:
                evaluation = self._evaluate_evidence(evidence, state)
                if evaluation.relevance >= MIN_EVIDENCE_SCORE:
                    evidence.metadata["evaluation"] = evaluation.model_dump()
                    state.evidence_store[evidence.url] = evidence
                    state.evidence_scores[evidence.url] = evaluation.relevance
                    evidence_list.append(evidence)
                    
                    # Extract connections
                    self._extract_evidence_connections(evidence, state)
                    
        return evidence_list
        
    def _deep_analyze(self, state: SubtaskState, focus: Optional[str]) -> Dict[str, Any]:
        """Perform deep analysis of collected evidence"""
        if not state.evidence_store:
            return {"findings": [], "insights": []}
            
        state.analysis_count += 1
        self.metrics["analyses"] += 1
        
        # Select high-quality evidence for analysis
        top_evidence = sorted(
            state.evidence_store.values(),
            key=lambda e: state.evidence_scores.get(e.url, 0),
            reverse=True
        )[:10]
        
        # Prepare evidence summary
        evidence_summary = "\n\n".join([
            f"Source: {e.title}\nKey Facts: {json.dumps(e.extracted_facts)}\nSummary: {e.summary}"
            for e in top_evidence
        ])
        
        # Get research questions
        questions = getattr(
            self._get_current_subtask(),
            'research_questions',
            ["What are the key findings?"]
        )
        
        # Perform critical analysis
        analysis_prompt = CRITICAL_ANALYZER_PROMPT.format(
            questions="\n".join(questions),
            evidence=evidence_summary
        )
        
        analysis_result = self.llm.chat([
            {"role": "system", "content": "You are a critical research analyst. Be rigorous and objective."},
            {"role": "user", "content": analysis_prompt}
        ])
        
        self.metrics["llm_calls"] += 1
        
        # Parse analysis results
        findings = []
        insights = []
        
        # Extract patterns and insights
        for line in analysis_result.split("\n"):
            if "PATTERN:" in line or "FINDING:" in line:
                findings.append(line.strip())
            elif "INSIGHT:" in line or "CONCLUSION:" in line:
                insights.append(line.strip())
                
        # Update state
        state.key_findings.extend([
            {"finding": f, "evidence_urls": [e.url for e in top_evidence[:3]], "timestamp": datetime.utcnow().isoformat()}
            for f in findings
        ])
        
        return {
            "findings": findings,
            "insights": insights
        }
        
    def _synthesize_findings(self, task: TaskSpec, subtask: Subtask, state: SubtaskState) -> Dict[str, Any]:
        """Synthesize current findings into coherent narrative"""
        if not state.key_findings:
            return {"summary": "Insufficient findings for synthesis."}
            
        # Group findings by theme
        themed_findings = self._group_findings_by_theme(state.key_findings)
        
        # Create synthesis
        synthesis = []
        for theme, findings in themed_findings.items():
            theme_summary = f"**{theme}**: " + " ".join([f["finding"] for f in findings[:3]])
            synthesis.append(theme_summary)
            
        return {
            "summary": "\n\n".join(synthesis),
            "themes": list(themed_findings.keys())
        }
        
    def _evaluate_completeness(self, subtask: Subtask, state: SubtaskState) -> Dict[str, Any]:
        """Evaluate if research satisfies acceptance criteria"""
        criteria = subtask.acceptance_criteria
        if not criteria:
            return {"complete": len(state.evidence_store) >= 5, "gaps": []}
            
        satisfied = []
        gaps = []
        
        for criterion in criteria:
            # Check if we have evidence addressing this criterion
            relevant_evidence = [
                e for e in state.evidence_store.values()
                if any(criterion.lower() in fact.lower() for fact in e.extracted_facts)
            ]
            
            if relevant_evidence:
                satisfied.append(criterion)
            else:
                gaps.append(criterion)
                
        return {
            "complete": len(gaps) == 0,
            "satisfied": satisfied,
            "gaps": gaps,
            "completion_ratio": len(satisfied) / len(criteria) if criteria else 0
        }
        
    # ---------- Helper Methods ----------
    
    def _extract_execution_context(self, subtask: Subtask) -> Optional[ExecutionContext]:
        """Extract execution context from subtask metadata"""
        if hasattr(subtask, 'metadata') and 'execution_context' in subtask.metadata:
            context_data = subtask.metadata['execution_context']
            if isinstance(context_data, dict):
                return ExecutionContext(**context_data)
        return None
        
    def _adapt_budget(
        self,
        budget: Optional[StepBudget],
        strategy: ResearchStrategy,
        state: SubtaskState
    ) -> StepBudget:
        """Adapt budget based on strategy and progress"""
        base_budget = budget or StepBudget()
        
        # Adjust based on strategy
        if strategy == ResearchStrategy.DEEP_DIVE:
            base_budget.max_actions = min(20, base_budget.max_actions * 2)
            base_budget.max_searches = min(25, (base_budget.max_searches or 10) * 2)
        elif strategy == ResearchStrategy.FACT_FINDING:
            base_budget.max_searches = min(20, (base_budget.max_searches or 10) + 10)
        elif strategy == ResearchStrategy.SYSTEMATIC_REVIEW:
            base_budget.max_summaries = min(10, (base_budget.max_summaries or 3) * 3)
            
        # Adjust based on progress
        if state.action_count > 20 and not state.key_findings:
            # Struggling - increase budget
            base_budget.max_actions = min(15, base_budget.max_actions + 5)
            
        return base_budget
        
    def _search_web(self, query: str) -> List[Dict[str, str]]:
        """Execute web search"""
        params = {
            "q": query,
            "format": "json",
            "language": "en",
            "safesearch": 1,
            "engines": "google,bing,duckduckgo",
            "categories": "general,science,news",
            "time_range": "year",  # Prefer recent content
        }
        
        try:
            resp = requests.get(
                SEARXNG_URL,
                params=params,
                timeout=20,
                headers={"User-Agent": "KestrelResearch/1.0"}
            )
            resp.raise_for_status()
            data = resp.json()
            
            results = []
            for r in data.get("results", [])[:SEARCH_RESULTS]:
                if r.get("url") and not any(
                    skip in r["url"] for skip in ["youtube.com", "tiktok.com", "instagram.com"]
                ):
                    results.append({
                        "title": r.get("title", "")[:200],
                        "url": r.get("url"),
                        "snippet": r.get("content", "")[:500],
                        "engine": r.get("engine", "unknown")
                    })
                    
            return results
            
        except Exception as e:
            if DEBUG:
                print(f"Search error for '{query}': {e}")
            return []
            
    def _process_search_result(self, result: Dict[str, str], query: str) -> Optional[Evidence]:
        """Process search result into evidence"""
        try:
            # Fetch and extract content
            content = self._extract_content(result["url"])
            if not content or len(content) < 100:
                return None
                
            # Extract key facts
            facts = self._extract_facts(content, query)
            
            # Create evidence object
            evidence = Evidence(
                url=result["url"],
                title=result["title"],
                content=content[:MAX_SNIPPET_LENGTH],
                summary=result["snippet"],
                extracted_facts=facts,
                relevance_score=0.0,  # Will be set by evaluator
                credibility_score=self._assess_source_credibility(result["url"]),
                timestamp=datetime.utcnow(),
                metadata={
                    "query": query,
                    "engine": result.get("engine", "unknown")
                }
            )
            
            return evidence
            
        except Exception as e:
            if DEBUG:
                print(f"Error processing {result.get('url', 'unknown')}: {e}")
            return None
            
    def _extract_content(self, url: str) -> str:
        """Extract clean text content from URL"""
        try:
            resp = requests.get(
                url,
                timeout=15,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; KestrelBot/1.0)",
                    "Accept": "text/html,application/xhtml+xml"
                }
            )
            resp.raise_for_status()
            
            soup = BeautifulSoup(resp.text[:FETCH_BYTES], "html.parser")
            
            # Remove unwanted elements
            for tag in soup(["script", "style", "nav", "footer", "aside"]):
                tag.decompose()
                
            # Extract main content
            main_content = soup.find("main") or soup.find("article") or soup.find("body")
            if not main_content:
                return ""
                
            # Clean and extract text
            text = " ".join(main_content.get_text(" ", strip=True).split())
            
            # Basic cleaning
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'Cookie Policy.*?Accept', '', text, flags=re.IGNORECASE)
            
            return text
            
        except Exception as e:
            if DEBUG:
                print(f"Content extraction error for {url}: {e}")
            return ""
            
    def _extract_facts(self, content: str, query: str) -> List[str]:
        """Extract concrete facts from content"""
        facts = []
        
        # Look for sentences with numbers, dates, or specific claims
        sentences = content.split(". ")
        query_terms = query.lower().split()
        
        for sent in sentences[:50]:  # Limit processing
            sent = sent.strip()
            if len(sent) < 20 or len(sent) > 300:
                continue
                
            # Check if relevant to query
            if not any(term in sent.lower() for term in query_terms):
                continue
                
            # Check for concrete information
            has_number = bool(re.search(r'\d+', sent))
            has_date = bool(re.search(r'\b(19|20)\d{2}\b|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', sent))
            has_percent = '%' in sent
            has_quote = '"' in sent or '"' in sent
            
            if has_number or has_date or has_percent or has_quote:
                facts.append(sent + ".")
                
            if len(facts) >= 10:
                break
                
        return facts
        
    def _assess_source_credibility(self, url: str) -> float:
        """Assess credibility based on source"""
        high_credibility = ['.gov', '.edu', 'nature.com', 'science.org', 'ieee.org', '.ac.uk']
        medium_credibility = ['.org', 'wikipedia.org', 'reuters.com', 'apnews.com']
        
        for pattern in high_credibility:
            if pattern in url:
                return 8.0
                
        for pattern in medium_credibility:
            if pattern in url:
                return 6.0
                
        return 4.0  # Default
        
    def _evaluate_evidence(self, evidence: Evidence, state: SubtaskState) -> EvidenceEvaluation:
        """Evaluate evidence quality"""
        context = {
            "research_focus": state.working_context,
            "existing_facts": [f for findings in state.key_findings for f in findings.get("finding", "")][:10]
        }
        
        eval_prompt = EVIDENCE_EVALUATOR_PROMPT.format(
            evidence=json.dumps({
                "title": evidence.title,
                "facts": evidence.extracted_facts[:5],
                "summary": evidence.summary
            }, indent=2),
            context=json.dumps(context, indent=2)
        )
        
        # Use evaluator model if available
        evaluator = self.evaluator_model.with_structured_output(EvidenceEvaluation)
        evaluation = evaluator.invoke([
            SystemMessage(content="Evaluate evidence critically and objectively."),
            HumanMessage(content=eval_prompt)
        ])
        
        self.metrics["llm_calls"] += 1
        
        # Update evidence relevance score
        evidence.relevance_score = evaluation.relevance
        
        return evaluation
        
    def _optimize_query(self, query: str, state: SubtaskState) -> str:
        """Optimize search query based on context"""
        if len(state.failed_queries) < 3:
            return query
            
        optimization_prompt = QUERY_OPTIMIZER_PROMPT.format(
            goal=query,
            previous=list(state.queries_attempted)[-5:],
            knowledge=[f["finding"] for f in state.key_findings[-3:]],
            failed=list(state.failed_queries)[-3:]
        )
        
        optimized = self.llm.chat([
            {"role": "system", "content": "You are a search query optimization expert."},
            {"role": "user", "content": optimization_prompt}
        ])
        
        self.metrics["llm_calls"] += 1
        return optimized.strip()
        
    def _extract_evidence_connections(self, evidence: Evidence, state: SubtaskState) -> None:
        """Extract connections between evidence pieces"""
        # Simple entity extraction for connections
        entities = set()
        for fact in evidence.extracted_facts:
            # Extract capitalized phrases as potential entities
            entities.update(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', fact))
            
        # Link to other evidence mentioning same entities
        for other_url, other_evidence in state.evidence_store.items():
            if other_url == evidence.url:
                continue
                
            other_entities = set()
            for fact in other_evidence.extracted_facts:
                other_entities.update(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', fact))
                
            # If significant overlap, create connection
            if len(entities & other_entities) >= 2:
                state.evidence_graph[evidence.url].append(other_url)
                state.evidence_graph[other_url].append(evidence.url)
                
    def _assess_progress(self, state: SubtaskState, subtask: Subtask) -> str:
        """Assess research progress"""
        # Check acceptance criteria satisfaction
        evaluation = self._evaluate_completeness(subtask, state)
        
        if evaluation["complete"]:
            return "complete"
        elif evaluation["completion_ratio"] > 0.7:
            return "good"
        elif state.evidence_store and state.key_findings:
            return "partial"
        else:
            return "none"
            
    def _calculate_confidence(self, state: SubtaskState) -> float:
        """Calculate confidence in findings"""
        if not state.evidence_store:
            return 0.1
            
        # Factors affecting confidence
        evidence_quality = sum(state.evidence_scores.values()) / len(state.evidence_scores) if state.evidence_scores else 0
        source_diversity = len(set(e.metadata.get("engine", "unknown") for e in state.evidence_store.values()))
        finding_consistency = self._assess_finding_consistency(state)
        
        confidence = (
            evidence_quality / 10 * 0.4 +  # Quality of evidence
            min(source_diversity / 3, 1.0) * 0.3 +  # Source diversity
            finding_consistency * 0.3  # Consistency
        )
        
        return round(min(max(confidence, 0.1), 0.95), 2)
        
    def _assess_finding_consistency(self, state: SubtaskState) -> float:
        """Assess consistency of findings"""
        if not state.contradictions:
            return 1.0
        if not state.key_findings:
            return 0.5
            
        ratio = len(state.contradictions) / len(state.key_findings)
        return max(1.0 - ratio, 0.3)
        
    def _propose_next_steps(self, state: SubtaskState, subtask: Subtask) -> List[str]:
        """Propose concrete next steps"""
        steps = []
        
        # Based on gaps
        for gap in state.gaps[:3]:
            steps.append(f"Search for evidence on: {gap}")
            
        # Based on low-confidence areas
        low_confidence_themes = [
            theme for theme, thread in state.threads.items()
            if thread.confidence < 0.6 and thread.status == "active"
        ]
        for theme in low_confidence_themes[:2]:
            steps.append(f"Deepen investigation of: {theme}")
            
        # Based on promising leads
        if state.evidence_store:
            high_value_evidence = sorted(
                state.evidence_store.values(),
                key=lambda e: len(e.metadata.get("evaluation", {}).get("follow_up_questions", [])),
                reverse=True
            )[:2]
            for evidence in high_value_evidence:
                if evidence.metadata.get("evaluation", {}).get("follow_up_questions"):
                    steps.append(evidence.metadata["evaluation"]["follow_up_questions"][0])
                    
        return steps[:6]
        
    def _generate_methodology_notes(self, state: SubtaskState, actions: int) -> str:
        """Generate notes on research methodology"""
        notes = []
        
        # Search strategy
        if state.successful_queries:
            notes.append(f"Conducted {len(state.successful_queries)} successful searches across {actions} actions")
            
        # Evidence quality
        if state.evidence_scores:
            avg_score = sum(state.evidence_scores.values()) / len(state.evidence_scores)
            notes.append(f"Average evidence quality: {avg_score:.1f}/10")
            
        # Coverage
        if hasattr(self._get_current_subtask(), 'acceptance_criteria'):
            evaluation = self._evaluate_completeness(self._get_current_subtask(), state)
            notes.append(f"Criteria coverage: {evaluation['completion_ratio']*100:.0f}%")
            
        return ". ".join(notes)
        
    def _prepare_evidence_map(self, state: SubtaskState) -> Dict[str, Dict[str, Any]]:
        """Prepare evidence for report generation"""
        evidence_map = {}
        
        # Sort evidence by relevance
        sorted_evidence = sorted(
            state.evidence_store.items(),
            key=lambda x: state.evidence_scores.get(x[0], 0),
            reverse=True
        )[:15]
        
        for idx, (url, evidence) in enumerate(sorted_evidence, 1):
            evidence_map[f"source_{idx}"] = {
                "url": url,
                "title": evidence.title,
                "credibility": evidence.credibility_score,
                "key_facts": evidence.extracted_facts[:3],
                "relevance": evidence.relevance_score
            }
            
        return evidence_map
        
    def _extract_key_findings(self, state: SubtaskState) -> List[Dict[str, Any]]:
        """Extract and organize key findings"""
        # Group findings by theme
        themed_findings = self._group_findings_by_theme(state.key_findings)
        
        # Format for report
        formatted_findings = []
        for theme, findings in themed_findings.items():
            formatted_findings.append({
                "theme": theme,
                "findings": [f["finding"] for f in findings[:5]],
                "evidence_count": len(findings),
                "confidence": self._calculate_theme_confidence(findings, state)
            })
            
        return formatted_findings
        
    def _group_findings_by_theme(self, findings: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group findings by common themes"""
        # Simple keyword-based grouping
        themes = defaultdict(list)
        
        for finding in findings:
            finding_text = finding.get("finding", "").lower()
            
            # Assign to themes based on keywords
            if any(word in finding_text for word in ["cost", "price", "expense", "budget", "financial"]):
                themes["Financial Aspects"].append(finding)
            elif any(word in finding_text for word in ["impact", "effect", "result", "outcome"]):
                themes["Impacts and Outcomes"].append(finding)
            elif any(word in finding_text for word in ["method", "approach", "technique", "process"]):
                themes["Methodology and Approaches"].append(finding)
            elif any(word in finding_text for word in ["challenge", "problem", "issue", "barrier"]):
                themes["Challenges and Barriers"].append(finding)
            elif any(word in finding_text for word in ["benefit", "advantage", "opportunity", "strength"]):
                themes["Benefits and Opportunities"].append(finding)
            else:
                themes["General Findings"].append(finding)
                
        return dict(themes)
        
    def _calculate_theme_confidence(self, findings: List[Dict[str, Any]], state: SubtaskState) -> float:
        """Calculate confidence for a theme"""
        if not findings:
            return 0.0
            
        # Get evidence URLs for these findings
        evidence_urls = set()
        for f in findings:
            evidence_urls.update(f.get("evidence_urls", []))
            
        # Calculate average evidence score
        scores = [state.evidence_scores.get(url, 0) for url in evidence_urls if url in state.evidence_scores]
        if not scores:
            return 0.5
            
        return min(sum(scores) / len(scores) / 10, 0.95)
        
    def _generate_footnotes(self, state: SubtaskState) -> str:
        """Generate footnotes section for report"""
        footnotes = []
        
        # Get unique sources used
        used_sources = sorted(
            state.evidence_store.items(),
            key=lambda x: state.evidence_scores.get(x[0], 0),
            reverse=True
        )[:20]
        
        for idx, (url, evidence) in enumerate(used_sources, 1):
            footnotes.append(
                f"[^{idx}]: {evidence.title} - {url} "
                f"(Relevance: {evidence.relevance_score:.1f}/10, "
                f"Credibility: {evidence.credibility_score:.1f}/10)"
            )
            
        return "## Sources\n\n" + "\n".join(footnotes)
        
    def _generate_report_metadata(self, state: SubtaskState) -> str:
        """Generate metadata section for report"""
        metadata = [
            "## Research Metadata",
            f"- Total searches conducted: {state.search_count}",
            f"- Evidence pieces evaluated: {len(state.evidence_store)}",
            f"- Key findings identified: {len(state.key_findings)}",
            f"- Analysis iterations: {state.analysis_count}",
            f"- Confidence level: {self._calculate_confidence(state)*100:.0f}%",
            f"- Research threads explored: {len(state.threads)}"
        ]
        
        if state.contradictions:
            metadata.append(f"- Contradictions found: {len(state.contradictions)}")
            
        return "\n".join(metadata)
        
    def _create_checkpoint(self, task: TaskSpec, subtask: Subtask, state: SubtaskState) -> None:
        """Create research checkpoint"""
        checkpoint = {
            "timestamp": datetime.utcnow().isoformat(),
            "progress": {
                "evidence_collected": len(state.evidence_store),
                "findings": len(state.key_findings),
                "queries": len(state.queries_attempted)
            },
            "quality_metrics": {
                "avg_evidence_score": sum(state.evidence_scores.values()) / len(state.evidence_scores) if state.evidence_scores else 0,
                "confidence": self._calculate_confidence(state)
            },
            "focus_areas": list(state.threads.keys())[:5]
        }
        
        state.checkpoints.append(json.dumps(checkpoint))
        state.last_checkpoint = checkpoint["timestamp"]
        
        # Save to memory
        self._add_to_memory(task, subtask, checkpoint, "checkpoint")
        
    def _add_to_memory(self, task: TaskSpec, subtask: Subtask, content: Any, doc_type: str) -> None:
        """Add content to vector store"""
        doc_id = f"{task.id}-{subtask.id}-{doc_type}-{uuid4().hex[:8]}"
        metadata = {
            "task_id": task.id,
            "task": task.name,
            "subtask_id": subtask.id,
            "subtask": subtask.title,
            "type": doc_type,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if isinstance(content, dict):
            content_str = json.dumps(content, indent=2)
        else:
            content_str = str(content)
            
        self.memory.add(doc_id, content_str, metadata)
        
    def _update_metrics(self, state: SubtaskState) -> None:
        """Update performance metrics"""
        self.metrics["total_actions"] = state.action_count
        self.metrics["total_evidence"] = len(state.evidence_store)
        self.metrics["total_findings"] = len(state.key_findings)
        
    def _get_current_subtask(self) -> Optional[Subtask]:
        """Get current subtask from context"""
        # This is a placeholder - in practice would get from current execution
        return None
        
    def get_metrics(self) -> Dict[str, int]:
        """Get performance metrics"""
        return dict(self.metrics)