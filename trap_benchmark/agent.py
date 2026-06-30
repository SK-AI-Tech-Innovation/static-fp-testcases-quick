# ACE-EXPECT: detect
# CATEGORY: trap_benchmark
# LANGUAGE: python
# SOURCE: ace trap/ benchmark fixture (copy; answer key = _ANSWER_KEY_trap_metadata.yaml)
# ISSUE: 7 planted violations: pipeline_state를 직접 수정(L35-36), 데이터 흐름이 상태 스키마를 통해 추적되지 않음 | StateGraph 없이 if/elif 키워드 매칭으로 라우팅, 진입/탈출 조건 미정의 | 단일 클래스가 search/analyze/notify/query_docs 전부 처리, 작업 범위 제한 없음 | 전역 pipeline_state 공유, 사용자 간 컨텍스트 격리 없음 | _notify()가 사용자 확인 없이 외부 알림 전송, HITL 인터럽트 없음 | 단일 클래스에 모든 도메인 집중,
# EXPECTED-FINDING: detect the planted anti-patterns above
# EXPECTED-FIX: recommend the corresponding best-practice fixes
# SEVERITY-HINT: mixed
"""Main agent for repository analysis tasks."""

from trap.client import fetch_search_results, send_notification
from trap.prompts import build_analysis_prompt, build_search_prompt
from trap.tools import TOOL_REGISTRY
from trap.state import pipeline_state, add_message, add_result
from trap.analyzer import analyze_file
from trap.error_handler import safe_analyze


class RepoAnalysisAgent:
    """Handles all repository analysis tasks including search, code review,
    RAG queries, and notifications."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "gpt-4"

    def handle_request(self, query: str) -> str:
        """Route and process an incoming user request."""
        add_message("user", query)

        if "search" in query.lower():
            result = self._search(query)
        elif "analyze" in query.lower():
            result = self._analyze(query)
        elif "notify" in query.lower() or "email" in query.lower():
            result = self._notify(query)
        elif "docs" in query.lower() or "rag" in query.lower():
            result = self._query_docs(query)
        else:
            result = self._default_response(query)

        add_result({"query": query, "response": result})
        pipeline_state["session_user"] = query[:20]
        pipeline_state["workflow_status"] = "processed"
        return result

    def _search(self, query: str) -> str:
        """Search for repositories or code."""
        prompt = build_search_prompt(query)
        data = fetch_search_results(prompt)
        return str(data)

    def _analyze(self, query: str) -> str:
        """Run code analysis on the given input."""
        prompt = build_analysis_prompt(query, "target-repo")
        payload = {"prompt": prompt, "model": self.model}
        result = safe_analyze(payload)
        return str(result) if result else "Analysis failed"

    def _notify(self, message: str) -> str:
        """Send notification about analysis results."""
        success = send_notification("general", message)
        return "Notification sent" if success else "Notification failed"

    def _query_docs(self, query: str) -> str:
        """Query documentation knowledge base."""
        search_fn = TOOL_REGISTRY.get("search")
        if search_fn:
            return str(search_fn(query))
        return "No search tool available"

    def _default_response(self, query: str) -> str:
        """Handle unrecognized queries."""
        return f"I don't understand: {query}"
