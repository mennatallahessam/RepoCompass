#!/usr/bin/env python3
"""
Interactive stage-by-stage verification suite for the RepoCompass agent pipeline.
Usage: python tests/test_agents.py <github_repo_url>
"""
import json
import sys
import traceback
from typing import Any, Dict

from src.agents.context_agent import context_agent
from src.agents.mentor_agent import mentor_agent
from src.agents.navigator_agent import navigator_agent
from src.agents.orchestrator_agent import orchestrator_agent
from src.agents.visualizer_agent import visualizer_agent
from src.github_client import GitHubClient
from src.state import build_initial_state


def print_separator(heading: str) -> None:
    """Print a formatted section divider."""
    print(f"\n{'═' * 78}\n  {heading}\n{'═' * 78}\n")


def apply_updates(state: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    """Merge partial agent state updates (replicating LangGraph reducer behavior)."""
    for field, val in delta.items():
        if field in ("messages", "errors") and isinstance(val, list):
            state[field] = state.get(field, []) + val
        else:
            state[field] = val
    return state


def test_navigator(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Stage 1: Structure Mapper."""
    print_separator("STAGE 1: STRUCTURE MAPPER (NAVIGATOR)")
    delta = navigator_agent(state)
    state = apply_updates(state, delta)
    if state.get("navigator_map"):
        print(json.dumps(state["navigator_map"], indent=2))
    return state


def test_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Stage 2: Code Inspector."""
    print_separator("STAGE 2: CODE INSPECTOR (CONTEXT)")
    delta = context_agent(state)
    state = apply_updates(state, delta)
    if state.get("context_summary"):
        print(state["context_summary"])
    return state


def test_mentor(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Stage 3: Onboarding Architect."""
    print_separator("STAGE 3: ONBOARDING ARCHITECT (MENTOR)")
    delta = mentor_agent(state)
    state = apply_updates(state, delta)
    if state.get("mentor_guide"):
        print(state["mentor_guide"])
    return state


def test_visualizer(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Stage 4: Diagram Synthesizer."""
    print_separator("STAGE 4: DIAGRAM SYNTHESIZER (VISUALIZER)")
    delta = visualizer_agent(state)
    state = apply_updates(state, delta)
    if state.get("visualization"):
        print(state["visualization"])
    return state


def test_orchestrator(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Stage 5: Report Compiler."""
    print_separator("STAGE 5: REPORT COMPILER (ORCHESTRATOR)")
    delta = orchestrator_agent(state)
    state = apply_updates(state, delta)
    if state.get("final_report"):
        print(state["final_report"])
    return state


def main() -> None:
    """Run all 5 agent stages sequentially."""
    if len(sys.argv) < 2:
        print("Usage: python tests/test_agents.py <github_repo_url>")
        sys.exit(1)

    repo_url = sys.argv[1]
    client = GitHubClient()
    owner, repo_name = client.parse_repo_url(repo_url)
    metadata = client.get_repo_metadata(owner, repo_name)

    clone_dir = client.clone_repo(repo_url)
    try:
        file_tree = client.walk_local_repo(clone_dir)
        code_samples = client.read_all_source_files(clone_dir, file_tree)
        readme_content = client.read_local_readme(clone_dir)
        config_files = client.read_local_config_files(clone_dir, file_tree)
    finally:
        client.cleanup_clone(clone_dir)

    state = build_initial_state(
        repo_url=repo_url,
        owner=owner,
        repo_name=repo_name,
        metadata=metadata,
        file_tree=file_tree,
        code_samples=code_samples,
        readme_content=readme_content,
        config_files=config_files,
        recent_commits=[],
        pull_requests=[],
    )

    try:
        state = test_navigator(state)
        state = test_context(state)
        state = test_mentor(state)
        state = test_visualizer(state)
        state = test_orchestrator(state)
        print_separator("VERIFICATION COMPLETE")
        for msg in state.get("messages", []):
            print(f"  ✓ {msg}")
    except Exception as exc:
        print(f"Verification failed: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
