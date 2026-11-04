"""
Pipeline state definitions for the RepoCompass multi-agent repository analyzer.
"""
from operator import add
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict):
    """Shared state dictionary passed across all stages of the LangGraph pipeline."""

    # Target repository identifiers
    repo_url: str
    owner: str
    repo_name: str

    # Extracted repository data (GitHub REST API + shallow local clone)
    metadata: Dict[str, Any]
    file_tree: List[Dict[str, Any]]
    code_samples: Dict[str, str]
    readme_content: Optional[str]
    config_files: Dict[str, str]
    recent_commits: List[Dict[str, Any]]
    pull_requests: List[Dict[str, Any]]

    # Outputs produced by each specialist agent
    navigator_map: Optional[Dict[str, Any]]
    context_output: Optional[Dict[str, Any]]
    context_summary: Optional[str]
    mentor_guide: Optional[str]
    visualization: Optional[str]
    final_report: Optional[str]

    # Pipeline telemetry & diagnostics (accumulated via operator.add)
    messages: Annotated[List[str], add]
    errors: Annotated[List[str], add]


def build_initial_state(
    repo_url: str,
    owner: str,
    repo_name: str,
    metadata: Dict[str, Any],
    file_tree: List[Dict[str, Any]],
    code_samples: Dict[str, str],
    readme_content: Optional[str],
    config_files: Dict[str, str],
    recent_commits: List[Dict[str, Any]],
    pull_requests: List[Dict[str, Any]],
) -> AgentState:
    """Construct a clean initial state dictionary for the LangGraph workflow."""
    return {
        "repo_url": repo_url,
        "owner": owner,
        "repo_name": repo_name,
        "metadata": metadata,
        "file_tree": file_tree,
        "code_samples": code_samples,
        "readme_content": readme_content,
        "config_files": config_files,
        "recent_commits": recent_commits,
        "pull_requests": pull_requests,
        "navigator_map": None,
        "context_output": None,
        "context_summary": None,
        "mentor_guide": None,
        "visualization": None,
        "final_report": None,
        "messages": [],
        "errors": [],
    }
