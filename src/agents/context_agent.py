"""
Stage 2: Code Inspector (Context Agent)
Performs deep static inspection of prioritized source files to identify key functions,
classes, frameworks, design patterns, API routes, data schemas, and complexity metrics.
"""
from typing import Any, Dict
from src.state import AgentState
from src.utils import create_chat_model, extract_json, invoke_llm_text

llm = create_chat_model(temperature=0.1)


def _select_priority_files(
    code_samples: Dict[str, str], navigator_map: Dict[str, Any], max_files: int = 50
) -> Dict[str, str]:
    """Rank and filter source files so entry points and core modules are inspected first."""
    entry_set = set(navigator_map.get("entry_points", []))
    core_prefixes = [m.rstrip("/") for m in navigator_map.get("core_modules", []) if m]

    high_priority: Dict[str, str] = {}
    secondary: Dict[str, str] = {}

    for rel_path, source_text in code_samples.items():
        if rel_path in entry_set or any(rel_path.startswith(pfx) for pfx in core_prefixes):
            high_priority[rel_path] = source_text
        else:
            secondary[rel_path] = source_text

    chosen = dict(list(high_priority.items())[:max_files])
    slots_left = max_files - len(chosen)
    if slots_left > 0:
        chosen.update(dict(list(secondary.items())[:slots_left]))

    return chosen


def _format_context_summary(
    inspected_count: int, total_count: int, payload: Dict[str, Any]
) -> str:
    """Build a concise human-readable digest of the code inspection results."""
    tech_list = ", ".join(payload.get("technologies", [])[:8]) or "None detected"
    pattern_list = ", ".join(payload.get("patterns", [])[:5]) or "Standard modular structure"
    fn_total = len(payload.get("key_functions", []))
    cls_total = len(payload.get("key_classes", []))
    ep_total = len(payload.get("api_endpoints", []))
    dm_total = len(payload.get("data_models", []))
    score = float(payload.get("complexity_score", 0.0))

    return (
        f"Inspected {inspected_count} of {total_count} source files.\n"
        f"Technologies: {tech_list}\n"
        f"Key Functions: {fn_total}\n"
        f"Key Classes: {cls_total}\n"
        f"Patterns: {pattern_list}\n"
        f"API Endpoints: {ep_total}\n"
        f"Data Models: {dm_total}\n"
        f"Complexity: {score:.1f}/1.0"
    )


def context_agent(state: AgentState) -> Dict[str, Any]:
    """
    Inspect prioritized repository source files and populate `context_output` and `context_summary`.
    """
    code_samples = state.get("code_samples", {})
    config_files = state.get("config_files", {})
    navigator_map = state.get("navigator_map") or {}

    prioritized_samples = _select_priority_files(code_samples, navigator_map, max_files=25)
    inspected_count = len(prioritized_samples)
    total_count = len(code_samples)

    source_bundle = "".join(
        f"\n=== FILE: {rel_path} ===\n{snippet[:2000]}\n"
        for rel_path, snippet in prioritized_samples.items()
    )

    manifest_bundle = ""
    if config_files:
        manifest_bundle = "\n\nCONFIGURATION & DEPENDENCY MANIFESTS:\n" + "".join(
            f"\n--- {fname} ---\n{content[:1000]}\n"
            for fname, content in config_files.items()
        )

    inspection_prompt = f"""You are a static code analysis engine. Examine the provided repository files and output valid JSON only.

ARCHITECTURAL CONTEXT:
- Entry Points: {navigator_map.get('entry_points', [])}
- Core Modules: {navigator_map.get('core_modules', [])}
- Architecture Classification: {navigator_map.get('architecture_type', 'unknown')}
- Overview: {navigator_map.get('project_summary', 'N/A')}
- Total Indexed Source Files: {total_count}

PRIORITIZED SOURCE FILES ({inspected_count} shown):
{source_bundle}
{manifest_bundle}

Extract the following technical artifacts directly from the code:
1. `key_functions`: Up to 10 pivotal functions (name, file path, purpose, parameter names).
2. `key_classes`: Up to 10 primary classes (name, file path, responsibility, key method names).
3. `technologies`: Libraries and frameworks verified from import statements and manifest files.
4. `patterns`: Architectural and software design patterns observed in the implementation.
5. `complexity_score`: Numeric rating between 0.0 (trivial script) and 1.0 (highly intricate system).
6. `api_endpoints`: Any HTTP/REST/GraphQL/RPC endpoints defined (empty list if none).
7. `data_models`: Any ORM entities, Pydantic schemas, TypedDicts, or dataclasses (empty list if none).

Respond with valid JSON only (no markdown code blocks):
{{
  "files_analyzed": {inspected_count},
  "key_functions": [
    {{"name": "run_analysis", "file": "src/graph.py", "purpose": "Executes the full repository inspection pipeline", "params": ["repo_url", "github_client"]}}
  ],
  "key_classes": [
    {{"name": "GitHubClient", "file": "src/github_client.py", "purpose": "Handles GitHub API requests and local clone indexing", "methods": ["parse_repo_url", "clone_repo"]}}
  ],
  "technologies": ["LangGraph", "Streamlit", "Pydantic"],
  "patterns": ["Multi-Agent Pipeline", "State Graph", "Service Client"],
  "complexity_score": 0.5,
  "api_endpoints": [
    {{"method": "GET", "path": "/health", "file": "server.py", "purpose": "Service health check"}}
  ],
  "data_models": [
    {{"name": "AgentState", "file": "src/state.py", "fields": ["repo_url", "metadata", "file_tree"]}}
  ]
}}

CONSTRAINTS:
- Report ONLY symbols, imports, routes, and models that genuinely exist in the provided code.
- Use `[]` when no endpoints or data models are present.
"""

    try:
        raw_reply = invoke_llm_text(llm, inspection_prompt)
        parsed_context = extract_json(raw_reply)
        digest = _format_context_summary(inspected_count, total_count, parsed_context)

        tech_count = len(parsed_context.get("technologies", []))
        fn_count = len(parsed_context.get("key_functions", []))
        cls_count = len(parsed_context.get("key_classes", []))

        return {
            "context_output": parsed_context,
            "context_summary": digest,
            "messages": [
                f"CODE INSPECTOR: {inspected_count}/{total_count} files inspected | "
                f"{tech_count} technologies, {fn_count} functions, {cls_count} classes"
            ],
        }
    except Exception as exc:
        return {
            "context_output": {},
            "context_summary": "Error analyzing code",
            "errors": [f"Code Inspector error: {exc}"],
            "messages": [f"CODE INSPECTOR: Failed - {exc}"],
        }
