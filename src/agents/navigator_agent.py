"""
Stage 1: Structure Mapper (Navigator Agent)
Analyzes the repository hierarchy, README documentation, and configuration manifests
to determine architectural style, entry points, core modules, and dependencies.
"""
from typing import Any, Dict, List
from src.state import AgentState
from src.utils import create_chat_model, extract_json, invoke_llm_text

llm = create_chat_model(temperature=0.1)


def _count_leaf_files(subtree: Dict[str, Any]) -> int:
    """Recursively count file leaves within a nested directory dictionary."""
    total = 0
    for child in subtree.values():
        if child is None:
            total += 1
        elif isinstance(child, dict):
            total += _count_leaf_files(child)
    return total


def _build_tree_view(file_tree: List[Dict[str, Any]]) -> str:
    """Format a flat list of file descriptors into an ASCII hierarchy tree."""
    hierarchy: Dict[str, Any] = {}

    for entry in file_tree:
        if entry.get("type") != "blob":
            continue
        segments = entry["path"].split("/")
        cursor = hierarchy
        for folder in segments[:-1]:
            if not isinstance(cursor.get(folder), dict):
                cursor[folder] = {}
            cursor = cursor[folder]
        leaf_name = segments[-1]
        if leaf_name not in cursor or cursor[leaf_name] is None:
            cursor[leaf_name] = None

    rendered_lines: List[str] = []

    def _walk_node(mapping: Dict[str, Any], indent: str = "") -> None:
        ordered = sorted(mapping.items(), key=lambda pair: (pair[1] is None, pair[0]))
        total_items = len(ordered)
        for idx, (label, branch) in enumerate(ordered):
            last_item = idx == (total_items - 1)
            branch_glyph = "└── " if last_item else "├── "
            if branch is None:
                rendered_lines.append(f"{indent}{branch_glyph}{label}")
            else:
                file_tally = _count_leaf_files(branch)
                rendered_lines.append(f"{indent}{branch_glyph}{label}/ ({file_tally} files)")
                next_indent = indent + ("    " if last_item else "│   ")
                _walk_node(branch, next_indent)

    _walk_node(hierarchy)
    return "\n".join(rendered_lines)


def _format_manifest_block(config_files: Dict[str, str]) -> str:
    """Format configuration and manifest files for inclusion in the analysis prompt."""
    if not config_files:
        return ""
    blocks = ["\nPROJECT MANIFESTS & CONFIGURATIONS:"]
    for path, raw_text in config_files.items():
        blocks.append(f"\n--- {path} ---\n{raw_text[:1500]}")
    return "\n".join(blocks)


def navigator_agent(state: AgentState) -> Dict[str, Any]:
    """
    Execute structural mapping on the repository state and populate `navigator_map`.
    """
    file_tree = state.get("file_tree", [])
    metadata = state.get("metadata", {})
    readme_text = state.get("readme_content")
    manifests = state.get("config_files", {})

    hierarchy_str = _build_tree_view(file_tree)
    hierarchy_rows = hierarchy_str.splitlines()
    if len(hierarchy_rows) > 200:
        overflow = len(hierarchy_rows) - 200
        hierarchy_str = "\n".join(hierarchy_rows[:200]) + f"\n... (+{overflow} additional entries)"

    readme_block = f"\nREADME EXCERPT:\n{readme_text[:3000]}\n" if readme_text else ""
    manifest_block = _format_manifest_block(manifests)

    analysis_prompt = f"""You are a senior software architect inspecting a GitHub repository. Return ONLY valid JSON.

TARGET REPOSITORY: {metadata.get('full_name', 'unknown')}
PRIMARY LANGUAGE: {metadata.get('language', 'Unknown')}
DESCRIPTION: {metadata.get('description', 'N/A')}
FILE COUNT: {len(file_tree)}

DIRECTORY HIERARCHY:
{hierarchy_str}
{readme_block}
{manifest_block}

Determine the following from the provided evidence:
1. `entry_points`: Primary bootstrap/startup files (e.g., app.py, main.py, index.ts, server.js, manage.py).
2. `core_modules`: Key directories and foundational modules across the repository.
3. `core_modules_detailed`: Array of objects describing each core path and its role.
4. `dependencies`: Concrete packages declared in the manifest files above (flat strings only).
5. `architecture_type`: Select the most accurate classification from the allowed categories below.

Output strictly valid JSON matching this schema (no markdown fences):
{{
  "entry_points": ["path/to/entry.py"],
  "core_modules": ["src/", "app.py"],
  "core_modules_detailed": [
    {{"path": "src/services/", "purpose": "Core domain logic and service handlers"}},
    {{"path": "app.py", "purpose": "Application bootstrap and UI entry point"}}
  ],
  "dependencies": ["streamlit>=1.31.0", "pydantic>=2.7.4"],
  "architecture_type": "selected category",
  "project_summary": "Concise 2-3 sentence technical overview of the repository's purpose and workflow.",
  "confidence_score": 0.9
}}

ALLOWED ARCHITECTURE CATEGORIES:
- "Web Application (Full-Stack)"
- "Web API / REST Service"
- "Desktop GUI Application"
- "CLI Tool"
- "Data Science / ML Pipeline"
- "Mobile Application"
- "Library / Package"
- "Microservices"
- "Monolith"
- "Static Website"
- "DevOps / Infrastructure"

STRICT GUIDELINES:
- Base every field strictly on the directory tree, README, and manifests above.
- Desktop UI frameworks (Tkinter, PyQt, Kivy, Electron, wxPython) must be classified as "Desktop GUI Application".
- Format every dependency item as a single string (e.g. "requests>=2.31.0"), never as nested key-value pairs.
- Provide a realistic `confidence_score` between 0.0 and 1.0.
"""

    try:
        raw_output = invoke_llm_text(llm, analysis_prompt)
        parsed_map = extract_json(raw_output)

        if readme_text:
            snippet = readme_text[:500].strip()
            parsed_map["readme_summary"] = f"{snippet}..." if len(readme_text) > 500 else snippet
        else:
            parsed_map["readme_summary"] = "No README documentation found"

        ep_count = len(parsed_map.get("entry_points", []))
        mod_count = len(parsed_map.get("core_modules", []))
        arch_label = parsed_map.get("architecture_type", "unknown")

        return {
            "navigator_map": parsed_map,
            "messages": [
                f"STRUCTURE MAPPER: Identified {ep_count} entry points, "
                f"{mod_count} core modules ({arch_label})"
            ],
        }
    except Exception as exc:
        return {
            "navigator_map": {
                "entry_points": [],
                "core_modules": [],
                "core_modules_detailed": [],
                "dependencies": [],
                "architecture_type": "unknown",
                "confidence_score": 0.0,
                "project_summary": "Structural mapping could not be completed.",
                "readme_summary": "No README documentation found",
            },
            "errors": [f"Structure Mapper error: {exc}"],
            "messages": [f"STRUCTURE MAPPER: Failed - {exc}"],
        }
