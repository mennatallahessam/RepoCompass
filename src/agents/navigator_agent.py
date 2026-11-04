"""
Stage 1: Structure Mapper (Navigator Agent)
Analyzes the repository hierarchy, README documentation, and configuration manifests
to determine architectural style, entry points, core modules, and dependencies.
Includes a deterministic static structure analyzer fallback when no LLM API key is active.
"""
import json
import re
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional
from src.state import AgentState
from src.utils import create_chat_model, extract_json, invoke_llm_text

llm = create_chat_model(temperature=0.1)

COMMON_ENTRY_FILENAMES = {
    "app.py", "main.py", "server.py", "manage.py", "cli.py", "run.py", "wsgi.py", "asgi.py",
    "__main__.py", "index.js", "index.ts", "server.js", "server.ts", "app.js", "app.ts",
    "main.go", "main.rs", "Main.java", "Program.cs",
}

DIRECTORY_PURPOSE_HINTS = {
    "src": "Primary application source code and domain logic",
    "app": "Application package, views, and routing handlers",
    "lib": "Shared library modules and reusable utilities",
    "core": "Core domain abstractions, state, and business rules",
    "api": "HTTP/REST API endpoints, controllers, and schemas",
    "routes": "URL route definitions and request handlers",
    "models": "Data models, ORM entities, and schema definitions",
    "schemas": "Request/response validation schemas",
    "services": "Business logic and external service integrations",
    "agents": "Autonomous AI agent implementations and prompts",
    "components": "UI components and reusable view building blocks",
    "pages": "Application page views and route screens",
    "utils": "Helper functions and cross-cutting utilities",
    "config": "Configuration settings and environment profiles",
    "tests": "Automated unit and integration test suites",
    "docs": "Project documentation and architecture guides",
    "scripts": "Automation, build, and maintenance scripts",
    "static": "Static frontend assets (CSS, images, scripts)",
    "templates": "HTML/Jinja view templates",
}


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


def _parse_manifest_dependencies(config_files: Dict[str, str]) -> List[str]:
    """Deterministically extract dependency strings from requirements.txt, package.json, or pyproject.toml."""
    deps: List[str] = []
    seen = set()

    for path, content in config_files.items():
        fname = PurePosixPath(path).name.lower()
        if fname == "requirements.txt":
            for raw_line in content.splitlines():
                line = raw_line.split("#")[0].strip()
                if line and not line.startswith("-"):
                    pkg_key = re.split(r"[><=~!]", line)[0].strip().lower()
                    if pkg_key not in seen:
                        seen.add(pkg_key)
                        deps.append(line)
        elif fname == "package.json":
            try:
                pkg_data = json.loads(content)
                for sec in ("dependencies", "devDependencies"):
                    for k, v in (pkg_data.get(sec) or {}).items():
                        if k.lower() not in seen:
                            seen.add(k.lower())
                            deps.append(f"{k} {v}")
            except Exception:
                pass
        elif fname == "pyproject.toml":
            for match in re.finditer(r'^\s*"([a-zA-Z0-9_\-\.\[\]>=<~]+)"\s*,?\s*$', content, re.MULTILINE):
                val = match.group(1).strip()
                pkg_key = re.split(r"[><=~!]", val)[0].strip().lower()
                if pkg_key and pkg_key not in seen:
                    seen.add(pkg_key)
                    deps.append(val)

    return deps[:40]


def _heuristic_structure_analysis(
    file_tree: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    readme_text: Optional[str],
    manifests: Dict[str, str],
) -> Dict[str, Any]:
    """Deterministically map entry points, core modules, dependencies, and architecture type."""
    all_paths = [f["path"] for f in file_tree if f.get("type") == "blob"]

    # 1. Identify entry points
    entry_points: List[str] = []
    for p in all_paths:
        parts = p.split("/")
        if len(parts) <= 2 and parts[-1] in COMMON_ENTRY_FILENAMES:
            entry_points.append(p)
    if not entry_points:
        for p in all_paths:
            if PurePosixPath(p).name in COMMON_ENTRY_FILENAMES:
                entry_points.append(p)
    if not entry_points:
        code_files = [p for p in all_paths if PurePosixPath(p).suffix in {".py", ".js", ".ts", ".go", ".rs", ".java"}]
        entry_points = code_files[:3]

    # 2. Identify core modules and detailed purposes
    dir_counts: Dict[str, int] = {}
    for p in all_paths:
        parts = p.split("/")
        if len(parts) > 1:
            top_dir = f"{parts[0]}/"
            dir_counts[top_dir] = dir_counts.get(top_dir, 0) + 1
            if len(parts) > 2 and parts[0] in {"src", "app", "lib"}:
                sub_dir = f"{parts[0]}/{parts[1]}/"
                dir_counts[sub_dir] = dir_counts.get(sub_dir, 0) + 1

    core_modules: List[str] = []
    core_modules_detailed: List[Dict[str, str]] = []

    for folder, count in sorted(dir_counts.items(), key=lambda x: (-x[1], x[0]))[:10]:
        core_modules.append(folder)
        leaf_folder = folder.strip("/").split("/")[-1].lower()
        hint = DIRECTORY_PURPOSE_HINTS.get(
            leaf_folder, f"Package module containing {count} files"
        )
        core_modules_detailed.append({"path": folder, "purpose": f"{hint} ({count} files)"})

    for ep in entry_points[:4]:
        if ep not in core_modules:
            core_modules.append(ep)
            core_modules_detailed.append({
                "path": ep,
                "purpose": "Application bootstrap and execution entry point",
            })

    # 3. Parse dependencies
    dependencies = _parse_manifest_dependencies(manifests)
    deps_lower = " ".join(dependencies).lower()

    # 4. Classify architecture
    if any(k in deps_lower for k in ("streamlit", "gradio", "next", "nuxt", "react", "vue", "angular")):
        arch_type = "Web Application (Full-Stack)"
    elif any(k in deps_lower for k in ("fastapi", "flask", "django", "express", "uvicorn", "aiohttp", "gin")):
        arch_type = "Web API / REST Service"
    elif any(k in deps_lower for k in ("tkinter", "pyqt", "pyside", "kivy", "wxpython", "electron")):
        arch_type = "Desktop GUI Application"
    elif any(k in deps_lower for k in ("torch", "tensorflow", "scikit-learn", "sklearn", "keras", "xgboost")) or any(
        p.endswith(".ipynb") for p in all_paths
    ):
        arch_type = "Data Science / ML Pipeline"
    elif any(PurePosixPath(p).name in {"setup.py", "pyproject.toml", "Cargo.toml"} for p in all_paths) and not entry_points:
        arch_type = "Library / Package"
    elif entry_points:
        arch_type = "CLI Tool"
    else:
        arch_type = "Monolith"

    # 5. Extract project summary from README or metadata description
    summary = metadata.get("description") or ""
    if readme_text:
        paragraphs = [
            p.strip()
            for p in re.split(r"\n\s*\n", readme_text)
            if p.strip() and not p.strip().startswith(("#", "!", "[!", "<", "```", "---"))
        ]
        if paragraphs:
            clean_para = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", paragraphs[0])
            clean_para = re.sub(r"[*_`]", "", clean_para).strip()
            if len(clean_para) > 30:
                summary = clean_para[:350]

    if not summary or summary == "No description":
        summary = (
            f"{metadata.get('full_name', 'This repository')} is a {metadata.get('language', 'software')} "
            f"{arch_type} structured across {len(all_paths)} files and {len(core_modules)} core modules."
        )

    readme_summary = "No README documentation found"
    if readme_text:
        snippet = readme_text[:500].strip()
        readme_summary = f"{snippet}..." if len(readme_text) > 500 else snippet

    return {
        "entry_points": entry_points[:8],
        "core_modules": core_modules[:12],
        "core_modules_detailed": core_modules_detailed[:12],
        "dependencies": dependencies,
        "architecture_type": arch_type,
        "project_summary": summary,
        "readme_summary": readme_summary,
        "confidence_score": 0.88,
    }


def navigator_agent(state: AgentState) -> Dict[str, Any]:
    """
    Execute structural mapping on the repository state and populate `navigator_map`.
    Uses LLM analysis when configured, or deterministic static structure mapping as fallback.
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
    except Exception:
        fallback_map = _heuristic_structure_analysis(file_tree, metadata, readme_text, manifests)
        ep_count = len(fallback_map.get("entry_points", []))
        mod_count = len(fallback_map.get("core_modules", []))
        arch_label = fallback_map.get("architecture_type", "unknown")
        return {
            "navigator_map": fallback_map,
            "messages": [
                f"STRUCTURE MAPPER (Static AST/Manifest Engine): Identified {ep_count} entry points, "
                f"{mod_count} core modules ({arch_label})"
            ],
        }
