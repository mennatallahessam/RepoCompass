"""
Stage 2: Code Inspector (Context Agent)
Performs deep static inspection of prioritized source files to identify key functions,
classes, frameworks, design patterns, API routes, data schemas, and complexity metrics.
Includes a built-in Python AST + regex static code analyzer fallback.
"""
import ast
import re
from pathlib import PurePosixPath
from typing import Any, Dict, List
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


def _ast_static_code_analysis(
    selected_files: Dict[str, str], navigator_map: Dict[str, Any]
) -> Dict[str, Any]:
    """Perform deterministic AST and regex static code inspection across prioritized files."""
    key_functions: List[Dict[str, Any]] = []
    key_classes: List[Dict[str, Any]] = []
    api_endpoints: List[Dict[str, str]] = []
    data_models: List[Dict[str, Any]] = []
    imported_modules: set[str] = set()
    detected_patterns: set[str] = set()

    stdlib_ignore = {
        "os", "sys", "re", "json", "time", "math", "typing", "pathlib", "shutil",
        "tempfile", "subprocess", "operator", "collections", "itertools", "functools",
        "datetime", "logging", "traceback", "unittest", "abc", "copy", "io", "uuid",
    }

    for rel_path, content in selected_files.items():
        ext = PurePosixPath(rel_path).suffix.lower()

        # Detect HTTP routes via decorator/router patterns (FastAPI, Flask, Express)
        for route_match in re.finditer(
            r"@(?:\w+)\.(get|post|put|delete|patch|route)\(\s*['\"]([^'\"]+)['\"]",
            content,
            re.IGNORECASE,
        ):
            verb = route_match.group(1).upper()
            if verb == "ROUTE":
                verb = "HTTP"
            route_path = route_match.group(2)
            api_endpoints.append({
                "method": verb,
                "path": route_path,
                "file": rel_path,
                "purpose": f"Handles {verb} requests for {route_path}",
            })
            detected_patterns.add("REST API Controller / Decorator Routing")

        if ext == ".py":
            try:
                tree = ast.parse(content)
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top_pkg = alias.name.split(".")[0]
                        if top_pkg not in stdlib_ignore and not top_pkg.startswith("_"):
                            imported_modules.add(top_pkg)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    top_pkg = node.module.split(".")[0]
                    if top_pkg not in stdlib_ignore and not top_pkg.startswith("_") and top_pkg != "src":
                        imported_modules.add(top_pkg)

            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if isinstance(node, ast.AsyncFunctionDef):
                        detected_patterns.add("Async / Coroutine Execution")
                    if node.name.startswith("_") and len(key_functions) >= 6:
                        continue
                    doc = (ast.get_docstring(node) or "").splitlines()
                    purpose = doc[0].strip() if doc else f"Executes `{node.name}` workflow in `{rel_path}`"
                    params = [arg.arg for arg in node.args.args if arg.arg != "self"]
                    key_functions.append({
                        "name": node.name,
                        "file": rel_path,
                        "purpose": purpose[:140],
                        "params": params[:6],
                    })
                elif isinstance(node, ast.ClassDef):
                    bases = [
                        ast.unparse(b) if hasattr(ast, "unparse") else getattr(b, "id", "")
                        for b in node.bases
                    ]
                    doc = (ast.get_docstring(node) or "").splitlines()
                    purpose = doc[0].strip() if doc else f"Encapsulates `{node.name}` domain state and methods"
                    methods = [
                        item.name
                        for item in node.body
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and not item.name.startswith("__")
                    ]

                    is_schema = any(
                         any(k in b for k in ("BaseModel", "TypedDict", "Model", "Schema", "Enum"))
                        for b in bases
                    ) or any(
                        getattr(dec, "id", "") == "dataclass" for dec in node.decorator_list
                    )

                    if is_schema:
                        fields = [
                            stmt.target.id
                            for stmt in node.body
                            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
                        ]
                        data_models.append({
                            "name": node.name,
                            "file": rel_path,
                            "fields": fields[:12] or methods[:6],
                        })
                        detected_patterns.add("Typed Data Schema / Validation Model")
                    else:
                        key_classes.append({
                            "name": node.name,
                            "file": rel_path,
                            "purpose": purpose[:140],
                            "methods": methods[:8],
                        })
                        detected_patterns.add("Service / Client Encapsulation")

        elif ext in {".js", ".ts", ".jsx", ".tsx"}:
            for fn_match in re.finditer(r"(?:export\s+)?(?:async\s+)?function\s+([A-Za-z0-9_]+)\s*\(([^)]*)\)", content):
                fn_name = fn_match.group(1)
                raw_params = [p.strip().split(":")[0].strip() for p in fn_match.group(2).split(",") if p.strip()]
                key_functions.append({
                    "name": fn_name,
                    "file": rel_path,
                    "purpose": f"Module function `{fn_name}` in `{rel_path}`",
                    "params": raw_params[:6],
                })
            for cls_match in re.finditer(r"class\s+([A-Za-z0-9_]+)", content):
                key_classes.append({
                    "name": cls_match.group(1),
                    "file": rel_path,
                    "purpose": f"Component or service class in `{rel_path}`",
                    "methods": [],
                })

    # Combine manifest dependencies with imported third-party modules
    manifest_deps = [
        re.split(r"[><=~!\s]", d)[0] for d in navigator_map.get("dependencies", []) if d
    ]
    technologies: List[str] = []
    seen_tech = set()
    for t in manifest_deps + sorted(imported_modules):
        clean_t = t.strip()
        if clean_t and clean_t.lower() not in seen_tech:
            seen_tech.add(clean_t.lower())
            technologies.append(clean_t)

    if any("langgraph" in t.lower() for t in technologies):
        detected_patterns.add("Stateful Multi-Agent Pipeline (LangGraph)")
    if any("streamlit" in t.lower() for t in technologies):
        detected_patterns.add("Reactive Session-State Web UI")
    if not detected_patterns:
        detected_patterns.add("Modular Layered Architecture")

    # Calculate structural complexity index (0.2 to 0.9)
    raw_score = min(
        0.9,
        max(
            0.25,
            round(0.2 + (len(key_classes) * 0.04) + (len(key_functions) * 0.015) + (len(selected_files) * 0.01), 2),
        ),
    )

    return {
        "files_analyzed": len(selected_files),
        "key_functions": key_functions[:12],
        "key_classes": key_classes[:10],
        "technologies": technologies[:15],
        "patterns": sorted(detected_patterns)[:6],
        "complexity_score": raw_score,
        "api_endpoints": api_endpoints[:10],
        "data_models": data_models[:10],
    }


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
  "api_endpoints": [],
  "data_models": []
}}
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
    except Exception:
        static_result = _ast_static_code_analysis(prioritized_samples, navigator_map)
        digest = _format_context_summary(inspected_count, total_count, static_result)
        tech_count = len(static_result.get("technologies", []))
        fn_count = len(static_result.get("key_functions", []))
        cls_count = len(static_result.get("key_classes", []))

        return {
            "context_output": static_result,
            "context_summary": digest,
            "messages": [
                f"CODE INSPECTOR (Static AST Engine): {inspected_count}/{total_count} files inspected | "
                f"{tech_count} technologies, {fn_count} functions, {cls_count} classes"
            ],
        }
