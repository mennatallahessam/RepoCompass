"""
Stage 3: Onboarding Architect (Mentor Agent)
Designs a structured developer onboarding curriculum, prerequisite checklist,
and file-by-file reading sequence tailored to the repository's complexity.
Includes a deterministic curriculum generator fallback.
"""
from typing import Any, Dict, List
from src.state import AgentState
from src.utils import create_chat_model, extract_json, invoke_llm_text

llm = create_chat_model(temperature=0.1)


def _render_onboarding_guide(repo_title: str, plan: Dict[str, Any]) -> str:
    """Format the structured onboarding JSON into a clear text guide."""
    difficulty = str(plan.get("difficulty", "intermediate")).upper()
    hours = plan.get("estimated_total_hours", 2.0)
    prereqs = plan.get("prerequisites", [])
    steps = plan.get("learning_path", [])
    concepts = plan.get("key_concepts", [])

    lines = [
        f"ONBOARDING GUIDE - {repo_title}",
        "",
        f"DIFFICULTY: {difficulty}",
        f"ESTIMATED TIME: {hours} hours",
        "",
        "PREREQUISITES:",
    ]
    for item in prereqs:
        lines.append(f"  - {item}")

    lines.append("\nLEARNING PATH:")
    for item in steps:
        idx = item.get("step", "•")
        target_file = item.get("file", "unknown")
        duration = item.get("estimated_time", "30min")
        topics = ", ".join(item.get("concepts", []))
        lines.append(f"  {idx}. {target_file} ({duration}) - {topics}")

    lines.append(f"\nKEY CONCEPTS: {', '.join(concepts)}")
    return "\n".join(lines)


def _heuristic_onboarding_plan(
    metadata: Dict[str, Any], navigator_map: Dict[str, Any], context_output: Dict[str, Any]
) -> Dict[str, Any]:
    """Deterministically build a developer onboarding curriculum from structural and AST telemetry."""
    lang = metadata.get("language") or "Python"
    arch = navigator_map.get("architecture_type") or "Software System"
    techs = context_output.get("technologies", [])
    patterns = context_output.get("patterns", [])
    complexity = float(context_output.get("complexity_score", 0.5))

    learning_path: List[Dict[str, Any]] = []
    visited_files: set[str] = set()
    step_num = 1

    # Step 1: Project overview & README
    learning_path.append({
        "step": step_num,
        "file": "README.md & Project Manifests",
        "estimated_time": "20min",
        "concepts": ["Project Setup", "Dependencies", arch],
    })
    step_num += 1

    # Step 2: Data models / state definitions
    for dm in context_output.get("data_models", [])[:2]:
        f_path = dm.get("file", "")
        if f_path and f_path not in visited_files:
            visited_files.add(f_path)
            learning_path.append({
                "step": step_num,
                "file": f_path,
                "estimated_time": "25min",
                "concepts": [f"Data Model `{dm.get('name')}`", "State & Schema Contracts"],
            })
            step_num += 1

    # Step 3: Entry points
    for ep in navigator_map.get("entry_points", [])[:2]:
        if ep not in visited_files:
            visited_files.add(ep)
            learning_path.append({
                "step": step_num,
                "file": ep,
                "estimated_time": "30min",
                "concepts": ["Application Bootstrap", "Execution Flow"],
            })
            step_num += 1

    # Step 4: Core classes and functions
    for cls_item in context_output.get("key_classes", [])[:3]:
        f_path = cls_item.get("file", "")
        if f_path and f_path not in visited_files:
            visited_files.add(f_path)
            learning_path.append({
                "step": step_num,
                "file": f_path,
                "estimated_time": "35min",
                "concepts": [f"Class `{cls_item.get('name')}`", cls_item.get("purpose", "Core Service")[:50]],
            })
            step_num += 1

    for fn_item in context_output.get("key_functions", [])[:4]:
        f_path = fn_item.get("file", "")
        if f_path and f_path not in visited_files and len(learning_path) < 7:
            visited_files.add(f_path)
            learning_path.append({
                "step": step_num,
                "file": f_path,
                "estimated_time": "25min",
                "concepts": [f"Function `{fn_item.get('name')}()`", "Core Module Logic"],
            })
            step_num += 1

    prereqs = [f"{lang} proficiency", f"Understanding of {arch}"]
    for t in techs[:4]:
        prereqs.append(f"Familiarity with {t}")

    difficulty = "advanced" if complexity >= 0.7 else ("intermediate" if complexity >= 0.4 else "beginner")
    total_hours = round(max(1.5, len(learning_path) * 0.5), 1)
    key_concepts = (patterns[:3] + techs[:4]) or [arch, "Modular Architecture"]

    return {
        "learning_path": learning_path,
        "prerequisites": prereqs[:6],
        "estimated_total_hours": total_hours,
        "difficulty": difficulty,
        "key_concepts": key_concepts[:6],
    }


def mentor_agent(state: AgentState) -> Dict[str, Any]:
    """
    Generate a step-by-step developer learning roadmap and populate `mentor_guide`.
    """
    navigator_map = state.get("navigator_map") or {}
    context_output = state.get("context_output") or {}
    metadata = state.get("metadata", {})
    repo_title = metadata.get("full_name", "Repository")

    curriculum_prompt = f"""You are a staff engineering mentor designing an onboarding curriculum for new developers joining a repository. Return valid JSON only.

PROJECT: {repo_title}
PRIMARY LANGUAGE: {metadata.get('language', 'Unknown')}

ARCHITECTURAL LAYOUT:
- Entry Points: {navigator_map.get('entry_points', [])}
- Core Modules: {navigator_map.get('core_modules', [])}
- System Type: {navigator_map.get('architecture_type', 'unknown')}

CODEBASE PROFILE:
- Frameworks & Libraries: {context_output.get('technologies', [])}
- Architectural Patterns: {context_output.get('patterns', [])}
- Complexity Rating: {context_output.get('complexity_score', 0.5)}

Return valid JSON only (no markdown code blocks):
{{
  "learning_path": [
    {{"step": 1, "file": "src/state.py", "estimated_time": "20min", "concepts": ["State Schema", "TypedDict"]}}
  ],
  "prerequisites": ["Python 3.10+", "LangGraph fundamentals"],
  "estimated_total_hours": 2.5,
  "difficulty": "intermediate",
  "key_concepts": ["Stateful Agent Graphs", "Repository Indexing"]
}}
"""

    try:
        raw_reply = invoke_llm_text(llm, curriculum_prompt)
        plan_data = extract_json(raw_reply)
        formatted_guide = _render_onboarding_guide(repo_title, plan_data)

        step_count = len(plan_data.get("learning_path", []))
        total_hours = plan_data.get("estimated_total_hours", 0)

        return {
            "mentor_guide": formatted_guide,
            "messages": [
                f"ONBOARDING ARCHITECT: Created {step_count}-step learning path ({total_hours}h total)"
            ],
        }
    except Exception:
        fallback_plan = _heuristic_onboarding_plan(metadata, navigator_map, context_output)
        formatted_guide = _render_onboarding_guide(repo_title, fallback_plan)
        step_count = len(fallback_plan.get("learning_path", []))
        total_hours = fallback_plan.get("estimated_total_hours", 0)

        return {
            "mentor_guide": formatted_guide,
            "messages": [
                f"ONBOARDING ARCHITECT (Static Curriculum Engine): Created {step_count}-step learning path ({total_hours}h total)"
            ],
        }
