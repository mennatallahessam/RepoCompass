"""
Stage 3: Onboarding Architect (Mentor Agent)
Designs a structured developer onboarding curriculum, prerequisite checklist,
and file-by-file reading sequence tailored to the repository's complexity.
"""
from typing import Any, Dict
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

Construct a practical onboarding roadmap containing:
1. `learning_path`: Ordered sequence of real files/modules to study with realistic time estimates and core concepts.
2. `prerequisites`: Essential foundational skills required before diving into the code.
3. `estimated_total_hours`: Total onboarding time in hours (numeric float).
4. `difficulty`: Overall difficulty classification ("beginner", "intermediate", or "advanced").
5. `key_concepts`: Domain and architectural concepts central to mastering this project.

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
    except Exception as exc:
        return {
            "mentor_guide": "Error creating onboarding guide",
            "errors": [f"Onboarding Architect error: {exc}"],
            "messages": [f"ONBOARDING ARCHITECT: Failed - {exc}"],
        }
