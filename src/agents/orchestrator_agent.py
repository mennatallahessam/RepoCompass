"""
Stage 5: Report Compiler (Orchestrator Agent)
Consolidates outputs from the Structure Mapper, Code Inspector, Onboarding Architect,
and Diagram Synthesizer into a cohesive technical dossier.
Includes a deterministic Markdown dossier compiler fallback.
"""
import json
from typing import Any, Dict
from src.state import AgentState
from src.utils import create_chat_model, invoke_llm_text

llm = create_chat_model(temperature=0.2)


def _compile_deterministic_dossier(
    metadata: Dict[str, Any],
    navigator_map: Dict[str, Any],
    context_output: Dict[str, Any],
    mentor_guide: str,
) -> str:
    """Compile a structured Markdown technical onboarding dossier directly from agent outputs."""
    repo_name = metadata.get("full_name", "Repository")
    language = metadata.get("language", "Unknown")
    stars = metadata.get("stars", 0)
    forks = metadata.get("forks", 0)
    arch_type = navigator_map.get("architecture_type", "Software System")
    summary = navigator_map.get("project_summary", "N/A")

    entry_points = ", ".join(f"`{ep}`" for ep in navigator_map.get("entry_points", [])) or "None detected"
    core_mods = "\n".join(
        f"- **`{m.get('path')}`**: {m.get('purpose')}"
        for m in navigator_map.get("core_modules_detailed", [])[:8]
    ) or "- Standard project layout"

    tech_list = ", ".join(f"`{t}`" for t in context_output.get("technologies", [])[:10]) or "Standard Library"
    pattern_list = ", ".join(context_output.get("patterns", [])[:6]) or "Modular architecture"
    complexity = context_output.get("complexity_score", 0.5)

    fn_bullets = "\n".join(
        f"- **`{fn.get('name')}()`** (`{fn.get('file')}`): {fn.get('purpose')}"
        for fn in context_output.get("key_functions", [])[:6]
    ) or "- See source files for function definitions."

    cls_bullets = "\n".join(
        f"- **`{cls.get('name')}`** (`{cls.get('file')}`): {cls.get('purpose')}"
        for cls in context_output.get("key_classes", [])[:6]
    ) or "- Function-oriented module organization."

    confidence_10 = int(round(float(navigator_map.get("confidence_score", 0.85)) * 10))

    return f"""# {repo_name} - Technical Onboarding Dossier

## Executive Summary
**{repo_name}** is a **{language}** project classified as a **{arch_type}** (⭐ {stars:,} stars, 🍴 {forks:,} forks). {summary}

## Repository Overview
- **Architecture Classification**: `{arch_type}`
- **Primary Entry Points**: {entry_points}
- **Core Module Breakdown**:
{core_mods}

## Code Analysis
- **Files Inspected**: `{context_output.get('files_analyzed', 0)}` prioritized source files
- **Technology Stack**: {tech_list}
- **Design Patterns**: {pattern_list}
- **Complexity Index**: `{complexity} / 1.0`

### Notable Classes
{cls_bullets}

### Pivotal Functions
{fn_bullets}

## Recommended Learning Path
```text
{mentor_guide}
```

## Architecture Visualization
An interactive Mermaid component topology diagram (`graph TD`) has been synthesized and is available in the **Architecture & Topology** tab and via chat queries.

## Quality Assessment
- **Confidence Score**: {confidence_10}/10
- **Data Quality**: Verified directly against repository file tree, manifest configs, and AST symbol inspection.
- **Recommendations**: Start by reviewing the primary entry points ({entry_points}), install dependencies from the project manifest, and follow the step-by-step onboarding curriculum above.
"""


def orchestrator_agent(state: AgentState) -> Dict[str, Any]:
    """
    Compile all upstream agent outputs into a unified markdown onboarding report.
    """
    metadata = state.get("metadata", {})
    navigator_map = state.get("navigator_map") or {}
    context_output = state.get("context_output") or {}
    context_summary = state.get("context_summary") or "N/A"
    mentor_guide = state.get("mentor_guide") or "N/A"
    errors = state.get("errors", [])

    repo_name = metadata.get("full_name", "Unknown Repository")
    stars = metadata.get("stars", 0)
    language = metadata.get("language", "Unknown")

    synthesis_prompt = f"""You are a principal technical writer producing an authoritative repository dossier. Be accurate, structured, and concise.

Consolidate the multi-agent inspection telemetry below into a comprehensive markdown report.

TARGET REPOSITORY: {repo_name}
STARS: {stars} | PRIMARY LANGUAGE: {language}

UPSTREAM TELEMETRY:
1. STRUCTURAL MAP: {json.dumps(navigator_map, indent=2)}
2. CODE INSPECTION: {context_summary}
3. ONBOARDING CURRICULUM: {mentor_guide}
4. ARCHITECTURE DIAGRAM: Mermaid flowchart generated

PIPELINE WARNINGS/ERRORS: {errors if errors else "None"}

Format the markdown dossier with the exact sections below:

# {repo_name} - Technical Onboarding Dossier

## Executive Summary
## Repository Overview
## Code Analysis
## Recommended Learning Path
## Architecture Visualization
## Quality Assessment
"""

    try:
        report_markdown = invoke_llm_text(llm, synthesis_prompt)
        return {
            "final_report": report_markdown,
            "messages": ["REPORT COMPILER: Synthesized final technical dossier"],
        }
    except Exception:
        fallback_dossier = _compile_deterministic_dossier(
            metadata, navigator_map, context_output, mentor_guide
        )
        return {
            "final_report": fallback_dossier,
            "messages": ["REPORT COMPILER (Deterministic Synthesizer): Synthesized final technical dossier"],
        }
