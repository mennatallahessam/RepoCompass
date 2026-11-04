"""
Stage 5: Report Compiler (Orchestrator Agent)
Consolidates outputs from the Structure Mapper, Code Inspector, Onboarding Architect,
and Diagram Synthesizer into a cohesive technical dossier.
"""
import json
from typing import Any, Dict
from src.state import AgentState
from src.utils import create_chat_model, invoke_llm_text

llm = create_chat_model(temperature=0.2)


def orchestrator_agent(state: AgentState) -> Dict[str, Any]:
    """
    Compile all upstream agent outputs into a unified markdown onboarding report.
    """
    metadata = state.get("metadata", {})
    navigator_map = state.get("navigator_map") or {}
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
[2-3 sentences summarizing the project's purpose and architecture]

## Repository Overview
[Core structural highlights, entry points, and module organization]

## Code Analysis
[Key technologies, design patterns, functions, classes, and complexity evaluation]

## Recommended Learning Path
[Structured summary of the onboarding sequence and prerequisites]

## Architecture Visualization
[Brief note describing the generated component architecture diagram]

## Quality Assessment
- Confidence Score: X/10
- Data Quality: [Summary of completeness or any warnings]
- Recommendations: [Actionable next steps for a new contributor]
"""

    try:
        report_markdown = invoke_llm_text(llm, synthesis_prompt)
        return {
            "final_report": report_markdown,
            "messages": ["REPORT COMPILER: Synthesized final technical dossier"],
        }
    except Exception as exc:
        return {
            "final_report": f"Error creating final report: {exc}",
            "errors": [f"Report Compiler error: {exc}"],
            "messages": [f"REPORT COMPILER: Failed - {exc}"],
        }
