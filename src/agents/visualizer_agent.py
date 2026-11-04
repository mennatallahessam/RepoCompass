"""
Stage 4: Diagram Synthesizer (Visualizer Agent)
Constructs a clean Mermaid flowchart illustrating the repository's entry points,
core modules, component interactions, and external dependencies.
"""
from typing import Any, Dict
from src.state import AgentState
from src.utils import create_chat_model, extract_json, invoke_llm_text

llm = create_chat_model(temperature=0.1)


def visualizer_agent(state: AgentState) -> Dict[str, Any]:
    """
    Synthesize a Mermaid architecture diagram from structural and code analysis findings.
    """
    navigator_map = state.get("navigator_map") or {}
    context_output = state.get("context_output") or {}

    fn_names = [fn.get("name", "") for fn in context_output.get("key_functions", [])[:5] if fn.get("name")]
    cls_names = [cls.get("name", "") for cls in context_output.get("key_classes", [])[:5] if cls.get("name")]

    diagram_prompt = f"""You are a software architecture visualization engine. Return valid JSON only.

SYSTEM TOPOLOGY:
- Entry Points: {navigator_map.get('entry_points', [])}
- Core Modules: {navigator_map.get('core_modules', [])}
- Architecture Classification: {navigator_map.get('architecture_type', 'unknown')}

KEY CODE SYMBOLS:
- Notable Functions: {fn_names}
- Notable Classes: {cls_names}

Generate a clean Mermaid `graph TD` flowchart depicting:
1. Application entry points
2. Core internal modules and services
3. Control and data flow arrows between components
4. Key external services or libraries

Return valid JSON only (no markdown fences):
{{
  "mermaid_diagram": "graph TD\\n    Entry[app.py] --> Core[Core Pipeline]\\n    ...",
  "diagram_type": "flowchart TD",
  "component_count": 6,
  "relationships_mapped": 7
}}

MERMAID SYNTAX RULES:
- Keep the graph readable and focused (maximum 15 nodes).
- Use simple alphanumeric node IDs without spaces (e.g., `NodeA["app.py"]`).
- Escape newlines properly inside the JSON string.
"""

    try:
        raw_reply = invoke_llm_text(llm, diagram_prompt)
        parsed = extract_json(raw_reply)

        comp_count = parsed.get("component_count", 0)
        rel_count = parsed.get("relationships_mapped", 0)

        return {
            "visualization": parsed["mermaid_diagram"],
            "messages": [
                f"DIAGRAM SYNTHESIZER: Mapped {comp_count} components and {rel_count} relationships"
            ],
        }
    except Exception as exc:
        return {
            "visualization": "Error creating diagram",
            "errors": [f"Diagram Synthesizer error: {exc}"],
            "messages": [f"DIAGRAM SYNTHESIZER: Failed - {exc}"],
        }
