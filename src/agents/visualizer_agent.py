"""
Stage 4: Diagram Synthesizer (Visualizer Agent)
Constructs a clean Mermaid flowchart illustrating the repository's entry points,
core modules, component interactions, and external dependencies.
Includes a programmatic Mermaid generator fallback that guarantees valid diagrams.
"""
import re
from typing import Any, Dict, List, Tuple
from src.state import AgentState
from src.utils import create_chat_model, extract_json, invoke_llm_text

llm = create_chat_model(temperature=0.1)


def _clean_label(text: str) -> str:
    """Sanitize a node label for safe Mermaid rendering."""
    cleaned = re.sub(r'["\[\]{}()<>|]', "", str(text)).strip()
    return cleaned[:40] if cleaned else "Component"


def build_programmatic_mermaid(
    navigator_map: Dict[str, Any], context_output: Dict[str, Any]
) -> Tuple[str, int, int]:
    """
    Programmatically construct a valid Mermaid `graph TD` architecture diagram
    from verified repository entry points, core modules, classes/functions, and dependencies.
    """
    entry_points = navigator_map.get("entry_points", [])[:3] or ["main"]
    core_modules = [
        m for m in navigator_map.get("core_modules", []) if m not in entry_points
    ][:5]
    classes = context_output.get("key_classes", [])[:4]
    functions = context_output.get("key_functions", [])[:4]
    technologies = context_output.get("technologies", [])[:4]

    lines: List[str] = ["graph TD"]
    node_count = 0
    edge_count = 0

    # Entry point nodes
    ep_ids: List[str] = []
    for idx, ep in enumerate(entry_points):
        nid = f"EP{idx}"
        ep_ids.append(nid)
        lines.append(f'    {nid}["Entry: {_clean_label(ep)}"]')
        node_count += 1

    # Core module nodes
    mod_ids: List[Tuple[str, str]] = []
    for idx, mod in enumerate(core_modules):
        nid = f"MOD{idx}"
        mod_ids.append((nid, mod.rstrip("/")))
        lines.append(f'    {nid}["Module: {_clean_label(mod)}"]')
        node_count += 1
        for ep_id in ep_ids[:2]:
            lines.append(f"    {ep_id} --> {nid}")
            edge_count += 1

    # Key classes / functions nodes
    symbol_nodes: List[Tuple[str, str]] = []
    for idx, cls_item in enumerate(classes):
        nid = f"CLS{idx}"
        cls_name = cls_item.get("name", f"Class{idx}")
        cls_file = cls_item.get("file", "")
        symbol_nodes.append((nid, cls_file))
        lines.append(f'    {nid}["Class: {_clean_label(cls_name)}"]')
        node_count += 1

    if len(symbol_nodes) < 4:
        for idx, fn_item in enumerate(functions[: 4 - len(symbol_nodes)]):
            nid = f"FN{idx}"
            fn_name = fn_item.get("name", f"func_{idx}")
            fn_file = fn_item.get("file", "")
            symbol_nodes.append((nid, fn_file))
            lines.append(f'    {nid}["Func: {_clean_label(fn_name)}()"]')
            node_count += 1

    # Connect modules (or entry points) to symbols
    for sym_id, sym_file in symbol_nodes:
        matched_mod = None
        for mod_id, mod_prefix in mod_ids:
            if sym_file.startswith(mod_prefix):
                matched_mod = mod_id
                break
        source_id = matched_mod or (mod_ids[0][0] if mod_ids else ep_ids[0])
        lines.append(f"    {source_id} --> {sym_id}")
        edge_count += 1

    # External dependencies / tech stack
    if technologies:
        tech_label = _clean_label(", ".join(technologies[:3]))
        lines.append(f'    EXT["Stack: {tech_label}"]')
        node_count += 1
        target_sources = [s[0] for s in symbol_nodes[:2]] or [m[0] for m in mod_ids[:2]] or ep_ids[:1]
        for src in target_sources:
            lines.append(f"    {src} -.-> EXT")
            edge_count += 1

    return "\n".join(lines), node_count, edge_count


def visualizer_agent(state: AgentState) -> Dict[str, Any]:
    """
    Synthesize a Mermaid architecture diagram from structural and code analysis findings.
    Automatically falls back to programmatic Mermaid generation if the LLM is unavailable.
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

Return valid JSON only (no markdown fences):
{{
  "mermaid_diagram": "graph TD\\n    Entry[app.py] --> Core[Core Pipeline]\\n    ...",
  "diagram_type": "flowchart TD",
  "component_count": 6,
  "relationships_mapped": 7
}}
"""

    try:
        raw_reply = invoke_llm_text(llm, diagram_prompt)
        parsed = extract_json(raw_reply)
        diagram_str = parsed.get("mermaid_diagram", "").strip()
        if not diagram_str or "Error" in diagram_str:
            raise ValueError("Empty diagram from model")

        comp_count = parsed.get("component_count", 0)
        rel_count = parsed.get("relationships_mapped", 0)

        return {
            "visualization": diagram_str,
            "messages": [
                f"DIAGRAM SYNTHESIZER: Mapped {comp_count} components and {rel_count} relationships"
            ],
        }
    except Exception:
        diagram_str, comp_count, rel_count = build_programmatic_mermaid(navigator_map, context_output)
        return {
            "visualization": diagram_str,
            "messages": [
                f"DIAGRAM SYNTHESIZER (Programmatic Graph Engine): Mapped {comp_count} components and {rel_count} relationships"
            ],
        }
