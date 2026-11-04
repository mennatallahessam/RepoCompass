"""
RepoCompass - Autonomous Codebase Intelligence & Onboarding Studio.
Launch with: streamlit run app.py
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from streamlit_mermaid import st_mermaid

from src.github_client import GitHubClient
from src.graph import run_analysis
from src.utils import (
    create_chat_model,
    has_active_api_key,
    invoke_llm_text,
    set_runtime_api_config,
)

# --- Page Configuration ---
st.set_page_config(
    page_title="RepoCompass | Codebase Intelligence Studio",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "RepoCompass — Multi-Agent GitHub Repository Architecture & Onboarding Studio"
    },
)

# --- Custom Emerald & Cyan Glassmorphic Styling ---
st.markdown(
    """
<style>
    :root {
        --emerald-primary: #10b981;
        --cyan-accent: #06b6d4;
        --amber-warn: #f59e0b;
        --obsidian-bg: #090d16;
        --panel-surface: #111827;
        --panel-border: rgba(16, 185, 129, 0.25);
        --text-main: #f8fafc;
        --text-dim: #94a3b8;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Top studio banner */
    .atlas-banner {
        background: linear-gradient(125deg, #064e3b 0%, #0f172a 55%, #083344 100%);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 14px;
        padding: 1.75rem 2.25rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 32px rgba(6, 182, 212, 0.12);
    }

    .atlas-banner h1 {
        color: #ecfdf5;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
    }

    .atlas-banner p {
        color: #a7f3d0;
        font-size: 1.02rem;
        margin: 0.45rem 0 0 0;
    }

    /* KPI stat tiles */
    .kpi-tile {
        background: linear-gradient(145deg, #111827 0%, #1e293b 100%);
        border: 1px solid rgba(6, 182, 212, 0.28);
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    }

    .kpi-label {
        color: #94a3b8;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }

    .kpi-value {
        color: #ecfdf5;
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 0.25rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Feature cards on landing view */
    .stage-card {
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid rgba(16, 185, 129, 0.22);
        border-left: 4px solid #10b981;
        border-radius: 10px;
        padding: 1.15rem 1.35rem;
        margin-bottom: 0.85rem;
    }

    .stage-card h4 {
        color: #34d399;
        margin: 0 0 0.35rem 0;
        font-size: 1.05rem;
    }

    .stage-card p {
        color: #cbd5e1;
        margin: 0;
        font-size: 0.92rem;
    }

    /* Tag pills */
    .tech-pill {
        display: inline-block;
        background: rgba(16, 185, 129, 0.14);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 999px;
        padding: 0.2rem 0.75rem;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 0.2rem 0.25rem 0.2rem 0;
    }

    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1120 0%, #111827 100%);
        border-right: 1px solid rgba(16, 185, 129, 0.2);
    }

    .stButton > button {
        background: linear-gradient(135deg, #059669 0%, #0891b2 100%);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 0.65rem 1.4rem;
        font-weight: 600;
        transition: all 0.25s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(16, 185, 129, 0.35);
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- Conversational Model Initialization ---
chat_llm = create_chat_model(temperature=0.3)


def build_context(analysis: Dict[str, Any]) -> str:
    """Compile a comprehensive grounding prompt from all repository analysis artifacts."""
    nav = analysis.get("navigator_map") or {}
    ctx = analysis.get("context_output") or {}
    meta = analysis.get("metadata") or {}

    code_blocks: List[str] = []
    for rel_path, source_body in list(analysis.get("code_samples", {}).items())[:50]:
        code_blocks.append(f"\n### {rel_path}\n```\n{source_body}\n```\n")

    manifest_blocks: List[str] = []
    for manifest_name, manifest_body in analysis.get("config_files", {}).items():
        manifest_blocks.append(f"\n### {manifest_name}\n```\n{manifest_body}\n```\n")

    file_tree = analysis.get("file_tree", [])
    listed_paths = "\n".join(item["path"] for item in file_tree[:150])
    if len(file_tree) > 150:
        listed_paths += f"\n... and {len(file_tree) - 150} additional files"

    commit_lines = "\n".join(
        f"- {c.get('sha', '')[:7]} {c.get('message', '')} (by {c.get('author', 'unknown')})"
        for c in analysis.get("recent_commits", [])[:10]
    )

    pr_lines = "\n".join(
        f"- #{pr.get('number', '')} {pr.get('title', '')} [{pr.get('state', '')}] by {pr.get('author', 'unknown')}"
        for pr in analysis.get("pull_requests", [])[:10]
    )

    module_lines = "\n".join(
        f"- {m.get('path', '')}: {m.get('purpose', '')}"
        for m in nav.get("core_modules_detailed", [])
    )

    endpoint_lines = "\n".join(
        f"- {ep.get('method', '')} {ep.get('path', '')} ({ep.get('file', '')}) - {ep.get('purpose', '')}"
        for ep in ctx.get("api_endpoints", [])
    )

    model_lines = "\n".join(
        f"- {dm.get('name', '')} ({dm.get('file', '')}): {', '.join(dm.get('fields', []))}"
        for dm in ctx.get("data_models", [])
    )

    return f"""You are RepoCompass, an expert AI repository architect and onboarding assistant.
You have inspected the repository "{meta.get('full_name', 'unknown')}" and compiled the following verified telemetry.
Base every answer strictly on this repository data, citing exact file paths, classes, and functions.

## Repository Metadata
- Full Name: {meta.get('full_name')}
- Primary Language: {meta.get('language')}
- Stars: {meta.get('stars', 0)}
- Description: {meta.get('description', 'N/A')}

## Structural Topology (Structure Mapper)
- Entry Points: {nav.get('entry_points', [])}
- Core Modules: {nav.get('core_modules', [])}
- Dependencies: {nav.get('dependencies', [])}
- Architecture Classification: {nav.get('architecture_type', 'unknown')}
- Confidence Rating: {nav.get('confidence_score', 0)}
- Project Summary: {nav.get('project_summary', 'N/A')}
- README Excerpt: {nav.get('readme_summary', 'N/A')}

### Detailed Module Breakdown
{module_lines or 'N/A'}

## Deep Code Inspection (Code Inspector)
- Inspected Files: {ctx.get('files_analyzed', 0)}
- Detected Technologies: {ctx.get('technologies', [])}
- Design Patterns: {ctx.get('patterns', [])}
- Complexity Index: {ctx.get('complexity_score', 'N/A')}
- Key Functions: {json.dumps(ctx.get('key_functions', []), indent=2)}
- Key Classes: {json.dumps(ctx.get('key_classes', []), indent=2)}

### API Routes
{endpoint_lines or 'None detected'}

### Data Schemas & Models
{model_lines or 'None detected'}

## Onboarding Curriculum (Onboarding Architect)
{analysis.get('mentor_guide', 'N/A')}

## Component Architecture Diagram (Diagram Synthesizer)
```mermaid
{analysis.get('visualization', 'No diagram available')}
```

## Repository File Hierarchy ({len(file_tree)} files)
{listed_paths}

## Recent Commit History
{commit_lines or 'None available'}

## Pull Requests
{pr_lines or 'None available'}

## Manifest & Config Files
{''.join(manifest_blocks) if manifest_blocks else 'None found'}

## Indexed Source Code Samples
{''.join(code_blocks)}
"""


def render_message_with_mermaid(message: str) -> None:
    """Render markdown content while automatically converting fenced ```mermaid blocks into diagrams."""
    fence_regex = r"```mermaid\n(.*?)\n```"
    matches = list(re.finditer(fence_regex, message, re.DOTALL))

    if not matches:
        st.markdown(message)
        return

    cursor = 0
    for match in matches:
        if match.start() > cursor:
            st.markdown(message[cursor : match.start()])
        diagram_source = match.group(1)
        st_mermaid(diagram_source, height=420)
        cursor = match.end()

    if cursor < len(message):
        st.markdown(message[cursor:])


def _synthesize_grounded_fallback_answer(analysis: Dict[str, Any], user_msg: str) -> str:
    """
    Generate a rich, repository-grounded response directly from the analyzed AST,
    topology, and Mermaid artifacts when no external LLM API key is active.
    """
    q = user_msg.lower()
    meta = analysis.get("metadata") or {}
    nav = analysis.get("navigator_map") or {}
    ctx = analysis.get("context_output") or {}
    code_samples = analysis.get("code_samples") or {}
    diagram = analysis.get("visualization") or ""

    # 1. Check if the user is asking about a specific file in the repository
    for file_path, content in code_samples.items():
        basename = file_path.split("/")[-1].lower()
        if basename in q or file_path.lower() in q:
            funcs_in_file = [
                f"`{fn['name']}({', '.join(fn.get('params', []))})` — {fn.get('purpose', '')}"
                for fn in ctx.get("key_functions", [])
                if fn.get("file") == file_path
            ]
            classes_in_file = [
                f"`{c['name']}` — {c.get('purpose', '')} (methods: `{', '.join(c.get('methods', []))}`)"
                for c in ctx.get("key_classes", [])
                if c.get("file") == file_path
            ]
            preview = "\n".join(content.splitlines()[:35])
            symbols_md = ""
            if classes_in_file:
                symbols_md += "\n**Classes in this file:**\n" + "\n".join(f"- {c}" for c in classes_in_file)
            if funcs_in_file:
                symbols_md += "\n**Functions in this file:**\n" + "\n".join(f"- {f}" for f in funcs_in_file)

            return f"""### 📄 File Inspection: `{file_path}`
{symbols_md}

#### Source Excerpt:
```python
{preview}
```"""

    # 2. Architecture / Diagram / Data flow queries
    if any(k in q for k in ("diagram", "mermaid", "visual", "architecture", "flow", "topology", "structure")):
        mods_md = "\n".join(
            f"- **`{m.get('path')}`**: {m.get('purpose')}"
            for m in nav.get("core_modules_detailed", [])[:8]
        )
        ep_list = ", ".join(f"`{ep}`" for ep in nav.get("entry_points", [])) or "N/A"
        return f"""### 🗺️ Component Architecture & Data Flow ({meta.get('full_name')})

- **Architecture Pattern**: `{nav.get('architecture_type', 'unknown')}`
- **Entry Points**: {ep_list}

```mermaid
{diagram}
```

#### Core Modules & Responsibilities
{mods_md or '- Standard modular repository layout'}
"""

    # 3. Entry points / startup queries
    if any(k in q for k in ("entry", "start", "launch", "boot", "run")):
        eps = nav.get("entry_points", [])
        ep_details = []
        for ep in eps[:4]:
            snippet = "\n".join((code_samples.get(ep) or "").splitlines()[:20])
            ep_details.append(f"#### `{ep}`\n```python\n{snippet or '# Entry point file'}\n```")
        return f"""### 🚪 Application Entry Points

The primary bootstrap files for **{meta.get('full_name')}** are:
{chr(10).join(f'- `{ep}`' for ep in eps) or '- No explicit entry point script detected'}

{chr(10).join(ep_details)}"""

    # 4. Onboarding / learning path queries
    if any(k in q for k in ("onboard", "learn", "first", "roadmap", "guide", "mentor", "read")):
        return f"""### 🧭 Recommended Developer Onboarding Roadmap

```text
{analysis.get('mentor_guide', 'N/A')}
```

**Tip**: Start with the entry points ({', '.join(f'`{e}`' for e in nav.get('entry_points', [])[:3])}) and then trace calls into the core modules ({', '.join(f'`{m}`' for m in nav.get('core_modules', [])[:4])})."""

    # 5. Design patterns / stack / dependencies / technologies queries
    if any(k in q for k in ("pattern", "stack", "tech", "dependenc", "librar", "framework")):
        techs = ", ".join(f"`{t}`" for t in ctx.get("technologies", [])) or "Standard library"
        patterns = "\n".join(f"- **{p}**" for p in ctx.get("patterns", [])) or "- Modular design"
        deps = "\n".join(f"- `{d}`" for d in nav.get("dependencies", [])[:15]) or "- None listed in manifest"
        return f"""### 🧩 Design Patterns & Technology Stack

#### Detected Design Patterns
{patterns}

#### Frameworks & Libraries
{techs}

#### Manifest Dependencies
{deps}"""

    # 6. Functions / classes / API / models queries
    if any(k in q for k in ("function", "class", "method", "api", "endpoint", "route", "model", "schema")):
        fns = "\n".join(
            f"- **`{f.get('name')}({', '.join(f.get('params', []))})`** (`{f.get('file')}`): {f.get('purpose')}"
            for f in ctx.get("key_functions", [])[:8]
        )
        clss = "\n".join(
            f"- **`{c.get('name')}`** (`{c.get('file')}`): {c.get('purpose')}"
            for c in ctx.get("key_classes", [])[:8]
        )
        eps = "\n".join(
            f"- **`{e.get('method')} {e.get('path')}`** (`{e.get('file')}`)"
            for e in ctx.get("api_endpoints", [])[:8]
        )
        return f"""### 🔍 Key Code Symbols & Components

#### Key Classes
{clss or '- No classes detected'}

#### Key Functions
{fns or '- No standalone functions detected'}

#### API Endpoints
{eps or '- No HTTP routes detected'}"""

    # 7. Commits / PRs queries
    if any(k in q for k in ("commit", "pr", "pull request", "history", "recent", "author")):
        commits = "\n".join(
            f"- `{c.get('sha')}` **{c.get('message')}** _(by {c.get('author')}, {c.get('date')})_"
            for c in analysis.get("recent_commits", [])[:8]
        )
        prs = "\n".join(
            f"- **#{p.get('number')} {p.get('title')}** [`{p.get('state')}`] by `{p.get('author')}`"
            for p in analysis.get("pull_requests", [])[:8]
        )
        return f"""### 📜 Repository Activity

#### Recent Commits
{commits or '- No recent commits available'}

#### Pull Requests
{prs or '- No pull requests available'}"""

    # Default comprehensive overview
    return f"""### 🧭 Repository Intelligence Summary: **{meta.get('full_name')}**

{nav.get('project_summary', '')}

- **Architecture**: `{nav.get('architecture_type', 'unknown')}`
- **Entry Points**: {', '.join(f'`{e}`' for e in nav.get('entry_points', [])[:4])}
- **Core Modules**: {', '.join(f'`{m}`' for m in nav.get('core_modules', [])[:6])}
- **Key Technologies**: {', '.join(f'`{t}`' for t in ctx.get('technologies', [])[:8])}
- **Key Classes**: {', '.join(f'`{c.get("name")}`' for c in ctx.get('key_classes', [])[:5]) or 'None'}
- **Key Functions**: {', '.join(f'`{f.get("name")}()`' for f in ctx.get('key_functions', [])[:6]) or 'None'}

You can ask me about any specific file (e.g. `{nav.get('entry_points', ['app.py'])[0]}`), request the **architecture diagram**, explore **design patterns**, or view the **onboarding roadmap**!"""


def get_chat_response(
    context: str,
    chat_history: List[Tuple[str, str]],
    user_msg: str,
    analysis: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Query the conversational model with repository context and recent dialogue history,
    gracefully falling back to the built-in grounded repository responder on API errors.
    """
    dialogue_lines = [context, "\n\n## Dialogue History\n"]
    for speaker, text in chat_history[-6:]:
        label = "Developer" if speaker == "user" else "RepoCompass"
        dialogue_lines.append(f"{label}: {text}\n\n")
    dialogue_lines.append(f"Developer: {user_msg}\n\nRepoCompass:")

    full_prompt = "".join(dialogue_lines)
    try:
        return invoke_llm_text(chat_llm, full_prompt)
    except Exception:
        if analysis:
            return _synthesize_grounded_fallback_answer(analysis, user_msg)
        return "Please run repository analysis first so I can answer questions using verified repository data."


def _format_initial_digest(analysis: Dict[str, Any]) -> str:
    """Create the initial assistant greeting message summarizing the analyzed repository."""
    meta = analysis.get("metadata") or {}
    nav = analysis.get("navigator_map") or {}
    ctx = analysis.get("context_output") or {}

    techs = ", ".join(ctx.get("technologies", [])[:8]) or "Standard library"
    entries = ", ".join(nav.get("entry_points", [])[:4]) or "None identified"

    return f"""### 🧭 Repository Intelligence Ready: **{meta.get('full_name', 'Repository')}**

I have completed the 5-stage inspection pipeline. Here is an executive snapshot:

- **Architecture Style**: `{nav.get('architecture_type', 'unknown')}`
- **Primary Language**: `{meta.get('language', 'Unknown')}` (⭐ {meta.get('stars', 0):,} stars)
- **Entry Points**: `{entries}`
- **Detected Stack**: {techs}
- **Complexity Index**: `{ctx.get('complexity_score', 'N/A')} / 1.0`
- **Indexed Scope**: `{len(analysis.get('file_tree', []))}` files (`{len(analysis.get('code_samples', {}))}` source files parsed)

**Summary**: {nav.get('project_summary', 'N/A')}

---
💬 **Ask any technical question below**, or explore the dedicated tabs above for **Architecture & Topology**, **Code Intelligence**, **Onboarding & Dossier**, and **Activity & File Explorer**."""


# --- Session State Bootstrap ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "repo_url" not in st.session_state:
    st.session_state.repo_url = ""
if "context" not in st.session_state:
    st.session_state.context = ""


# --- Sidebar Control Center ---
with st.sidebar:
    st.markdown(
        """
        <div style='text-align: left; padding: 0.75rem 0 1rem 0;'>
            <div style='font-size: 0.75rem; color: #10b981; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;'>
                Multi-Agent Codebase Studio
            </div>
            <h1 style='color: #f8fafc; font-size: 2.1rem; margin: 0.2rem 0;'>🧭 RepoCompass</h1>
            <p style='color: #94a3b8; margin: 0; font-size: 0.9rem;'>
                Automated Architecture Mapping, Code Inspection & Developer Onboarding
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 🔗 Target GitHub Repository")
    repo_url_input = st.text_input(
        "GitHub Repository URL",
        value=st.session_state.repo_url,
        placeholder="https://github.com/owner/repository",
        label_visibility="collapsed",
    )

    launch_analysis = st.button(
        "⚡ Run 5-Agent Inspection", type="primary", use_container_width=True
    )

    if st.session_state.analysis:
        if st.button("🔄 Reset Workspace", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.analysis = None
            st.session_state.repo_url = ""
            st.session_state.context = ""
            st.rerun()

    st.markdown("---")
    with st.expander("🔑 Engine & API Configuration", expanded=False):
        engine_status = "🟢 OpenAI LLM Active" if has_active_api_key() else "⚡ Built-in AST & Graph Engine Active"
        st.caption(f"Current Mode: **{engine_status}**")
        user_api_key = st.text_input(
            "OpenAI API Key (Optional)",
            type="password",
            placeholder="sk-... (leave blank for built-in AST engine)",
        )
        user_base_url = st.text_input(
            "Custom Base URL (Optional)",
            placeholder="https://api.openai.com/v1",
        )
        if user_api_key:
            set_runtime_api_config(user_api_key, user_base_url)
            st.success("API key applied for live LLM calls.")

    with st.expander("🛠️ 5-Stage Pipeline Architecture", expanded=False):
        st.markdown(
            """
            1. **Structure Mapper**: Maps directory hierarchy, entry points & manifests.
            2. **Code Inspector**: Extracts key classes, functions, APIs & design patterns.
            3. **Onboarding Architect**: Builds a sequenced learning curriculum.
            4. **Diagram Synthesizer**: Generates Mermaid topology flowcharts.
            5. **Report Compiler**: Synthesizes the technical onboarding dossier.
            """
        )

    if st.session_state.analysis and st.session_state.analysis.get("messages"):
        with st.expander("📡 Agent Telemetry Log", expanded=True):
            for log_entry in st.session_state.analysis.get("messages", []):
                st.caption(f"✓ {log_entry}")


# --- Pipeline Execution Trigger ---
if launch_analysis and repo_url_input:
    st.session_state.chat_history = []
    st.session_state.analysis = None
    st.session_state.repo_url = repo_url_input.strip()

    with st.status("🧭 Executing RepoCompass 5-Stage Inspection Pipeline...", expanded=True) as status_box:
        try:
            st.write("🔌 Initializing GitHub API & clone service...")
            gh_service = GitHubClient()

            st.write("📥 Cloning repository & running specialist LangGraph agents...")
            result_state = run_analysis(st.session_state.repo_url, gh_service)

            st.session_state.analysis = result_state
            st.session_state.context = build_context(result_state)
            st.session_state.chat_history.append(
                ("assistant", _format_initial_digest(result_state))
            )

            status_box.update(label="✅ Repository Inspection Complete!", state="complete")
        except Exception as err:
            status_box.update(label=f"❌ Inspection Failed: {str(err)[:90]}", state="error")
            st.error(
                f"**Unable to complete repository inspection**\n\n`{err}`\n\n"
                "Verify that the repository URL is public, valid, and reachable."
            )


# --- Main Studio Banner ---
st.markdown(
    """
<div class='atlas-banner'>
    <h1>🧭 RepoCompass Intelligence Studio</h1>
    <p>Deep architectural mapping, static symbol extraction, visual topology diagrams, and interactive codebase Q&A</p>
</div>
""",
    unsafe_allow_html=True,
)


# --- Landing View vs. Analyzed Workspace View ---
if not st.session_state.analysis:
    left_col, right_col = st.columns([1.15, 1])

    with left_col:
        st.markdown("### ⚡ Autonomous 5-Agent Inspection Pipeline")
        st.markdown(
            """
            <div class='stage-card'>
                <h4>1. Structure Mapper (Navigator)</h4>
                <p>Constructs the complete file hierarchy, identifies application entry points, and parses manifest dependencies.</p>
            </div>
            <div class='stage-card'>
                <h4>2. Code Inspector (Context)</h4>
                <p>Performs static analysis on core modules to extract key functions, classes, API endpoints, data schemas, and design patterns.</p>
            </div>
            <div class='stage-card'>
                <h4>3. Onboarding Architect (Mentor)</h4>
                <p>Generates a tailored developer learning path with prerequisites, time estimates, and file-by-file reading order.</p>
            </div>
            <div class='stage-card'>
                <h4>4. Diagram Synthesizer (Visualizer)</h4>
                <p>Builds interactive Mermaid architecture flowcharts illustrating component relationships and data flow.</p>
            </div>
            <div class='stage-card'>
                <h4>5. Report Compiler (Orchestrator)</h4>
                <p>Consolidates all findings into a structured technical dossier ready for engineering teams.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown("### 🎯 Quick-Launch Sample Repositories")
        st.info(
            "Paste any public GitHub URL in the left sidebar and click **Run 5-Agent Inspection**, "
            "or click a sample repository below to load its URL."
        )
        sample_repos = [
            ("RepoCompass (Self-Inspection)", "https://github.com/mennatallahessam/RepoCompass"),
            ("Flask Microframework", "https://github.com/pallets/flask"),
            ("FastAPI Framework", "https://github.com/tiangolo/fastapi"),
        ]
        for label, url in sample_repos:
            if st.button(f"📂 {label} — {url.split('github.com/')[-1]}", use_container_width=True):
                st.session_state.repo_url = url
                st.rerun()

        st.markdown("---")
        st.markdown(
            """
            #### 💡 Studio Capabilities
            - **Interactive Codebase Chat**: Grounded Q&A referencing real files, commits, and PRs.
            - **Live Mermaid Diagrams**: Visual component graphs rendered directly in the browser.
            - **Zero-Key Static AST Fallback**: Works out-of-the-box with built-in AST & Mermaid synthesis or with your own OpenAI API key.
            """
        )

else:
    analysis = st.session_state.analysis
    meta = analysis.get("metadata") or {}
    nav = analysis.get("navigator_map") or {}
    ctx = analysis.get("context_output") or {}

    # --- Top KPI Metrics Row ---
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(
            f"""<div class='kpi-tile'>
                <div class='kpi-label'>Repository</div>
                <div class='kpi-value' title='{meta.get("full_name", "")}'>{meta.get("name") or meta.get("full_name", "Unknown")}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""<div class='kpi-tile'>
                <div class='kpi-label'>Architecture</div>
                <div class='kpi-value' title='{nav.get("architecture_type", "unknown")}'>{nav.get("architecture_type", "unknown")}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""<div class='kpi-tile'>
                <div class='kpi-label'>Language & Stars</div>
                <div class='kpi-value'>{meta.get("language", "N/A")} · ⭐ {meta.get("stars", 0):,}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""<div class='kpi-tile'>
                <div class='kpi-label'>Complexity Index</div>
                <div class='kpi-value'>{ctx.get("complexity_score", 0.0)} / 1.0</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            f"""<div class='kpi-tile'>
                <div class='kpi-label'>Indexed Files</div>
                <div class='kpi-value'>{len(analysis.get("code_samples", {}))} / {len(analysis.get("file_tree", []))}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # --- Multi-Tab Studio Interface ---
    tab_chat, tab_arch, tab_code, tab_onboard, tab_activity = st.tabs([
        "💬 Interactive Q&A Chat",
        "🗺️ Architecture & Topology",
        "🔍 Code Intelligence",
        "🎓 Onboarding & Dossier",
        "📜 Activity & File Explorer",
    ])

    # ==================== TAB 1: INTERACTIVE Q&A CHAT ====================
    with tab_chat:
        st.markdown("##### ⚡ Quick Prompts")
        qp1, qp2, qp3, qp4 = st.columns(4)
        triggered_prompt = None
        if qp1.button("📊 Render Architecture Diagram", use_container_width=True):
            triggered_prompt = "Show me the Mermaid architecture diagram and explain the data flow."
        if qp2.button("🚪 Explain Entry Points", use_container_width=True):
            triggered_prompt = "What are the entry points of this application and how does startup work?"
        if qp3.button("🧭 Onboarding Roadmap", use_container_width=True):
            triggered_prompt = "What should a new developer read first and in what order?"
        if qp4.button("🧩 Design Patterns & Stack", use_container_width=True):
            triggered_prompt = "What design patterns and core libraries are used across the codebase?"

        # Render conversation messages
        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                if role == "assistant":
                    render_message_with_mermaid(msg)
                else:
                    st.markdown(msg)

        chat_box_value = st.chat_input("Ask RepoCompass about files, functions, architecture, or commits...")
        active_query = triggered_prompt or chat_box_value

        if active_query:
            st.session_state.chat_history.append(("user", active_query))
            with st.chat_message("user"):
                st.markdown(active_query)

            with st.chat_message("assistant"):
                with st.spinner("Synthesizing answer from repository telemetry..."):
                    ai_reply = get_chat_response(
                        st.session_state.context,
                        st.session_state.chat_history,
                        active_query,
                        analysis=analysis,
                    )
                render_message_with_mermaid(ai_reply)
                st.session_state.chat_history.append(("assistant", ai_reply))
            if triggered_prompt:
                st.rerun()

    # ==================== TAB 2: ARCHITECTURE & TOPOLOGY ====================
    with tab_arch:
        st.markdown("### 🗺️ Component Architecture Diagram")
        diagram_code = analysis.get("visualization")
        if diagram_code and not diagram_code.startswith("Error"):
            st_mermaid(diagram_code, height=460)
            with st.expander("View Raw Mermaid Definition"):
                st.code(diagram_code, language="mermaid")
        else:
            st.warning("No Mermaid diagram was generated for this repository.")

        st.markdown("---")
        col_ep, col_mod = st.columns([1, 1.5])
        with col_ep:
            st.markdown("#### 🚪 Entry Points")
            for ep in nav.get("entry_points", []):
                st.markdown(f"- `{ep}`")
            st.markdown("#### 📝 Project Summary")
            st.write(nav.get("project_summary", "N/A"))

        with col_mod:
            st.markdown("#### 🧱 Core Modules Breakdown")
            detailed_mods = nav.get("core_modules_detailed", [])
            if detailed_mods:
                for mod in detailed_mods:
                    st.markdown(f"- **`{mod.get('path', '')}`** — {mod.get('purpose', '')}")
            else:
                for mod_name in nav.get("core_modules", []):
                    st.markdown(f"- `{mod_name}`")

    # ==================== TAB 3: CODE INTELLIGENCE ====================
    with tab_code:
        st.markdown("### 🛠️ Detected Technologies & Design Patterns")
        tech_pills = "".join(
            f"<span class='tech-pill'>{t}</span>" for t in ctx.get("technologies", [])
        )
        pattern_pills = "".join(
            f"<span class='tech-pill' style='border-color:#06b6d4;color:#67e8f9;background:rgba(6,182,212,0.14);'>{p}</span>"
            for p in ctx.get("patterns", [])
        )
        st.markdown(tech_pills or "_No external libraries detected_", unsafe_allow_html=True)
        st.markdown(pattern_pills or "", unsafe_allow_html=True)

        st.markdown("---")
        fn_col, cls_col = st.columns(2)
        with fn_col:
            st.markdown("#### ⚡ Key Functions")
            for fn in ctx.get("key_functions", []):
                params_str = ", ".join(fn.get("params", []))
                st.markdown(
                    f"- **`{fn.get('name', '')}({params_str})`** in `{fn.get('file', '')}`\n"
                    f"  _{fn.get('purpose', '')}_"
                )
        with cls_col:
            st.markdown("#### 🏛️ Key Classes")
            for cls_item in ctx.get("key_classes", []):
                methods_str = ", ".join(cls_item.get("methods", [])[:6])
                st.markdown(
                    f"- **`{cls_item.get('name', '')}`** in `{cls_item.get('file', '')}`\n"
                    f"  _{cls_item.get('purpose', '')}_ (Methods: `{methods_str}`)"
                )

        st.markdown("---")
        ep_col, dm_col = st.columns(2)
        with ep_col:
            st.markdown("#### 🌐 API Endpoints")
            endpoints = ctx.get("api_endpoints", [])
            if endpoints:
                for ep in endpoints:
                    st.markdown(
                        f"- **`{ep.get('method', 'GET')} {ep.get('path', '')}`** (`{ep.get('file', '')}`) — {ep.get('purpose', '')}"
                    )
            else:
                st.caption("No HTTP/API routes detected.")

        with dm_col:
            st.markdown("#### 🗃️ Data Schemas & Models")
            models = ctx.get("data_models", [])
            if models:
                for dm in models:
                    fields_str = ", ".join(dm.get("fields", []))
                    st.markdown(
                        f"- **`{dm.get('name', '')}`** (`{dm.get('file', '')}`): `{fields_str}`"
                    )
            else:
                st.caption("No structured data models detected.")

    # ==================== TAB 4: ONBOARDING & DOSSIER ====================
    with tab_onboard:
        onboard_col, report_col = st.columns([1, 1.3])
        with onboard_col:
            st.markdown("### 🎓 Developer Onboarding Curriculum")
            st.code(analysis.get("mentor_guide", "N/A"), language="text")
        with report_col:
            st.markdown("### 📋 Synthesized Technical Dossier")
            st.markdown(analysis.get("final_report", "No report generated."))

    # ==================== TAB 5: ACTIVITY & FILE EXPLORER ====================
    with tab_activity:
        c_col, pr_col = st.columns(2)
        with c_col:
            st.markdown("#### 🕒 Recent Commits")
            commits = analysis.get("recent_commits", [])
            if commits:
                for c in commits:
                    st.markdown(
                        f"- `{c.get('sha', '')}` **{c.get('message', '')}** _(by {c.get('author', '')}, {c.get('date', '')})_"
                    )
            else:
                st.caption("No commit history available.")

        with pr_col:
            st.markdown("#### 🔀 Recent Pull Requests")
            prs = analysis.get("pull_requests", [])
            if prs:
                for pr in prs:
                    st.markdown(
                        f"- **#{pr.get('number', '')} {pr.get('title', '')}** [`{pr.get('state', '')}`] by `{pr.get('author', '')}`"
                    )
            else:
                st.caption("No pull requests found.")

        st.markdown("---")
        st.markdown(f"#### 📁 Indexed File Tree ({len(analysis.get('file_tree', []))} files)")
        all_paths = [f["path"] for f in analysis.get("file_tree", [])]
        st.code("\n".join(all_paths[:300]), language="text")
