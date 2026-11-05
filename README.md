# 🧭 RepoCompass

**Autonomous Multi-Agent GitHub Repository Architecture & Onboarding Studio**

🚀 **Live Demo:** [repocompass-bgebefkbrnkxvppmzvdkqv.streamlit.app](https://repocompass-bgebefkbrnkxvppmzvdkqv.streamlit.app/)

RepoCompass is a 5-stage LangGraph multi-agent system that inspects GitHub repositories to map directory topology, perform static code analysis, construct developer onboarding curriculums, synthesize interactive Mermaid architecture diagrams, and provide grounded conversational Q&A.

---

## ✨ Core Capabilities

- **🗺️ Structural Topology Mapping** — Detects application entry points, core modules, project manifests, and architectural classifications.
- **🔍 Deep Code Inspection** — Extracts pivotal functions, classes, design patterns, HTTP/API routes, data models, and complexity scores.
- **🎓 Onboarding Curriculum Generation** — Produces an ordered file-by-file learning roadmap with prerequisites and time estimates.
- **📊 Visual Architecture Synthesis** — Generates interactive Mermaid component flowcharts directly inside the studio UI.
- **💬 Grounded Codebase Q&A** — Multi-turn conversational assistant grounded in actual repository source files, commit history, and pull requests.

---

## 🏗️ 5-Stage Agent Pipeline

`Structure Mapper` ➔ `Code Inspector` ➔ `Onboarding Architect` ➔ `Diagram Synthesizer` ➔ `Report Compiler`

1. **Structure Mapper (`navigator_agent.py`)** — Analyzes file trees, README documentation, and manifest dependencies.
2. **Code Inspector (`context_agent.py`)** — Inspects prioritized source files for classes, functions, endpoints, and patterns.
3. **Onboarding Architect (`mentor_agent.py`)** — Designs a tailored developer learning path and prerequisite checklist.
4. **Diagram Synthesizer (`visualizer_agent.py`)** — Builds Mermaid `graph TD` component topology diagrams.
5. **Report Compiler (`orchestrator_agent.py`)** — Compiles all upstream telemetry into a unified technical dossier.

---

## 📁 Project Layout

```text
RepoCompass/
├── app.py                        # Multi-tab Streamlit Studio UI
├── main.py                       # Command-line pipeline runner
├── requirements.txt              # Python package dependencies
├── src/
│   ├── github_client.py          # GitHub REST API + local shallow-clone indexer
│   ├── graph.py                  # LangGraph state machine builder & runner
│   ├── state.py                  # Shared AgentState schema
│   ├── utils.py                  # LLM factory & resilient JSON parser
│   └── agents/
│       ├── navigator_agent.py    # Stage 1: Structure Mapper
│       ├── context_agent.py      # Stage 2: Code Inspector
│       ├── mentor_agent.py       # Stage 3: Onboarding Architect
│       ├── visualizer_agent.py   # Stage 4: Diagram Synthesizer
│       └── orchestrator_agent.py # Stage 5: Report Compiler
└── tests/
    ├── test_agents.py            # Step-by-step interactive agent test harness
    └── test_navigator.py         # Standalone Structure Mapper test & PDF exporter
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
Create a `.env` file or export your credentials:
```bash
OPENAI_API_KEY="your-api-key"
GITHUB_TOKEN="your-optional-github-token"
```

### 3. Launch the Studio UI
```bash
streamlit run app.py
```

### 4. Run via Command Line
```bash
python main.py https://github.com/tiangolo/fastapi
```
