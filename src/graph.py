"""
LangGraph multi-stage pipeline builder and execution coordinator for RepoCompass.
"""
from langgraph.graph import END, StateGraph
from src.agents.context_agent import context_agent
from src.agents.mentor_agent import mentor_agent
from src.agents.navigator_agent import navigator_agent
from src.agents.orchestrator_agent import orchestrator_agent
from src.agents.visualizer_agent import visualizer_agent
from src.state import AgentState, build_initial_state

PIPELINE_STAGES = (
    ("navigator", navigator_agent),
    ("context", context_agent),
    ("mentor", mentor_agent),
    ("visualizer", visualizer_agent),
    ("orchestrator", orchestrator_agent),
)


def create_agent_graph():
    """
    Assemble and compile the sequential 5-stage LangGraph analysis pipeline:
    navigator -> context -> mentor -> visualizer -> orchestrator -> END
    """
    graph_builder = StateGraph(AgentState)

    for stage_name, stage_fn in PIPELINE_STAGES:
        graph_builder.add_node(stage_name, stage_fn)

    graph_builder.set_entry_point(PIPELINE_STAGES[0][0])

    for idx in range(len(PIPELINE_STAGES) - 1):
        current_stage = PIPELINE_STAGES[idx][0]
        next_stage = PIPELINE_STAGES[idx + 1][0]
        graph_builder.add_edge(current_stage, next_stage)

    graph_builder.add_edge(PIPELINE_STAGES[-1][0], END)
    return graph_builder.compile()


def run_analysis(repo_url: str, github_client) -> AgentState:
    """
    Acquire repository metadata and files (via GitHub API and local shallow clone)
    and execute the compiled 5-stage LangGraph pipeline.
    """
    owner, repo_name = github_client.parse_repo_url(repo_url)

    print(f"[RepoCompass] Querying repository metadata for {owner}/{repo_name}...")
    metadata = github_client.get_repo_metadata(owner, repo_name)

    print("[RepoCompass] Performing shallow clone of target repository...")
    clone_path = github_client.clone_repo(repo_url)

    try:
        print("[RepoCompass] Indexing repository file hierarchy...")
        file_tree = github_client.walk_local_repo(clone_path)

        print(f"[RepoCompass] Extracting source and document files ({len(file_tree)} total files)...")
        code_samples = github_client.read_all_source_files(clone_path, file_tree)

        print("[RepoCompass] Reading primary README documentation...")
        readme_content = github_client.read_local_readme(clone_path)

        print("[RepoCompass] Reading project manifests and configuration files...")
        config_files = github_client.read_local_config_files(clone_path, file_tree)
    finally:
        print("[RepoCompass] Removing temporary clone directory...")
        github_client.cleanup_clone(clone_path)

    print("[RepoCompass] Retrieving commit history and pull requests...")
    recent_commits = github_client.get_recent_commits(owner, repo_name)
    pull_requests = github_client.get_pull_requests(owner, repo_name)

    print(
        f"[RepoCompass] Acquisition summary: {len(code_samples)} code files, "
        f"{len(config_files)} manifests, {len(recent_commits)} commits, {len(pull_requests)} PRs"
    )

    state_input = build_initial_state(
        repo_url=repo_url,
        owner=owner,
        repo_name=repo_name,
        metadata=metadata,
        file_tree=file_tree,
        code_samples=code_samples,
        readme_content=readme_content,
        config_files=config_files,
        recent_commits=recent_commits,
        pull_requests=pull_requests,
    )

    compiled_pipeline = create_agent_graph()
    print("[RepoCompass] Executing 5-stage specialist agent pipeline...\n")
    return compiled_pipeline.invoke(state_input)
