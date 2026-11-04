"""
Standalone verification script for the Structure Mapper (Navigator Agent).
"""
import os
import traceback
from dotenv import load_dotenv
from src.agents.navigator_agent import navigator_agent
from src.github_client import GitHubClient
from src.state import build_initial_state

load_dotenv()


def test_navigator() -> None:
    """Run Structure Mapper on a sample repository and display results."""
    client = GitHubClient(token=os.getenv("GITHUB_TOKEN"))
    sample_url = "https://github.com/JanaElfeky/CustomerChurnPredictor"
    print(f"[RepoCompass] Inspecting sample repository: {sample_url}\n")

    try:
        owner, repo_name = client.parse_repo_url(sample_url)
        metadata = client.get_repo_metadata(owner, repo_name)

        clone_dir = client.clone_repo(sample_url)
        try:
            file_tree = client.walk_local_repo(clone_dir)
            readme_content = client.read_local_readme(clone_dir)
            config_files = client.read_local_config_files(clone_dir, file_tree)
        finally:
            client.cleanup_clone(clone_dir)

        state = build_initial_state(
            repo_url=sample_url,
            owner=owner,
            repo_name=repo_name,
            metadata=metadata,
            file_tree=file_tree,
            code_samples={},
            readme_content=readme_content,
            config_files=config_files,
            recent_commits=[],
            pull_requests=[],
        )

        outcome = navigator_agent(state)
        nav_map = outcome.get("navigator_map") or {}

        print("─" * 78)
        print("STRUCTURE MAPPER RESULTS")
        print("─" * 78)
        print(f"Entry Points : {nav_map.get('entry_points', [])}")
        print(f"Core Modules : {nav_map.get('core_modules', [])}")
        print(f"Dependencies : {nav_map.get('dependencies', [])}")
        print(f"Architecture : {nav_map.get('architecture_type', 'unknown')}")
        print(f"Confidence   : {nav_map.get('confidence_score', 0):.0%}")
        print("─" * 78)
    except Exception as exc:
        print(f"Structure Mapper test error: {exc}")
        traceback.print_exc()


if __name__ == "__main__":
    test_navigator()
