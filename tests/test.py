"""Unit sanity checks for RepoCompass core utilities, client, and workflow compilation."""
from src.agents.context_agent import _select_priority_files
from src.agents.navigator_agent import _build_tree_view
from src.github_client import GitHubClient
from src.graph import create_agent_graph
from src.utils import extract_json


def run_checks() -> None:
    client = GitHubClient()
    assert client.parse_repo_url("https://github.com/tiangolo/fastapi") == ("tiangolo", "fastapi")
    assert client.parse_repo_url("https://github.com/pallets/flask.git/") == ("pallets", "flask")

    malformed = '```json\n{"deps": ["flask": "^2.0", "requests"],}\n```'
    parsed = extract_json(malformed)
    assert parsed == {"deps": ["flask ^2.0", "requests"]}

    sample_tree = [
        {"path": "app.py", "type": "blob", "size": 100},
        {"path": "src/graph.py", "type": "blob", "size": 200},
    ]
    tree_str = _build_tree_view(sample_tree)
    assert "app.py" in tree_str and "src/" in tree_str

    prioritized = _select_priority_files(
        {"other.py": "x", "app.py": "y", "src/core.py": "z"},
        {"entry_points": ["app.py"], "core_modules": ["src/"]},
        max_files=2,
    )
    assert "app.py" in prioritized and "src/core.py" in prioritized

    compiled = create_agent_graph()
    assert compiled is not None
    print("All RepoCompass unit checks passed successfully!")


if __name__ == "__main__":
    run_checks()

