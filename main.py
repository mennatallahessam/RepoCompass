#!/usr/bin/env python3
"""
RepoCompass CLI — Command-line runner for the 5-stage GitHub repository analysis pipeline.
"""
import sys
import time
import traceback
from typing import Any, Dict, List

from src.github_client import GitHubClient
from src.graph import run_analysis


def print_repo_info(metadata: Dict[str, Any]) -> None:
    """Display formatted repository metadata in the terminal."""
    summary_text = metadata.get("description") or "No description provided"
    if len(summary_text) > 80:
        summary_text = summary_text[:80] + "..."

    print("\n" + "─" * 78)
    print("  REPOSITORY METADATA")
    print("─" * 78)
    print(f"  Full Name   : {metadata.get('full_name')}")
    print(f"  Language    : {metadata.get('language')}")
    print(f"  Stars       : {metadata.get('stars', 0):,}")
    print(f"  Forks       : {metadata.get('forks', 0):,}")
    print(f"  Description : {summary_text}")
    print("─" * 78)


def print_agent_outputs(state: Dict[str, Any]) -> None:
    """Print each specialist agent's output summary."""
    nav = state.get("navigator_map")
    if nav:
        print("\n[STAGE 1 · STRUCTURE MAPPER]")
        print(f"  Entry Points : {', '.join(nav.get('entry_points', [])[:5])}")
        print(f"  Core Modules : {', '.join(nav.get('core_modules', [])[:5])}")
        print(f"  Architecture : {nav.get('architecture_type', 'unknown')}")
        print(f"  Confidence   : {nav.get('confidence_score', 0):.0%}")

    if state.get("context_summary"):
        print("\n[STAGE 2 · CODE INSPECTOR]")
        print(state["context_summary"])

    if state.get("mentor_guide"):
        print("\n[STAGE 3 · ONBOARDING ARCHITECT]")
        print(state["mentor_guide"])

    vis = state.get("visualization")
    if vis:
        print("\n[STAGE 4 · DIAGRAM SYNTHESIZER]")
        preview = vis[:200] + "..." if len(vis) > 200 else vis
        print(f"  Mermaid Diagram Preview:\n{preview}")


def print_final_report(report: str) -> None:
    """Display the synthesized technical onboarding dossier."""
    print("\n" + "═" * 78)
    print("  TECHNICAL ONBOARDING DOSSIER")
    print("═" * 78)
    print(report)
    print("═" * 78)


def print_agent_log(messages: List[str]) -> None:
    """Display the sequential execution log from all stages."""
    if not messages:
        return
    print("\nPipeline Telemetry Log:")
    for idx, entry in enumerate(messages, start=1):
        print(f"  {idx}. {entry}")


def main() -> None:
    """CLI entry point."""
    print("\n" + "═" * 78)
    print("  RepoCompass — Multi-Agent Repository Intelligence CLI")
    print("═" * 78)

    if len(sys.argv) < 2:
        print("Error: Target GitHub repository URL is required.")
        print("\nUsage  : python main.py <github_repo_url>")
        print("Example: python main.py https://github.com/tiangolo/fastapi")
        sys.exit(1)

    target_url = sys.argv[1]
    print(f"Target Repository: {target_url}\n")

    try:
        client = GitHubClient()
    except Exception as exc:
        print(f"Failed to initialize GitHub client: {exc}")
        sys.exit(1)

    t0 = time.time()

    try:
        final_state = run_analysis(target_url, client)
        duration = time.time() - t0

        print(f"\nPipeline completed in {duration:.1f}s")
        print_repo_info(final_state["metadata"])
        print_agent_outputs(final_state)

        if final_state.get("final_report"):
            print_final_report(final_state["final_report"])

        print_agent_log(final_state.get("messages", []))

        if final_state.get("errors"):
            print("\nWarnings / Errors:")
            for err in final_state["errors"]:
                print(f"  - {err}")

        print(f"\nTotal Runtime : {duration:.1f}s")
        print(f"Stages Run    : {len(final_state.get('messages', []))}")
    except Exception as exc:
        print(f"\nUnhandled error during analysis: {exc}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
