"""Command line for the harness: `python -m EvoScientist.research_tree <status|finalize> --root DIR`.

The harness uses `finalize` after the agent has stopped, so the record is
settled (outcomes, final tag, TRAJECTORY.md, the champion's deliverable at the
root) even when the agent ended without calling `tree_finalize`.
"""

from __future__ import annotations

import argparse
import sys

from .tree import ResearchTree, TreeError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m EvoScientist.research_tree")
    parser.add_argument("command", choices=("status", "finalize"))
    parser.add_argument(
        "--root", required=True, help="project root (the agent's workspace)"
    )
    args = parser.parse_args(argv)
    tree = ResearchTree(args.root)
    try:
        print(tree.status() if args.command == "status" else tree.finalize())
    except TreeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
