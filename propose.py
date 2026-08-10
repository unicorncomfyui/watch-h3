#!/usr/bin/env python3
"""Open a pull request on the pod repository for the mechanical changes.

WHAT THIS WILL AND WILL NOT DO.

It proposes exactly one kind of change: moving a pinned custom-node SHA
forward. That is mechanical - the new value is a fact, not a judgement, and
the diff is one line per node. Everything else the watcher notices stays a
checkbox in the issue, because everything else requires a decision:

  - a new LoRA is a quality trade-off, and this project has measured three of
    them going the opposite way to expectation;
  - a ComfyUI bump breaks pinned nodes, and one of them is already known to
    break on any frontend change;
  - a new wheel needs a compile and a bench before it means anything.

The pull request is a PROPOSAL WITH A DIFF, never a merge. It targets
`develop`, never `main`, so the branch model does its job: the image builds as
`cu130-develop`, a test pod pulls it, and the bench decides. Auto-merge is
deliberately absent - the whole session that produced this file consisted of
measurements contradicting confident expectations.

Needs a fine-grained token with Contents and Pull requests write on the pod
repository alone, in GH_TOKEN. Without it the script exits quietly: a
repository clone of this project must work with no secret at all.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

POD = os.environ.get("POD_REPO", "unicorncomfyui/pod-comfyui-h3").strip()
BASE = os.environ.get("POD_BASE_BRANCH", "develop").strip()
# One commit is a README; ten is a change of behaviour. Below this a pull
# request costs more attention than the drift it reports.
THRESHOLD = int(os.environ.get("BUMP_THRESHOLD", "10"))


def run(*args: str, cwd: str | None = None, check: bool = True) -> str:
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if check and p.returncode:
        raise RuntimeError(f"{' '.join(args[:3])}: {p.stderr.strip()[:400]}")
    return p.stdout.strip()


def candidates(state: dict) -> dict[str, dict]:
    """Pinned nodes far enough behind to be worth a proposal."""
    pins = state.get("pins/pod-comfyui-h3") or {}
    return {repo: v for repo, v in pins.items()
            if isinstance(v, dict) and v.get("head")
            and v.get("behind", 0) >= THRESHOLD}


def rewrite(dockerfile: Path, repo: str, new_sha: str) -> tuple[str, str] | None:
    """Replace one pin, matched through its repository rather than its SHA.

    Anchoring on the repository URL is what keeps this honest: matching the
    old SHA alone would rewrite any other line that happened to carry it, and
    matching a line number would break the first time the file is reordered.
    """
    src = dockerfile.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(github\.com/" + re.escape(repo) + r"\.git\s*\\\s*"
        r"&& git -C [\w.-]+ checkout -q )([0-9a-f]{40})")
    m = pattern.search(src)
    if not m or m.group(2) == new_sha:
        return None
    dockerfile.write_text(pattern.sub(r"\g<1>" + new_sha, src, count=1),
                          encoding="utf-8")
    return m.group(2), new_sha


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", default="state.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # A Windows console defaults to cp1252 and raises on the first arrow in a
    # diff summary. Losing a dry run to the act of PRINTING it would be an
    # absurd failure, so force the stream and let anything unmappable degrade.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    token = os.environ.get("GH_TOKEN", "").strip()
    if not token and not args.dry_run:
        print("No GH_TOKEN: nothing proposed. This is not an error - the "
              "watcher works without it, it simply reports instead.")
        return 0

    state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    todo = candidates(state)
    if not todo:
        print(f"No pin is {THRESHOLD}+ commits behind; nothing to propose.")
        return 0

    work = tempfile.mkdtemp()
    repo_dir = str(Path(work, "pod"))
    url = f"https://x-access-token:{token}@github.com/{POD}.git" if token \
        else f"https://github.com/{POD}.git"
    run("git", "clone", "--depth", "1", "-b", BASE, url, repo_dir)
    run("git", "config", "user.name", "watch-h3", cwd=repo_dir)
    run("git", "config", "user.email",
        "watch-h3@users.noreply.github.com", cwd=repo_dir)

    dockerfile = Path(repo_dir, "Dockerfile")
    applied: list[str] = []
    for repo, info in sorted(todo.items()):
        changed = rewrite(dockerfile, repo, info["head"])
        if changed:
            old, new = changed
            applied.append(f"- `{repo}` `{old[:12]}` → `{new[:12]}` "
                           f"({info['behind']} commits, latest: "
                           f"*{info.get('latest', '?')}*)")

    if not applied:
        print("Every pin already matches upstream in the checked-out branch.")
        return 0

    branch = "watch/bump-pinned-nodes"
    run("git", "checkout", "-B", branch, cwd=repo_dir)
    run("git", "add", "Dockerfile", cwd=repo_dir)
    body = (
        "Proposed by [watch-h3](https://github.com/unicorncomfyui/watch-h3). "
        "**Not verified — this only moves the pins.**\n\n"
        + "\n".join(applied)
        + "\n\n### Before merging\n\n"
        "- [ ] `cu130-develop` builds\n"
        "- [ ] a test pod starts and ComfyUI loads every workflow "
        "(VideoHelperSuite breaks on frontend changes, and a `VHS_` node in a "
        "graph stops it loading entirely)\n"
        "- [ ] `scripts/bench.py --config baseline --config sage` shows no "
        "regression\n\n"
        "Targets `" + BASE + "` on purpose: the branch model exists so that "
        "an unverified change reaches a test pod and never `cu130` or "
        "`latest`.\n")

    if args.dry_run:
        print(f"[dry-run] branch {branch} on {POD}, targeting {BASE}")
        print(run("git", "diff", "--cached", cwd=repo_dir)[:1500])
        print("\n--- pull request body ---\n" + body)
        return 0

    run("git", "commit", "-m",
        "chore(nodes): move pinned custom nodes forward\n\n"
        "Opened automatically because these pins drifted past "
        f"{THRESHOLD} commits. The SHAs are facts; whether the image still "
        "works with them is not, which is what the checklist is for.",
        cwd=repo_dir)
    run("git", "push", "-f", "origin", branch, cwd=repo_dir)

    env = {**os.environ, "GH_TOKEN": token}
    existing = subprocess.run(
        ["gh", "pr", "list", "--repo", POD, "--head", branch, "--json", "number"],
        capture_output=True, text=True, env=env).stdout
    if existing and json.loads(existing or "[]"):
        # The branch is force-pushed, so an open pull request already carries
        # the newest proposal. Opening a second one would just split the
        # conversation.
        print("An open pull request already tracks this branch; updated it.")
        return 0

    out = subprocess.run(
        ["gh", "pr", "create", "--repo", POD, "--base", BASE, "--head", branch,
         "--title", "chore(nodes): move pinned custom nodes forward",
         "--body", body],
        capture_output=True, text=True, env=env)
    print(out.stdout or out.stderr)
    return 0 if out.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
