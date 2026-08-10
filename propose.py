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
import urllib.request
from pathlib import Path

POD = os.environ.get("POD_REPO", "unicorncomfyui/pod-comfyui-h3").strip()
BASE = os.environ.get("POD_BASE_BRANCH", "develop").strip()
IMAGE = os.environ.get("POD_IMAGE", "vlop12ui/pod-comfyui-h3").strip()
# One commit is a README; ten is a change of behaviour. Below this a pull
# request costs more attention than the drift it reports.
THRESHOLD = int(os.environ.get("BUMP_THRESHOLD", "10"))


def run(*args: str, cwd: str | None = None, check: bool = True) -> str:
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if check and p.returncode:
        raise RuntimeError(f"{' '.join(args[:3])}: {p.stderr.strip()[:400]}")
    return p.stdout.strip()


# Reading the diff is a different privilege from writing the pull request, so
# it uses a different token when one is provided. The Actions GITHUB_TOKEN is
# the right one: 1000 requests/hour and no write access anywhere. Falls back to
# the write token, which a fine-grained PAT still lets read public repositories
# with, and then to anonymous - 60/hour, enough for a handful of pins.
READ_TOKEN = os.environ.get("READ_TOKEN", "").strip()

# Whether a changed file can move what the image DOES. A registry of node
# metadata cannot; a sampler can. This is the whole basis of the verdict below,
# so it errs towards calling things code.
CODE = re.compile(r"\.(py|pyi|pyx|c|h|cpp|cu|cuh|js|mjs|ts|jsx|tsx|sh|toml|cfg)$", re.I)
DATA = re.compile(r"\.(json|ya?ml|csv|txt|lock)$", re.I)
# Things this project has already been bitten by, or is actively working on.
TENDER = re.compile(r"minimax|\bh3\b|sage|turbo|attention|frontend|web/|requirements", re.I)


def api(path: str) -> dict:
    """One GitHub API read. stdlib only - this runs on a bare runner."""
    req = urllib.request.Request(
        "https://api.github.com" + path,
        headers={"Accept": "application/vnd.github+json",
                 "User-Agent": "watch-h3-propose"})
    token = READ_TOKEN or os.environ.get("GH_TOKEN", "").strip()
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def analyse(repo: str, old_sha: str, new_sha: str) -> tuple[str, bool]:
    """What the drift actually contains, as markdown.

    "12 commits behind" is not reviewable. The same twelve commits are either
    a registry refresh you merge in ten seconds or a rewritten sampler you
    bench for an hour, and only the file list tells you which. Without this,
    every proposal costs a full build to evaluate - which is the surest way to
    make a bot's pull requests get ignored.

    Never fatal: a failed read degrades to the plain count. A proposal that
    arrives without its analysis is worth less, but a proposal that never
    arrives because an API call failed is worth nothing.

    The second return value says whether source changed. Unknown counts as
    yes: the checklist is allowed to be shortened by evidence, never by the
    absence of it.
    """
    try:
        cmp = api(f"/repos/{repo}/compare/{old_sha}...{new_sha}")
    except Exception as e:  # noqa: BLE001
        return (f"\n  <sub>diff unavailable ({type(e).__name__}); "
                f"review upstream by hand</sub>\n", True)

    commits = cmp.get("commits") or []
    files = cmp.get("files") or []
    code = [f for f in files if CODE.search(f["filename"])]
    data = [f for f in files if DATA.search(f["filename"]) and f not in code]

    if not files:
        verdict = "no file list returned by the API"
    elif not code:
        verdict = (f"**data only** — {len(data)} of {len(files)} files are "
                   f"registries or metadata, no source file changed. "
                   f"Behaviour cannot move; a green build is enough.")
    else:
        names = ", ".join(f"`{f['filename']}`" for f in code[:6])
        verdict = (f"**{len(code)} source file(s) changed** — {names}"
                   f"{' …' if len(code) > 6 else ''}. Bench before promoting.")

    # Commits that touch what this project is currently working on get pulled
    # out, because they are levers rather than housekeeping.
    flagged = [c["commit"]["message"].splitlines()[0]
               for c in commits if TENDER.search(c["commit"]["message"])]

    out = [f"\n  {verdict}"]
    if flagged:
        out.append("\n  Relevant to this project:")
        out += [f"\n  - ⚑ {s[:100]}" for s in flagged[:5]]
    out.append("\n\n  <details><summary>"
               f"{len(commits)} commits, {len(files)} files</summary>\n\n")
    for c in commits[-15:]:
        subject = c["commit"]["message"].splitlines()[0][:100]
        out.append(f"  - `{c['sha'][:8]}` {subject}\n")
    out.append("\n")
    for f in files[:20]:
        out.append(f"  - `{f['filename']}` "
                   f"+{f.get('additions', 0)}/-{f.get('deletions', 0)}\n")
    out.append("\n  </details>\n")
    return "".join(out), bool(code or not files)


def candidates(state: dict, only: list[str] | None = None) -> dict[str, dict]:
    """Pinned nodes worth a proposal.

    `only` names the repositories a person explicitly asked for, by ticking a
    box in the report. That selection overrides the drift threshold: the
    threshold exists to guess what deserves attention, and a human having
    already given it makes the guess irrelevant.
    """
    pins = state.get("pins/pod-comfyui-h3") or {}
    picked = {}
    for repo, v in pins.items():
        if not isinstance(v, dict) or not v.get("head"):
            continue
        if only:
            if repo in only:
                picked[repo] = v
        elif v.get("behind", 0) >= THRESHOLD:
            picked[repo] = v
    return picked


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
    ap.add_argument("--only", action="append", default=[], metavar="OWNER/REPO",
                    help="propose only these pins, whatever their drift "
                         "(repeatable). This is what a ticked box passes.")
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
    todo = candidates(state, args.only)
    if not todo:
        if args.only:
            # Asked for by name and not found: say which, because silence here
            # looks exactly like success.
            print(f"Nothing to propose for {', '.join(args.only)} — not in "
                  f"the state file, or no upstream head recorded for it.")
        else:
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
    risky = False
    for repo, info in sorted(todo.items()):
        changed = rewrite(dockerfile, repo, info["head"])
        if changed:
            old, new = changed
            detail, code_touched = analyse(repo, old, new)
            risky = risky or code_touched
            applied.append(f"- `{repo}` `{old[:12]}` → `{new[:12]}` "
                           f"({info['behind']} commits)" + detail)

    if not applied:
        print("Every pin already matches upstream in the checked-out branch.")
        return 0

    branch = "watch/bump-pinned-nodes"
    run("git", "checkout", "-B", branch, cwd=repo_dir)
    run("git", "add", "Dockerfile", cwd=repo_dir)
    # The checklist is only as long as the diff justifies. Asking for a bench
    # on a registry refresh is how a checklist stops being read at all.
    checks = ["- [ ] `cu130-develop` builds"]
    if risky:
        checks += [
            "- [ ] a test pod starts and ComfyUI loads every workflow "
            "(VideoHelperSuite breaks on frontend changes, and a `VHS_` node "
            "in a graph stops it loading entirely)",
            "- [ ] `scripts/bench.py --config baseline --config sage` shows "
            "no regression",
        ]
    else:
        checks.append("- [ ] a test pod starts and ComfyUI loads every "
                      "workflow — *no source file changed, so this is a "
                      "smoke test, not a bench*")

    body = (
        "Proposed by [watch-h3](https://github.com/unicorncomfyui/watch-h3). "
        "**Not verified — this only moves the pins.**\n\n"
        + "\n".join(applied)
        + "\n\n### Test it on a pod\n\n"
        "Pushing this branch builds the primary target. When that run is "
        "green, start a pod on:\n\n"
        "```\n" + IMAGE + ":cu130-proposal\n```\n\n"
        "That tag moves with each proposal; the immutable "
        "`cu130-<date>-<sha>` from the same run is there if you need to come "
        "back to this exact build.\n\n"
        "### Before merging\n\n"
        + "\n".join(checks)
        + "\n\nTargets `" + BASE + "` on purpose: the branch model exists so "
        "that an unverified change reaches a test pod and never `cu130` or "
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
