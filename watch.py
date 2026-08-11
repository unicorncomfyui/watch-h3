#!/usr/bin/env python3
"""Watch a fast-moving stack and report only what actually moved.

The MiniMax H3 ecosystem changes faster than anyone can follow by hand:
weights, prompt guides, ComfyUI itself, a dozen custom-node packs, an
attention library that gained a 3x kernel between two Fridays, and a Reddit
thread that is often the only place a breaking change is announced.

Sources are declared in sources.json, not in this file. Adding one should be
an edit to data, not to code - that is the difference between a watcher that
grows with the stack and one that is abandoned after three months.

State is a JSON file committed to the repository. The diff of that file IS
the changelog: reviewable, attributable, no database, no credential. A
watcher that needs infrastructure is a watcher that stops working the month
you stop paying attention to it.

Every source is isolated. One rate-limited endpoint reports itself and the
run continues; a watcher that fails whole because Reddit throttled it teaches
you to ignore its failures, which is worse than having none.

    python watch.py                       # fetch, diff against state.json
    python watch.py --source hf/h3        # one source, for debugging
    python watch.py --dry-run             # fetch and print, never write state
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = "watch-h3/1.0 (upstream release watcher; contact via repo issues)"
TIMEOUT = 30


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

_last_hit: dict[str, float] = {}

# Reddit refuses anonymous clients hard from a datacenter address, and once it
# starts refusing it keeps refusing for a while. Two guards, because pacing
# alone was not enough:
#
#   BUDGET   an upper bound on requests per run. Sixteen were being fired -
#            four searches plus twelve thread bodies - and the run spent five
#            minutes mostly sleeping.
#   BLOCKED  a circuit breaker. When a request exhausts its retries, every
#            later Reddit call fails instantly instead of queueing behind
#            another two minutes of backoff. The log that prompted this had
#            fifteen [wait] lines AFTER the source had already given up.
REDDIT_BUDGET = int(os.environ.get("REDDIT_BUDGET", "10"))
# How many scored threads a subreddit may contribute to one report.
# Beyond this the section stops being read at all.
TITLES_KEPT = int(os.environ.get("TITLES_KEPT", "8"))
_reddit_spent = 0
_reddit_blocked = False


class RedditUnavailable(RuntimeError):
    """Reddit is refusing us for now; stop asking within this run."""


def throttle(url: str, seconds: float = 30.0) -> None:
    """Space out requests to the same host.

    Reddit's anonymous budget is small enough that four searches fired back to
    back earn a 429, and the retry then costs more time than the pause would
    have. Politeness here is not manners, it is the fastest path.
    """
    global _reddit_spent
    host = urllib.parse.urlparse(url).netloc
    if "reddit" not in host:
        return
    if _reddit_blocked:
        raise RedditUnavailable("circuit open after a refused request")
    if _reddit_spent >= REDDIT_BUDGET:
        raise RedditUnavailable(f"budget of {REDDIT_BUDGET} requests spent")
    _reddit_spent += 1
    wait = seconds - (time.monotonic() - _last_hit.get(host, 0))
    if wait > 0:
        time.sleep(wait)
    _last_hit[host] = time.monotonic()


def fetch(url: str, token: str | None = None, raw: bool = False):
    """One request, with the two headers that decide whether it works.

    The User-Agent is not politeness: Reddit answers a default urllib UA with
    429 more or less always. The bearer token only ever goes to GitHub - a
    token pasted into an arbitrary host is how credentials leak.
    """
    throttle(url)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    if not raw:
        req.add_header("Accept", "application/json")
    if token and url.startswith("https://api.github.com/"):
        req.add_header("Authorization", f"Bearer {token}")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                body = r.read().decode("utf-8", "replace")
                return body if raw else json.loads(body)
        except urllib.error.HTTPError as e:
            # 429 is a rate limit and resets; retrying costs seconds and often
            # works. 403 does not: from GitHub it means the hourly budget is
            # spent, from Reddit it means the endpoint refuses anonymous
            # clients outright. Retrying either just makes the run slower and
            # the log longer. 404 is an answer, not a failure.
            if e.code == 403 and "api.github.com" in url and not token:
                raise RuntimeError(
                    "GitHub rate limit: 60 requests/hour without a token. "
                    "Export GITHUB_TOKEN to raise it to 5000 - Actions "
                    "provides one automatically, so this only bites locally."
                ) from e
            if e.code == 429 and attempt < 1:
                # One retry, not two. A second one doubled the wall clock and
                # never once succeeded in the runs that produced this comment;
                # Reddit's refusal outlasts any backoff worth sitting through.
                wait = int(e.headers.get("Retry-After") or 0) or 45
                print(f"[wait] 429, one retry in {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            if e.code in (429, 403) and "reddit" in url:
                global _reddit_blocked
                _reddit_blocked = True
                print("[stop] Reddit refused; skipping it for the rest of "
                      "this run", file=sys.stderr)
                raise RedditUnavailable(f"HTTP {e.code}") from e
            raise


# --------------------------------------------------------------------------
# Fetchers. Each returns a flat-ish dict; the differ walks it.
# --------------------------------------------------------------------------

def hf_model(spec: dict, token: str | None,
             prev: dict | None = None, asof: str | None = None) -> dict:
    """A HuggingFace repo: revision, timestamp, and the file list.

    The file list is the point. A new entry is how Regenerate-2K, an official
    low-step checkpoint or a rewritten prompt guide will first appear -
    usually days before anyone writes about it.
    """
    url = f"https://huggingface.co/api/models/{spec['repo']}"
    if asof:
        # Rewind by asking for the revision that was current then. If the repo
        # has not moved since, the answer is identical to now - which is the
        # correct baseline, not a missing one.
        hist = fetch(f"{url}/commits/main?limit=50")
        older = [c for c in hist if c.get("date", "") <= asof]
        if older:
            url += f"?revision={older[0]['id']}"
    d = fetch(url)
    files = sorted(f["rfilename"] for f in d.get("siblings", []))
    keep = spec.get("only")
    if keep:
        files = [f for f in files if any(k in f for k in keep)]
    return {"sha": (d.get("sha") or "")[:12],
            "modified": d.get("lastModified", ""),
            "files": files}


def github(spec: dict, token: str | None,
           prev: dict | None = None, asof: str | None = None) -> dict:
    """A GitHub repo: head of default branch, latest release, chosen blobs.

    Per-file blob SHAs matter more than the head commit for documentation:
    they say whether the prompt Skill changed, not merely whether somebody
    touched the repository.
    """
    repo = spec["repo"]
    out: dict = {}
    until = f"&until={asof}" if asof else ""
    head = fetch(f"https://api.github.com/repos/{repo}/commits?per_page=1{until}",
                 token)
    ref = ""
    if head:
        ref = head[0]["sha"]
        out["head"] = ref[:12]
        out["head_date"] = head[0]["commit"]["committer"]["date"]
        out["head_msg"] = head[0]["commit"]["message"].splitlines()[0][:100]
    try:
        if asof:
            # /releases/latest has no time filter, so take the newest release
            # that had already been published then.
            rels = fetch(f"https://api.github.com/repos/{repo}/releases"
                         f"?per_page=30", token)
            past = [r for r in rels if (r.get("published_at") or "") <= asof]
            rel = past[0] if past else None
            if rel is None:
                raise urllib.error.HTTPError("", 404, "none yet", {}, None)
        else:
            rel = fetch(f"https://api.github.com/repos/{repo}/releases/latest",
                        token)
        out["release"] = rel.get("tag_name")
        out["release_date"] = rel.get("published_at")
        out["release_name"] = (rel.get("name") or "")[:100]
    except urllib.error.HTTPError as e:
        if e.code != 404:                    # 404 = this repo cuts no releases
            raise
    for p in spec.get("files", []):
        try:
            # Pinned to the commit chosen above, so a file's SHA belongs to
            # the same instant as the head it is reported next to.
            at = f"?ref={ref}" if ref else ""
            f = fetch(f"https://api.github.com/repos/{repo}/contents/"
                      f"{urllib.parse.quote(p)}{at}", token)
            out[f"file:{p}"] = (f.get("sha") or "")[:12]
        except urllib.error.HTTPError:
            out[f"file:{p}"] = "absent"
    return out


def pins(spec: dict, token: str | None, prev: dict | None = None,
         asof: str | None = None) -> dict:
    """How far a pinned custom node sits behind its upstream head.

    The pins are read from the Dockerfile that actually builds the image,
    fetched raw over HTTP. Copying them into this repo would create a second
    list, and the drift between the two lists would be invisible - which is
    precisely the failure this tool exists to prevent.

    Reported as a count, not a boolean. "3 commits behind" is a note; "sixty
    commits and four months behind" is a decision to take before the next
    ComfyUI bump.
    """
    src = fetch(spec["dockerfile_url"], raw=True)
    pairs = re.findall(
        r"github\.com/([\w.-]+/[\w.-]+)\.git\s*\\\s*"
        r"&& git -C [\w.-]+ checkout -q ([0-9a-f]{40})", src)
    prev = prev or {}
    out: dict = {}
    failures = 0
    for repo, sha in pairs:
        repo = repo.removesuffix(".git")
        # The repository name comes from a file fetched over the network and
        # is then pasted into a URL path. `[\w.-]+` happily matches `..`, so
        # a name of `../..` would walk out of /repos/ and the server would
        # normalise it away - a request to a path we never intended. It takes
        # control of that Dockerfile to exploit, which is a high bar, but a
        # value from the network belongs in a URL only after it has been
        # checked, not because reaching it looked difficult.
        if ".." in repo or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
            print(f"[skip] refusing suspicious repo name {repo!r}",
                  file=sys.stderr)
            continue
        try:
            target = "HEAD"
            if asof:
                # Compare against where the node stood then, so the drift we
                # report is the drift that existed at that moment.
                h = fetch(f"https://api.github.com/repos/{repo}/commits"
                          f"?per_page=1&until={asof}", token)
                target = h[0]["sha"] if h else "HEAD"
            cmp = fetch(f"https://api.github.com/repos/{repo}/compare/"
                        f"{sha}...{target}", token)
            commits = cmp.get("commits") or []
            entry = {"pinned": sha[:12], "behind": cmp.get("ahead_by", 0)}
            if commits:
                # The full upstream SHA, because a bump has to write forty
                # characters into the Dockerfile. Storing the short form
                # would make the proposal step fetch it all over again.
                entry["head"] = commits[-1]["sha"]
                # The newest subject line, free of charge - the compare
                # response already carries it. A count says how far we have
                # drifted; this says whether the drift is a README tweak or a
                # rewritten sampler, which is the part that decides anything.
                entry["latest"] = commits[-1]["commit"]["message"] \
                    .splitlines()[0][:100]
                # The same response already carries the file list, so the
                # triage below is free. Asking "is this worth a rebuild"
                # from a commit count alone is how a registry refresh and a
                # rewritten sampler end up looking identical.
                subjects = [c["commit"]["message"].splitlines()[0]
                            for c in commits]
                code = [f["filename"] for f in (cmp.get("files") or [])
                        if CODE_FILE.search(f["filename"])]
                entry["code"] = len(code)
                entry["kind"], entry["why"] = classify(subjects, len(code))
            out[repo] = entry
        except Exception:  # noqa: BLE001 - one node must not sink the run
            # Carry the last known answer rather than recording the failure.
            # Storing an error here would report the outage as a change, then
            # report the recovery as a second one - and the first version of
            # this function did exactly that, silently, while announcing
            # itself as a successful source.
            failures += 1
            if repo in prev:
                out[repo] = prev[repo]
    if not pairs:
        raise RuntimeError("no pins matched - the Dockerfile layout changed")
    if failures == len(pairs):
        raise RuntimeError(f"every pin lookup failed ({failures})")
    return out


def reddit(spec: dict, token: str | None,
           prev: dict | None = None, asof: str | None = None) -> dict:
    """Threads matching a query, newest first, over the Atom feed.

    Not the .json endpoint: Reddit now answers it with 403 Blocked for
    unauthenticated clients, from residential and datacenter addresses alike.
    The .rss search endpoint still serves anonymously, which buys us the whole
    source without an OAuth application, a client secret to rotate, and a
    token refresh to get wrong.

    The trade is that the feed carries no score. We only ever diffed titles -
    a score moves on every run, and a watcher that reports something every run
    is a watcher nobody reads - so nothing of value is lost.
    """
    import xml.etree.ElementTree as ET

    sub = spec.get("subreddit")
    base = (f"https://www.reddit.com/r/{sub}/search.rss?restrict_sr=1&"
            if sub else "https://www.reddit.com/search.rss?")
    url = base + urllib.parse.urlencode({
        "q": spec["query"], "sort": spec.get("sort", "new"),
        "t": spec.get("window", "week"), "limit": spec.get("limit", 25)})
    body = fetch(url, raw=True)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    posts = []
    for e in ET.fromstring(body).findall("a:entry", ns):
        link = e.find("a:link", ns)
        author = e.find("a:author/a:name", ns)
        posts.append({
            "id": (e.findtext("a:id", "", ns) or "").rsplit("/", 1)[-1],
            "title": (e.findtext("a:title", "", ns) or "")[:160],
            "author": author.text if author is not None else "",
            "url": link.get("href") if link is not None else "",
            "updated": e.findtext("a:updated", "", ns) or "",
        })
    if asof:
        # Keep only what already existed then; the diff is therefore exactly
        # the threads opened since.
        posts = [p for p in posts if p.get("updated", "") <= asof]
    return {"posts": posts}


def page(spec: dict, token: str | None,
         prev: dict | None = None, asof: str | None = None) -> dict:
    """Any URL, reduced to a length and a hash.

    For pages with no API - a docs page, a model card rendered as HTML. It
    cannot say WHAT changed, only that something did, which is enough to send
    a human to look.
    """
    import hashlib
    body = fetch(spec["url"], raw=True)
    if spec.get("strip_dynamic", True):
        # Timestamps, CSRF tokens and build ids change on every fetch and
        # would make this source cry wolf daily.
        body = re.sub(r'(nonce|csrf|build|timestamp|_t)="[^"]*"', "", body, flags=re.I)
        body = re.sub(r"\b\d{10,13}\b", "", body)
    return {"bytes": len(body),
            "sha256": hashlib.sha256(body.encode()).hexdigest()[:16]}


def civitai(spec: dict, token: str | None, prev: dict | None = None,
            asof: str | None = None) -> dict:
    """A workflow published on Civitai, tracked by its versions.

    Workflows are where the community's findings become usable before anyone
    writes them down: a version bump usually IS the changelog. The page itself
    is client-rendered and its hash moves on view counts and ratings, so the
    JSON API is the only stable surface - and it needs no key for a public
    model.

    Only the version list and its dates are kept. Download counts and ratings
    change hourly and would report a change every single run, which is how a
    source becomes noise you stop reading.
    """
    d = fetch(f"https://civitai.com/api/v1/models/{spec['model_id']}")
    versions = d.get("modelVersions") or []
    out = {
        "name": d.get("name"),
        "creator": (d.get("creator") or {}).get("username"),
        "versions": [v.get("name") for v in versions][:12],
    }
    if versions:
        out["latest"] = versions[0].get("name")
        out["published"] = (versions[0].get("publishedAt") or "")[:10]
    return out


FETCHERS = {"hf_model": hf_model, "github": github, "pins": pins,
            "reddit": reddit, "page": page, "civitai": civitai}


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------

# What separates a documented benchmark from "look at my video". Weights are
# blunt on purpose: this decides reading ORDER, not truth, and a scorer with
# thirty tuned coefficients would be a second thing to maintain.
#
# The signals are all marks of someone showing their work - measurements with
# units, exact versions, a seed, a link to the artefacts. A post that carries
# four of them is worth more than a hundred that carry none, and that ratio is
# roughly what the subreddit actually contains.
SIGNALS: list[tuple[str, str, int]] = [
    (r"\b\d+(?:[.,]\d+)?\s*(?:s|sec|seconds|ms)\b", "timings", 3),
    (r"\b\d+(?:[.,]\d+)?\s*%", "percentages", 3),
    (r"\b\d+\s*(?:GB|GiB|VRAM)\b", "memory figures", 2),
    (r"\b(?:seed|même graine|same seed)\b", "a fixed seed", 3),
    (r"\b\d+\s*steps?\b", "step counts", 2),
    (r"\bgithub\.com/[\w.-]+/[\w.-]+", "a repository link", 2),
    (r"\bhuggingface\.co/[\w.-]+", "a HuggingFace link", 2),
    (r"\b(?:benchmark|comparison|A/B|same prompt)\b", "a comparison", 3),
    (r"\b(?:workflow|\.json|API graph)\b", "a workflow", 2),
    (r"\b(?:torch|cuda|sm_?\d{2,3}|v\d+\.\d+\.\d+)\b", "exact versions", 2),
    (r"\|.+\|.+\|", "a table", 2),
]

# Words that mark a post as a result rather than a method. Not a penalty for
# being pretty - just a demotion below anything that measured something.
NOISE = re.compile(r"\b(?:my first|just joining|thank you|incredible|amazing|"
                   r"insane|check out|lol)\b", re.I)


def score_body(text: str) -> tuple[int, list[str]]:
    """Rank a post by how much of its method it shows."""
    found, total = [], 0
    for pattern, label, weight in SIGNALS:
        hits = len(re.findall(pattern, text, re.I))
        if hits:
            # Sub-linear: ten timings are better than one, not ten times
            # better, and without this a post listing every frame time would
            # outrank a genuine comparison.
            total += weight * min(hits, 3)
            found.append(label)
    if NOISE.search(text[:400]):
        total -= 4
    if len(text) > 1500:
        total += 2                      # someone wrote at length about method
    return total, found


def read_thread(url: str) -> str:
    """One Reddit thread - body and comments - rendered as markdown.

    The watcher reports that a thread exists; this reads it. Announcement
    posts for node packs carry the part that matters in the body and in the
    author's replies: which ComfyUI internals get patched, what breaks on
    upgrade, which settings were removed. None of that is in the title.

    Atom again, for the same reason as the search: .json answers 403 to
    anonymous clients. The feed carries the post and its comments as escaped
    HTML, which is unescaped and stripped of tags here - crude, but these are
    plain-text posts and a parser dependency would not earn its place.
    """
    import html
    import xml.etree.ElementTree as ET

    url = url.split("?")[0].rstrip("/")
    body = fetch(f"{url}/.rss?limit=100", raw=True)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out: list[str] = []
    for i, e in enumerate(ET.fromstring(body).findall("a:entry", ns)):
        author = e.find("a:author/a:name", ns)
        who = author.text if author is not None else "?"
        content = html.unescape(e.findtext("a:content", "", ns) or "")
        content = re.sub(r"<[^>]+>", "", content)
        content = html.unescape(content)
        content = re.sub(r"\n{3,}", "\n\n", content).strip()
        head = e.findtext("a:title", "", ns) if i == 0 else f"comment — {who}"
        out.append(f"### {head}\n\n{content}\n")
    return "\n".join(out) if out else "(empty feed)"


# --------------------------------------------------------------------------
# Diff
# --------------------------------------------------------------------------

def diff_reddit(old: dict, new: dict, deep: int = 0) -> list[str]:
    """New threads, best first, with the good ones opened and weighed.

    Never removals: a thread leaving a one-week window is the window sliding,
    not an event, and reporting it would bury what matters under a daily list
    of things that merely aged.

    `deep` caps how many bodies are read. Each costs a throttled request, and
    the point is not to mirror the subreddit - it is to find the two posts a
    week that carry a measurement. Titles alone cannot tell those apart:
    "MiniMax H3 RTX PRO 6000 follow-up" and "just joining in the fun, minimax
    H3" look equally promising until you open them.
    """
    seen = {p["id"] for p in (old or {}).get("posts", [])}
    fresh = [p for p in new.get("posts", []) if p["id"] not in seen]
    if not fresh:
        return []

    ranked, read = [], 0
    for p in fresh:
        # Bodies are read while Reddit allows it, then the rest fall back to
        # their titles. Degrading the RANKING is a small loss; losing the
        # whole source to a refused body read is not.
        if read < deep and not _reddit_blocked:
            try:
                n, why = score_body(read_thread(p["url"]))
                read += 1
                ranked.append((n, why, p))
                continue
            except RedditUnavailable:
                pass
            except Exception:  # noqa: BLE001 - an unread body is a lost rank
                pass
        n, why = score_body(p["title"])
        ranked.append((n, why, p))

    # A busy day on a busy subreddit produced twenty-five links, of which
    # twenty carried no measurable signal at all. Listing them in full buries
    # the two that do and trains the reader to skip the section. The scored
    # ones are always shown; the rest are counted, not enumerated.
    ranked.sort(key=lambda r: -r[0])
    keep = [r for r in ranked if r[0] > 0][:TITLES_KEPT]
    lines = []
    for n, why, p in keep:
        mark = "**" if n >= 12 else ""
        lines.append(f"  - {mark}[{p['title']}]({p['url']}){mark}"
                     + (f" — *{n} pts: {', '.join(why[:4])}*" if why else "")
                     + f" — {p.get('author', '?')}")
    rest = len(ranked) - len(keep)
    if rest:
        lines.append(f"  - <sub>{rest} more with no measurable signal, "
                     f"not listed</sub>")
    return lines


def diff(old, new, path: str = "") -> list[str]:
    """Leaf-level changes in words. Lists report additions and removals
    separately, because "one file appeared" and "one file vanished" are not
    the same event and must not cancel out."""
    lines: list[str] = []
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            lines += diff(old.get(key), new.get(key),
                          f"{path}.{key}" if path else key)
    elif isinstance(old, list) and isinstance(new, list):
        so, sn = {json.dumps(x, sort_keys=True) for x in old}, \
                 {json.dumps(x, sort_keys=True) for x in new}
        for x in sorted(sn - so):
            lines.append(f"  - `{path}` **+** `{x[:150]}`")
        for x in sorted(so - sn):
            lines.append(f"  - `{path}` **−** `{x[:150]}`")
    elif old != new:
        if old is None:
            lines.append(f"  - `{path}` appeared: `{new}`")
        elif new is None:
            lines.append(f"  - `{path}` disappeared (was `{old}`)")
        else:
            lines.append(f"  - `{path}`: `{old}` → `{new}`")
    return lines


# --------------------------------------------------------------------------
# Levers
# --------------------------------------------------------------------------

POD = "unicorncomfyui/pod-comfyui-h3"
# Never `main`. The branch model is the safety here: a proposal reaches a test
# pod as `cu130-develop`, and `main` stays what production pulls.
POD_BASE = os.environ.get("POD_BASE_BRANCH", "develop")

# When a pinned node is worth proposing. See the drift loop below for why
# there are two numbers rather than one.
BUMP_AT = int(os.environ.get("BUMP_THRESHOLD", "10"))
BUMP_AT_RELEVANT = int(os.environ.get("BUMP_THRESHOLD_RELEVANT", "3"))
# One commit is enough when it closes a hole in code this image ships, whatever
# else is or is not in the drift.
BUMP_AT_SECURITY = 1
SECURITY = re.compile(r"arbitrary code|code execution|\bRCE\b|CVE-\d|"
                      r"pickle\.load|path traversal|sanitiz|unsafe (?:eval|load)",
                      re.I)
RELEVANT = re.compile(r"minimax|\bh3\b|sage|turbo|attention|sampler|vae",
                      re.I)

# Whether a changed file can move what the image DOES. A registry of node
# metadata cannot; a sampler can.
CODE_FILE = re.compile(
    r"\.(py|pyi|pyx|c|h|cpp|cu|cuh|js|mjs|ts|jsx|tsx|sh|toml|cfg)$", re.I)

# What a drift is FOR. The question in front of every pin is the same one -
# is this worth a rebuild - and it splits three ways: something got faster,
# something was broken, or something upstream now requires it. Anything else
# can wait for the next bump that isn't optional.
KINDS: list[tuple[str, re.Pattern]] = [
    ("perf", re.compile(r"optimi[sz]|faster|speed|perf\b|memory|vram|"
                        r"attention|quant|fp8|int8|cache|throughput", re.I)),
    ("fix", re.compile(r"\bfix|crash|error|regress|broken|hotfix|revert|"
                       r"leak|oom\b", re.I)),
    ("compat", re.compile(r"compat|support for|requires?|torch|python 3|"
                          r"comfyui v?\d|frontend|breaking|deprecat|"
                          r"sm_?\d{2,3}|blackwell|cuda", re.I)),
    ("feature", re.compile(r"\badd\b|\bfeat\b|new node|implement", re.I)),
]


def classify(subjects: list[str], code_files: int) -> tuple[str, list[str]]:
    """What this drift contains, and the commit lines that say so.

    Deliberately blunt: it reads commit subjects, so it inherits whatever
    discipline the upstream author had. It is a triage hint that saves opening
    twelve commits, never a verdict - which is why the evidence travels with
    the label instead of being summarised away.
    """
    if not code_files:
        return "registry/docs", []
    labels, why = [], []
    for name, rx in KINDS:
        matched = [s for s in subjects if rx.search(s)]
        if matched:
            labels.append(name)
            why += matched[:2]
    return ("+".join(labels) if labels else "chore"), why[:3]



def hygiene(new: dict) -> str:
    """Every pinned node that has drifted at all, worth-it verdict included.

    The levers list only what crosses a threshold. This is the standing view:
    seven rows, read in ten seconds, that answer "is any of my pins carrying a
    fix I don't have" without opening a single upstream repository. A pin at
    zero is omitted - a table of "nothing to do" teaches you to skip the
    table.
    """
    pins = new.get("pins/pod-comfyui-h3") or {}
    rows = [(r, v) for r, v in sorted(pins.items())
            if isinstance(v, dict) and v.get("behind", 0) > 0]
    if not rows:
        return ""
    out = ["## Pins\n",
           "| node | behind | contains | latest |",
           "|---|---:|---|---|"]
    for repo, v in sorted(rows, key=lambda kv: -kv[1].get("behind", 0)):
        kind = v.get("kind", "?")
        if v.get("code"):
            kind = f"**{kind}** ({v['code']} src)"
        latest = (v.get("latest") or "?").replace("|", "\\|")[:70]
        out.append(f"| `{repo}` | {v['behind']} | {kind} | {latest} |")
    return "\n".join(out) + "\n"


def levers(old: dict, new: dict) -> list[str]:
    """Turn what moved into what to do about it.

    A watcher that only reports leaves the translation to a human every
    morning, and that translation is the part that gets skipped. These are
    checkboxes, phrased as the next concrete step, with the file or the button
    named - not "SageAttention moved" but "rebuild the wheel, here".

    Nothing here decides anything. Every lever ends at a bench on the actual
    hardware, because every measurement this project has made contradicted at
    least one confident expectation.
    """
    out: list[str] = []

    def moved(name, key=None):
        o, n = old.get(name) or {}, new.get(name) or {}
        if not o or not n:
            return None                     # no baseline: not a movement
        return (o.get(key), n.get(key)) if key and o.get(key) != n.get(key) \
            else (None if key else (o != n))

    # Pinned nodes drifting. Ten commits is a change of behaviour worth a
    # rebuild - but volume is a poor proxy for relevance, and this project has
    # already seen it fail both ways: twelve ComfyUI-Manager commits that were
    # pure registry JSON, against five KJNodes commits containing an H3
    # memory-efficient Sage fix. So a pin whose newest commit names something
    # we actually depend on surfaces four times sooner.
    #
    # Listing costs nothing now: the line is a proposal, and nothing is opened
    # until a person ticks it.
    o_pins = (old.get("pins/pod-comfyui-h3") or {})
    for repo, cur in (new.get("pins/pod-comfyui-h3") or {}).items():
        if not isinstance(cur, dict) or "behind" not in cur:
            continue
        latest = cur.get("latest") or ""
        # Scan every subject, not only the newest. KJNodes carried a fix for
        # arbitrary code execution while its head commit was an anodyne merge,
        # so reading `latest` alone left a shipped vulnerability below the
        # threshold and out of the report entirely.
        subjects = latest + " " + " ".join(cur.get("why") or [])
        if SECURITY.search(subjects):
            limit = BUMP_AT_SECURITY
        elif RELEVANT.search(subjects):
            limit = BUMP_AT_RELEVANT
        else:
            limit = BUMP_AT
        if cur["behind"] < limit:
            continue
        was = (o_pins.get(repo) or {}).get("behind")
        moved_by = f" (was {was})" if was is not None and was != cur["behind"] \
            else ""
        kind = cur.get("kind", "?")
        code = cur.get("code")
        weight = "no source file changed" if code == 0 else \
            f"{code} source file(s)" if code else "contents unread"
        out.append(
            f"- [ ] **{repo}** — {cur['behind']} commits behind{moved_by}, "
            f"**{kind}**, {weight}. Tick to open the bump pull request on "
            f"`{POD_BASE}`. <!--bump:{repo}-->")
        for line in (cur.get("why") or [])[:2]:
            out.append(f"      - {line[:110]}")
        if not cur.get("why"):
            out.append(f"      - latest: *{latest or '?'}*")

    if moved("gh/sageattention", "head"):
        out.append(
            "- [ ] **SageAttention moved.** Our wheel is compiled from that "
            "branch, so it is now stale. Run *Build SageAttention wheel* on "
            f"[{POD}](https://github.com/{POD}/actions), paste the new URL "
            "into the `sa_wheel` matrix entry.")

    rel = moved("gh/comfyui", "release")
    if rel:
        out.append(
            f"- [ ] **ComfyUI {rel[1]}** released (was {rel[0]}). Re-test the "
            "pinned node set before bumping `COMFYUI_VERSION` - "
            "VideoHelperSuite already breaks on a frontend change.")

    for src, what in (("gh/h3-turbo", "the step-distillation LoRA"),
                      ("gh/h3-spectrum", "the Spectrum sampler accelerator"),
                      ("gh/h3-motion-context", "the clip chaining pack")):
        r = moved(src, "release")
        if r:
            out.append(f"- [ ] **{src.split('/')[-1]} {r[1]}** released — "
                       f"{what}. Read the notes before updating: these packs "
                       f"patch ComfyUI internals and only one can own them.")

    # New weights. The file list is the earliest possible signal.
    o_files = set((old.get("hf/minimax-h3") or {}).get("files", []))
    n_files = set((new.get("hf/minimax-h3") or {}).get("files", []))
    for f in sorted(n_files - o_files):
        if o_files:
            hot = re.search(r"turbo|lora|2k|regenerate|distill|step", f, re.I)
            out.append(f"- [{'x' if hot else ' '}] {'**' if hot else ''}New "
                       f"file in the H3 repo: `{f}`{'**' if hot else ''}"
                       + (" — candidate for `models/manifest.json`." if hot
                          else ""))

    ver = moved("civitai/h3-filmmaking", "latest")
    if ver:
        out.append(
            f"- [ ] **The all-in-one workflow moved to {ver[1]}** (was "
            f"{ver[0]}). Community workflows are where a finding becomes "
            f"usable before it is written down - diff the graph against "
            f"`workflows/gui/` and take what applies.")

    if moved("doc/h3-prompt-guide-ref"):
        out.append(
            "- [ ] **The reference-mode prompt schema changed.** It defines "
            "the six sections a ref2v prompt must carry, so an edit here "
            "changes what a working prompt looks like - and `lambda_enrich` "
            "generates against it.")

    if moved("doc/comfyui-h3"):
        out.append(
            "- [ ] **The ComfyUI H3 tutorial changed.** It carries the Sage "
            "Attention instructions and the required model list; diff it "
            "against what the image installs.")

    return out


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sources", default=str(HERE / "sources.json"))
    ap.add_argument("--state", default=str(HERE / "state.json"))
    ap.add_argument("--report", help="write the markdown summary here")
    ap.add_argument("--source", help="run only this source, by name")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch and print, never write state")
    ap.add_argument("--thread", metavar="URL",
                    help="print one Reddit thread, comments included, and exit")
    ap.add_argument("--as-of", type=float, metavar="HOURS",
                    help="build the baseline as it stood HOURS ago and report "
                         "what has changed since. Makes a first run useful "
                         "instead of silent, and is the only way to test the "
                         "reporting path against real movement rather than a "
                         "hand-edited state file.")
    args = ap.parse_args()

    # Upstream text is full of emoji and typographic quotes, and a Windows
    # console defaults to cp1252, which raises on the first one. Failing to
    # PRINT a report we successfully fetched would be an absurd way to lose a
    # run, so force the stream and let unmappable characters degrade.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    if args.thread:
        print(read_thread(args.thread))
        return 0

    token = os.environ.get("GITHUB_TOKEN", "").strip() or None
    specs = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    if args.source:
        specs = {k: v for k, v in specs.items() if k == args.source}
        if not specs:
            return print(f"no source named {args.source!r}") or 1

    state_path = Path(args.state)
    old = json.loads(state_path.read_text(encoding="utf-8")) \
        if state_path.is_file() else {}

    # --as-of reconstructs the baseline instead of reading it. Sources that
    # cannot be rewound - a rendered page has no history - return their current
    # value, so they report no change rather than a false one.
    asof = None
    no_baseline: set[str] = set()
    if args.as_of:
        from datetime import datetime, timedelta, timezone
        asof = (datetime.now(timezone.utc)
                - timedelta(hours=args.as_of)).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"Baseline rebuilt as of {asof} ({args.as_of:g} h ago)",
              file=sys.stderr)
        old = {}
        for name, spec in specs.items():
            fn = FETCHERS.get(spec.get("type", ""))
            if fn is None:
                continue
            try:
                old[name] = fn(spec, token, None, asof)
                print(f"[was]  {name}", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                # No baseline means nothing to compare against, which is not
                # the same as "everything is new". Leaving the source out of
                # `old` would make the differ announce it as having appeared -
                # which is exactly what the first version did, once per source
                # the rate limit had blocked.
                no_baseline.add(name)
                print(f"[was]  {name}: {type(e).__name__}: {e}", file=sys.stderr)
        asof = None                   # the second pass must fetch the present

    new: dict = {}
    failed: list[str] = []
    for name, spec in specs.items():
        fn = FETCHERS.get(spec.get("type", ""))
        if fn is None:
            new[name] = {"_error": f"unknown type {spec.get('type')!r}"}
            failed.append(name)
            continue
        try:
            new[name] = fn(spec, token, old.get(name), asof)
            print(f"[ok]   {name}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            # Keep the PREVIOUS value on failure. Overwriting it with an error
            # would report the outage as a change, and then report the
            # recovery as a second change - two false alarms per hiccup.
            new[name] = old.get(name, {})
            failed.append(f"{name} ({type(e).__name__}: {e})")
            print(f"[fail] {name}: {type(e).__name__}: {e}", file=sys.stderr)

    changes: list[str] = []
    for name in specs:
        if name in no_baseline:
            continue
        # A source added since the last run has nothing to compare against.
        # Diffing it against nothing dumped its whole state as "appeared"
        # lines, burying the run that introduced it under the one thing in it
        # that was not news.
        if name not in old and name in new:
            changes.append(f"### {name}\n"
                           f"  - first observation, nothing to compare yet")
            continue
        d = (diff_reddit(old.get(name, {}), new.get(name, {}),
                         specs[name].get("deep", 0))
             if specs[name]["type"] == "reddit"
             else diff(old.get(name), new.get(name)))
        if d:
            changes.append(f"### {name}\n" + "\n".join(d))

    if args.dry_run:
        print(json.dumps(new, indent=2, ensure_ascii=False))
        return 0

    state_path.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")

    if not old:
        print("First run: baseline recorded, nothing to compare against.")
        return 0

    # The action list comes first, because it is the only part that asks
    # something of a person. A report that opens with forty lines of SHAs
    # buries the one line that matters under the evidence for it.
    todo = levers(old, new)
    report = ("## Do this\n\n" + "\n".join(todo) + "\n\n") if todo else ""
    # Between the decisions and the evidence: the standing state of the pins.
    # It is the part that gets read on a day when nothing needs doing.
    table = hygiene(new)
    if table:
        report += table + "\n"
    report += ("## What moved\n\n" + "\n\n".join(changes) + "\n"
               if changes else "Nothing moved.\n")
    if failed:
        report += "\n<sub>unreachable this run: " + ", ".join(failed) + "</sub>\n"
    # Only the mechanical levers carry a marker. Saying so plainly matters:
    # an unticked box that silently does nothing and a ticked box that opens a
    # pull request must not look alike.
    if "<!--bump:" in report:
        report += (
            "\n---\n\n**Ticking a marked box opens the pull request** on "
            f"`{POD_BASE}` of [{POD}](https://github.com/{POD}) — editing this "
            "issue is the trigger. Nothing is opened until you do, and nothing "
            "is ever merged. The other boxes are notes to yourself.\n")
    print(report)
    if args.report:
        Path(args.report).write_text(report, encoding="utf-8")
    # The exit code carries the signal so the workflow branches without
    # parsing anything: 0 nothing, 10 something moved.
    return 10 if changes else 0


if __name__ == "__main__":
    sys.exit(main())
