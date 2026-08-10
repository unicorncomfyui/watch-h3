# watch-h3

Watches the MiniMax H3 stack and reports only what moved.

The ecosystem changes faster than anyone can follow by hand. In a single week
SageAttention gained a kernel that cut generation time by 40%, a chaining pack
fixed the seam between clips, a sampler-level accelerator changed its defaults
after users reported audio damage, and the model team answered an AMA that
settled three open questions about the roadmap. None of that arrived through a
channel you can subscribe to.

This runs once a day, compares everything to yesterday, and opens an issue
when — and only when — something actually changed.

## What it watches

Declared in [`sources.json`](sources.json), with a note on each explaining why
it earns its place. Five kinds:

| type | what it captures |
|---|---|
| `hf_model` | a HuggingFace repo: revision, timestamp, **file list** |
| `github` | head commit, latest release, blob SHAs of chosen files |
| `pins` | how far each pinned custom node has drifted from upstream |
| `reddit` | threads matching an exact phrase, newest first |
| `page` | any URL with no API, reduced to a hash |

The file list is the point of `hf_model`. A new filename in the H3 repository
is how `Regenerate-2K`, an official low-step checkpoint, or a rewritten prompt
guide will first appear — usually days before anyone writes about it.

`pins` reads the SHAs straight from the `Dockerfile` that builds the pod image,
fetched raw over HTTP. Copying them here would create a second list, and the
drift between the two lists would be exactly the kind of thing this tool exists
to catch. It reports a distance, not a boolean: *3 commits behind* is a note,
*sixty commits and four months behind* is a decision to take before the next
ComfyUI bump.

## Running it

```bash
python watch.py                     # fetch, diff, update state.json
python watch.py --dry-run           # fetch and print, never write state
python watch.py --source gh/comfyui # one source, for debugging
```

Exit code carries the signal so the workflow branches without parsing
anything: `0` nothing moved, `10` something did.

**Set `GITHUB_TOKEN` when running locally.** Unauthenticated GitHub allows 60
requests an hour, which two test runs exhaust. Actions provides one
automatically, so this only bites on a laptop.

## Reading a thread

The watcher tells you a thread exists. This reads it, comments included:

```bash
python watch.py --thread https://reddit.com/r/StableDiffusion/comments/xxxxx/
```

Announcement posts carry the part that matters in the body and in the author's
replies — which ComfyUI internals get patched, what breaks on upgrade, which
settings were removed. None of that is in the title.

## Why Atom and not the JSON API

Reddit answers `search.json` with `403 Blocked` for unauthenticated clients,
from residential and datacenter addresses alike. `search.rss` still serves
anonymously, which buys the whole source without an OAuth application, a client
secret to rotate, and a token refresh to get wrong. The trade is that the feed
carries no score — and since only titles were ever diffed, nothing of value is
lost.

Requests to Reddit are spaced twelve seconds apart. That is not manners: four
searches fired back to back earn a `429`, and the retry then costs more than the
pause would have.

## State

`state.json` is committed. Its diff **is** the changelog — attributable,
reviewable, and still there in six months when you want to know when a pack
started drifting. A cache would give none of that and would expire on its own.

A source that fails keeps its **previous** value rather than recording the
error. Overwriting it would report the outage as a change, then report the
recovery as a second change: two false alarms per hiccup, and a watcher whose
alarms you learn to ignore is worse than no watcher.

## Secrets

None. Every endpoint is public. `GITHUB_TOKEN` is minted per run by Actions,
scoped to this repository, and dies with the job.
