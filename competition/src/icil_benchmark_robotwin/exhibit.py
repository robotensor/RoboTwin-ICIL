"""Turn a finished run directory into the document the competition publishes as a reference.

A reference measurement is a run that is **no field's score** - this benchmark's own oracle, a
released checkpoint measured here, a run to see whether a field is reachable at all. The
competition signs one onto no ladder (`references/<id>.json`); what it needs from us is the
document, and only this repository knows how to read a run directory.

Two rules shape this module:

- **Nothing here imports `icilval`.** It emits a plain dict. The competition validates, signs and
  publishes it.
- **Every number is derived from the records; every sentence is written by a person.** A run
  cannot tell you what the policy was *shown* - that is a property of the adapter, and it is the
  one claim a reference exists to make - so the prose comes from a file and is required, not
  defaulted. Deriving it would be guessing about the only thing that matters.

The document is rebuilt, never edited, so it stays true as episodes land.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from robotwin_icil import report as report_
from robotwin_icil.records import RunDir

#: Sentences a person must write. Each is a claim the records cannot support on their own.
REQUIRED_PROSE = (
    "reference_id",
    "headline",
    "not_a_competition_score",
    "demonstration_shown",
)


class ExhibitError(ValueError):
    """The runs and the prose do not add up to a document worth publishing."""


def _scored(run: RunDir) -> list[dict[str, Any]]:
    """Episodes that produced a score. A rejected or invalid episode is not a result."""
    return [r for r in run.records() if str(getattr(r.status, "value", r.status)) == "scored"]


def _row(record: Any, prefix: str) -> dict[str, Any]:
    return {
        "id": f"{prefix}-{record.episode}",
        "task": record.task,
        # The benchmark's own category, named so it cannot be mistaken for a competition skill
        # id: RoboTwin's `pick_and_place` and the sensorimotor field's are different things with
        # the same string, and this document is read beside the latter.
        "benchmark_category": record.skill_category,
        "scored": 1,
        "successes": 1 if record.success else 0,
        "steps": record.steps,
        # Per task, from RoboTwin's `_eval_step_limit.yml` - 400 here, but 500 for
        # place_empty_cup and 900 for stack_bowls_two. Beside `steps` it is what makes "400" read
        # as "hit the cap" rather than "stopped". It is emphatically not a run-wide constant, and
        # putting it in `protocol` made the document false as soon as a 500-step task landed.
        "step_limit": record.step_limit,
        "scene_seed": record.scene_seed,
    }


def _clips(run_dir: Path, record: Any, doc_dir: Path) -> list[dict[str, Any]]:
    """The clips of one scored episode, as sources relative to the document.

    Only an episode in the records gets clips. A directory left behind by an episode that was
    interrupted holds a demonstration the policy was never scored against, and publishing it
    would show a video with no result behind it.
    """
    out = []
    ep = run_dir / f"episode_{record.episode:05d}"
    for name, role, label in (
        ("demonstration.mp4", "demonstration", "Demonstration — the expert, shown to the policy"),
        (
            "evaluation_same_scene.mp4",
            "evaluation",
            "Evaluation — the policy, from the identical scene",
        ),
    ):
        path = ep / name
        if not path.exists():
            continue
        out.append(
            {
                "label": f"{label} ({record.task})",
                "role": role,
                "row": f"run-{record.episode}",
                "source": _relative(path, doc_dir),
                # Carried so the document is self-verifying: the publisher hashes the source and
                # refuses it if the bytes are not the ones this document was built from. Run
                # directories are not in git, so the hash is the durable identity of the clip.
                "sha256": _sha256(path),
            }
        )
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _relative(path: Path, base: Path) -> str:
    """A source path the publisher resolves against the document, wherever the repo is checked out."""
    import os

    return os.path.relpath(path.resolve(), base.resolve())


def build(
    *,
    run_dir: str | Path,
    prose: dict[str, Any],
    oracle_dir: str | Path | None = None,
    doc_dir: str | Path | None = None,
) -> dict[str, Any]:
    """The exhibit document for one run, with an optional oracle run as its ceiling."""
    missing = [k for k in REQUIRED_PROSE if not prose.get(k)]
    if missing:
        raise ExhibitError(
            f"the prose file must say {', '.join(missing)}: these are claims the run records "
            "cannot make for themselves"
        )
    if "view" not in prose["demonstration_shown"]:
        raise ExhibitError(
            "demonstration_shown.view is required - an exhibit that does not say what the policy "
            "was shown is the mistake this format exists to prevent"
        )

    run_dir = Path(run_dir)
    doc_dir = Path(doc_dir) if doc_dir else run_dir
    run = RunDir(run_dir)
    manifest = run.manifest()
    scored = _scored(run)
    if not scored:
        raise ExhibitError(f"{run_dir} has no scored episode; there is nothing to exhibit")

    attempted = {r.task for r in run.records()}
    # Named, never zeroed: a task nobody ran is not a task that scored nothing.
    not_run = sorted(t for t in manifest.tasks if t not in attempted)

    media: list[dict[str, Any]] = []
    for record in scored:
        media.extend(_clips(run_dir, record, doc_dir))

    doc: dict[str, Any] = {
        "reference_id": prose["reference_id"],
        "headline": prose["headline"],
        "not_a_competition_score": prose["not_a_competition_score"],
        "benchmark": {
            "name": "robotwin-icil",
            "simulator": "robotwin",
            "repository": "robotensor/robotwin-icil-benchmark",
            "suite": manifest.suite,
            "benchmark_commit": manifest.benchmark_commit,
            "robotwin_commit": manifest.robotwin_commit,
            # Flattened deliberately: the reader's page renders scalar facts, so a nested
            # {name, sha256} would vanish from it silently - and the comparability caveat below
            # names the profile, so the subject's own must be on the page to compare against.
            "camera_profile": _profile_name(manifest.camera_profile()),
            "camera_profile_sha256": _profile_sha(manifest.camera_profile()),
            "gpu": (manifest.environment or {}).get("gpu"),
        },
        "protocol": {
            "evaluation_setting": manifest.evaluation_setting,
            "demonstrations_per_episode": 1,
            "demonstration_source": (
                "RoboTwin's own scripted expert, generated at evaluation time"
            ),
            "scored_from": "the identical initial scene as the demonstration",
            "global_seed": manifest.global_seed,
        },
        "demonstration_shown": dict(prose["demonstration_shown"]),
        "subject": {**_subject(manifest), **(prose.get("subject") or {})},
        "results": {
            "scored": len(scored),
            "successes": sum(1 for r in scored if r.success),
            "rows": [_row(r, "run") for r in scored],
            "not_run": not_run,
        },
        "media": media,
    }
    if prose.get("relevant_to"):
        doc["relevant_to"] = prose["relevant_to"]

    if oracle_dir:
        doc["ceiling"] = _ceiling(Path(oracle_dir), run, scored, prose)
    return doc


def _subject(manifest: Any) -> dict[str, Any]:
    """What ran, from the run's own record of it. The judgement about it comes from the prose."""
    policy = manifest.policy or {}
    out = {
        "label": policy.get("policy"),
        "adapter": (
            f"{policy['adapter']} v{policy.get('adapter_version')}"
            if policy.get("adapter")
            else None
        ),
        "action_type": policy.get("action_type"),
        "source_checkpoint": policy.get("checkpoint"),
        # An adapter reports the hash of the checkpoint it was *given*, which for a policy that
        # converts or slims one is not the artifact it loaded. Naming it `source_` keeps a reader
        # from downloading that file, hashing it and finding it does not match what ran.
        "source_checkpoint_sha256": policy.get("checkpoint_sha256"),
        # This is the identity of the weights that actually ran: the frozen-policy audit
        # recomputes it from the loaded parameters at the end of the run.
        "parameter_checksum": policy.get("parameter_checksum"),
    }
    return {k: v for k, v in out.items() if v is not None}


def _profile_name(profile: Any) -> str:
    return profile if isinstance(profile, str) else str((profile or {}).get("name", "stock"))


def _profile_sha(profile: Any) -> str | None:
    return None if isinstance(profile, str) else (profile or {}).get("sha256")


def _ceiling(
    oracle_dir: Path, run: RunDir, scored: list[Any], prose: dict[str, Any]
) -> dict[str, Any]:
    """The oracle's result on the scenes this run was actually scored on.

    Restricted to those scenes on purpose: a ceiling computed over episodes nobody was scored on
    is not a ceiling for this measurement. `report.differences()` is carried verbatim - the
    benchmark refuses to print two runs as one table when they differ, and saying exactly how
    they differ is more useful than omitting it, and far better than loosening the check.
    """
    oracle = RunDir(oracle_dir)
    want = {(r.task, r.scene_seed) for r in scored}
    rows = [
        _row(r, "oracle")
        for r in _scored(oracle)
        if (r.task, r.scene_seed) in want
    ]
    ceiling = {
        "label": prose.get("ceiling", {}).get("label", "What the harness itself achieves"),
        "why": prose.get("ceiling", {}).get("why", ""),
        "rows": rows,
        "differences": report_.differences(run.manifest(), oracle.manifest()),
    }
    if not ceiling["why"]:
        raise ExhibitError("ceiling.why must say what the oracle rows show that the subject's do not")
    return ceiling


def load_prose(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
