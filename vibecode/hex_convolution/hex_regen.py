"""
Regenerate a result from the provenance chain stored in a PNG (or a grid file).

Every PNG hex_render.py writes carries the chain of records left by the
programs that produced it (solver -> connect -> render); grid files carry the
chain up to where they were written. This reads that chain and runs the same
steps again, in order, feeding each step's output grid to the next.

    py hex_regen.py hex_render_19.png
        Redo it exactly: each step reuses its recorded seed and options, and
        each grid it produces is checked against the hash recorded for it, so
        you are told whether you got the same grid or a different one (the
        code changed since, or the run wasn't deterministic).

    py hex_regen.py hex_render_19.png --new-seeds
        The same settings with a fresh random seed for every step that has
        one - a different result of the same kind. The new seeds are recorded
        in the new files, so that run can be repeated later too.

    py hex_regen.py hex_render_19.png --new-seeds --count 10
        Make a batch of ten variants: once the settings are right, any PNG
        carrying them is the recipe. Each variant must pass a quality gate -
        the solver finished and, if the chain has a connect step, it reached a
        single connected loop. A seed that fails is retried with a new one,
        up to --max-attempts (default 3 x count) attempts in all; failed
        attempts' files are kept unless you pass --discard-failed. A summary
        at the end lists every image made and the seeds behind it.

    py hex_regen.py hex_render_19.png --from render --grid hex_grid_connected_19.txt \\
            --set render.thickness=1.3
        Start at a later step on a grid you already have (no re-solving) and
        change an option. --set STEP.OPTION=VALUE works for any step, named
        walksat / connect / render or numbered from 1; option names are the
        command-line names (hyphen or underscore).

    py hex_regen.py hex_render_19.png --dry-run
        Show the steps and the equivalent commands without running anything.

Exit status is 1 if any regenerated grid differs from its recorded hash.
New files get fresh numbered names, so nothing earlier is overwritten. If you
change an option, steps from there on are not compared with the recorded
grids (they are expected to differ). The renderer records no hash of the image
it writes, so a re-render is checked only through the grid it started from.

Needs hex_common.py, hex_walksat.py, hex_connect.py and hex_render.py in the
same folder.
"""

import argparse
import json
import os
import sys

from PIL import Image

import hex_connect
import hex_render
import hex_walksat
from hex_common import UsageError, canonical_argv, grid_digest, namespace_from, read_meta

PROGRAMS = {
    "hex_walksat.py": ("walksat", hex_walksat),
    "hex_connect.py": ("connect", hex_connect),
    "hex_render.py": ("render", hex_render),
}
_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n"}


# ---------------------------------------------------------------------------
# Reading the chain
# ---------------------------------------------------------------------------

def load_chain(path):
    """The provenance records stored in a PNG (hex_pipeline text chunk) or a
    grid text file (trailing '# meta' lines), oldest first."""
    if not os.path.exists(path):
        raise UsageError(f"{path}: no such file")
    if path.lower().endswith(".png"):
        text = Image.open(path).text
        if "hex_pipeline" not in text:
            raise UsageError(
                f"{path} has no pipeline record (it was made before provenance metadata "
                "existed). Only its render command was saved"
                + (f": {text['hex_render_command']}" if "hex_render_command" in text else ".")
                + " The earlier steps can't be rebuilt from this file.")
        chain = json.loads(text["hex_pipeline"])
    else:
        chain = read_meta(path)
    if not chain:
        raise UsageError(f"{path} carries no provenance records, so there is nothing to regenerate")
    for i, rec in enumerate(chain, 1):
        if rec.get("program") not in PROGRAMS:
            raise UsageError(f"step {i} of the chain is from an unknown program: {rec.get('program')!r}")
    return chain


def _step_index(chain, selector):
    """0-based index of a step named by 'walksat'/'connect'/'render' (the
    first of that kind) or by its 1-based number."""
    if selector.isdigit():
        i = int(selector) - 1
        if not 0 <= i < len(chain):
            raise UsageError(f"there is no step {selector}; the chain has {len(chain)}")
        return i
    for i, rec in enumerate(chain):
        if PROGRAMS[rec["program"]][0] == selector:
            return i
    raise UsageError(f"the chain has no {selector!r} step "
                     f"(it has: {', '.join(PROGRAMS[r['program']][0] for r in chain)})")


def _parse_override(text, chain):
    if "=" not in text or "." not in text.split("=", 1)[0]:
        raise UsageError(f"--set {text!r}: expected STEP.OPTION=VALUE, e.g. render.thickness=1.3")
    key, value = text.split("=", 1)
    step, option = key.split(".", 1)
    return _step_index(chain, step), option.replace("-", "_"), value


def _convert(parser, dest, value):
    """A --set value as the option's own type: flags take true/false, others
    are left as text for namespace_from to convert like the command line."""
    action = next((a for a in parser._actions if a.dest == dest), None)
    if action is None:
        raise UsageError(f"unknown option {dest!r}")
    if action.nargs == 0:
        v = value.strip().lower()
        if v in _TRUE:
            return True
        if v in _FALSE:
            return False
        raise UsageError(f"{dest} is a yes/no option; use true or false, not {value!r}")
    return value


# ---------------------------------------------------------------------------
# Planning and running
# ---------------------------------------------------------------------------

def build_plan(chain, new_seeds=False, overrides=()):
    """One dict per recorded step: its program, the options to run it with,
    notes about differences from the recorded run, and whether its result can
    be compared with the recorded one. `overrides` are (step index, option,
    text) triples."""
    plan = []
    changed_from = None                  # first step whose result may legitimately differ
    for i, rec in enumerate(chain):
        prog = rec["program"]
        parser = PROGRAMS[prog][1].build_parser()
        valid = {a.dest for a in parser._actions if a.dest != "help"}
        args = dict(rec.get("args", {}))
        notes = []
        for k in sorted(k for k in args if k not in valid):
            del args[k]
            notes.append(f"recorded option '{k}' no longer exists; ignored")
        added = sorted(valid - set(args) - {"input"})
        if added:
            notes.append("options added since this was recorded take today's defaults: "
                         + ", ".join(added))
        args.pop("input", None)          # wired from the previous step's output instead
        seeded = "seed" in valid
        if new_seeds and seeded:
            args["seed"] = None
            changed_from = i if changed_from is None else changed_from
        for (j, option, text) in overrides:
            if j == i:
                if option == "input":
                    raise UsageError("the input grid comes from the previous step; use --grid to supply one")
                if option not in valid:
                    raise UsageError(f"step {i + 1} ({PROGRAMS[prog][0]}) has no option {option!r}")
                args[option] = _convert(parser, option, text)
                changed_from = i if changed_from is None else min(changed_from, i)
        if args.get("time_limit"):
            notes.append("it was run with a time limit, so its result depends on machine speed "
                         "and may not repeat exactly")
        plan.append({"index": i, "program": prog, "name": PROGRAMS[prog][0], "record": rec,
                     "args": args, "parser": parser, "notes": notes, "seeded": seeded})
    for step in plan:
        step["comparable"] = changed_from is None or step["index"] < changed_from
    return plan


def _command(step, input_text):
    parser = step["parser"]
    opts = dict(step["args"])
    if any(not a.option_strings for a in parser._actions):
        opts["input"] = input_text
    argv = canonical_argv(parser, namespace_from(parser, opts), step["program"],
                          getattr(PROGRAMS[step["program"]][1], "_FORMATTERS", None))
    return " ".join(argv)


def _quiet(*a, **k):
    pass


def regenerate(source, new_seeds=False, from_step=None, to_step=None, grid=None,
               overrides=(), dry_run=False, quiet=False, echo=print, plan_echo=None,
               gate=False):
    """Rebuild a result from the chain recorded in `source` (a PNG or grid
    file). Returns a list with one dict per step run: its program, the
    outputs it wrote, the seed it used, and `status` - 'same' / 'different'
    compared with the recorded grid, or a note on why it wasn't compared.

    overrides: ["STEP.OPTION=VALUE", ...]; from_step / to_step: step name or
    number; grid: the input for the first step run when it isn't the solver.
    plan_echo: where the plan lines go (default: echo). gate: stop at the
    first step whose result isn't usable - the solver didn't finish, or
    connect didn't reach a single connected loop - and mark it 'failed'."""
    chain = load_chain(source)
    parsed = [_parse_override(t, chain) for t in overrides]
    plan = build_plan(chain, new_seeds=new_seeds, overrides=parsed)
    start = _step_index(chain, str(from_step)) if from_step else 0
    stop = _step_index(chain, str(to_step)) if to_step else len(chain) - 1
    if start > stop:
        raise UsageError("--from comes after --to")
    out = []
    say = plan_echo or echo

    say(f"regenerating from {source}: {len(chain)} recorded step(s), "
         + ("new random seeds" if new_seeds else "recorded seeds (exact)")
         + (", dry run" if dry_run else ""))
    prev_grid = None
    for step in plan[start:stop + 1]:
        i, rec, prog = step["index"], step["record"], step["program"]
        label = f"step {i + 1}/{len(chain)}: {step['name']}"
        # ---- where this step's input grid comes from
        input_path, input_note = None, None
        if prog != "hex_walksat.py":
            if i == start:
                input_path = grid or rec.get("input_file")
                if not dry_run:
                    if not input_path or not os.path.exists(input_path):
                        raise UsageError(
                            f"{label} needs its input grid ({rec.get('input_file')}), which isn't "
                            "here. Pass --grid PATH, or start from an earlier step.")
                    g, n = hex_connect.read_grid(input_path)
                    want = rec.get("input_grid_sha256")
                    if want:
                        input_note = ("input grid matches the recorded one" if grid_digest(g, n) == want
                                      else "WARNING: this input grid differs from the one recorded for "
                                           "this step; results will differ from the recorded run")
            else:
                input_path = prev_grid
        # ---- show the plan
        seed_text = ""
        if step["seeded"]:
            seed_text = (" (new random seed)" if step["args"].get("seed") is None
                         else f" (seed {step['args']['seed']})")
        say(f"\n== {label}{seed_text}")
        say("   " + _command(step, input_path or "<previous step's grid>"))
        for note in step["notes"]:
            say("   note: " + note)
        if input_note:
            say("   " + input_note)
        if dry_run:
            prev_grid = "<previous step's grid>"
            out.append({"program": prog, "status": "dry run"})
            continue
        # ---- run it
        opts = dict(step["args"])
        if quiet and any(a.dest == "quiet" for a in step["parser"]._actions):
            opts["quiet"] = True
        step_echo = _quiet if quiet else echo
        module = PROGRAMS[prog][1]
        if prog == "hex_walksat.py":
            res = module.generate(echo=step_echo, **opts)
            prev_grid = res.grid_path
            outputs = {"grid": res.grid_path, "image": res.image_path}
        elif prog == "hex_connect.py":
            res = module.connect_file(input_path, echo=step_echo, **opts)
            prev_grid = res.grid_path
            outputs = {"grid": res.grid_path}
        else:
            res = module.render_file(input_path, echo=step_echo, **opts)
            outputs = {"image": res.image_path}
        # ---- compare with what was recorded
        status = "no hash was recorded for this output"
        if prog != "hex_render.py" and rec.get("grid_sha256"):
            if not step["comparable"]:
                status = "settings changed, so not compared with the recorded grid"
            elif res.record["grid_sha256"] == rec["grid_sha256"]:
                status = "same"
            else:
                status = "different"
        failed = None
        if gate:
            if prog == "hex_walksat.py" and not res.solved:
                failed = "the solver did not finish within its iteration limit"
            elif prog == "hex_connect.py" and not res.single_connected_loop:
                failed = "no single connected loop was reached"
        seed = res.record["args"].get("seed") if step["seeded"] else None
        out.append({"program": prog, "seed": seed, "outputs": outputs, "status": status,
                    "failed": failed, "record": res.record if hasattr(res, "record") else None})
        shown = ", ".join(f"{k}: {v}" for k, v in outputs.items())
        verdict = {"same": "reproduced exactly - identical grid",
                   "different": "DIFFERENT from the recorded grid (the code has changed since, "
                                "or this run isn't deterministic)"}.get(status, status)
        say(f"   -> {shown}\n   -> {verdict}")
        if failed:
            say(f"   -> FAILED: {failed}")
            break
    if out and not dry_run and not out[-1].get("failed"):
        last = out[-1]["outputs"]
        say("\nfinal: " + (last.get("image") or last.get("grid")))
    return out


def _outputs_of(steps):
    return [p for s in steps for p in (s.get("outputs") or {}).values()]


def variants(source, count=1, max_attempts=None, discard_failed=False, quiet=False,
             echo=print, dry_run=False, **kwargs):
    """Generate `count` new patterns from the settings recorded in `source`:
    run the chain again with fresh random seeds, over and over, keeping the
    attempts that pass the quality gate (see regenerate). A failed attempt is
    replaced by another, up to max_attempts attempts in all (default 3 x
    count). discard_failed deletes the files a failed attempt wrote (only
    files made by this call). Other keyword options (from_step, to_step, grid,
    overrides) are passed to regenerate.

    Returns {"good": [...], "failed": [...], "attempts": n}; each entry is the
    list of per-step dicts of that attempt."""
    if count < 1:
        raise UsageError("--count must be at least 1")
    chain = load_chain(source)
    start = _step_index(chain, str(kwargs["from_step"])) if kwargs.get("from_step") else 0
    stop = _step_index(chain, str(kwargs["to_step"])) if kwargs.get("to_step") else len(chain) - 1
    plan = build_plan(chain, new_seeds=True)
    if not any(s["seeded"] for s in plan[start:stop + 1]):
        raise UsageError("none of the steps being run uses a seed, so every variant would be "
                         "identical; start from an earlier step")
    if dry_run:
        regenerate(source, new_seeds=True, dry_run=True, echo=echo, gate=True, **kwargs)
        return {"good": [], "failed": [], "attempts": 0}
    max_attempts = max_attempts or 3 * count
    good, failed = [], []
    attempt = 0
    while len(good) < count and attempt < max_attempts:
        attempt += 1
        echo(f"\n--- attempt {attempt} of at most {max_attempts}: {len(good)} of {count} made")
        steps = regenerate(source, new_seeds=True, quiet=quiet, echo=echo, gate=True,
                           plan_echo=lambda *a, **k: None, **kwargs)
        seeds = [s["seed"] for s in steps if s.get("seed") is not None]
        if steps and not steps[-1].get("failed"):
            last = steps[-1]["outputs"]
            echo(f"    kept: {last.get('image') or last.get('grid')}  (seeds {', '.join(map(str, seeds))})")
            good.append(steps)
        else:
            echo(f"    not kept: {steps[-1]['failed']}  (seeds {', '.join(map(str, seeds))})")
            if discard_failed:
                for p in _outputs_of(steps):
                    if p and os.path.exists(p):
                        os.remove(p)
            failed.append(steps)
    echo(f"\n=== {len(good)} of {count} requested variant(s) made in {attempt} attempt(s)")
    for k, steps in enumerate(good, 1):
        last = steps[-1]["outputs"]
        seeds = ", ".join(str(s["seed"]) for s in steps if s.get("seed") is not None)
        echo(f"  {k}. {last.get('image') or last.get('grid')}   (seeds {seeds})")
    if failed:
        reasons = {}
        for steps in failed:
            reasons[steps[-1]["failed"]] = reasons.get(steps[-1]["failed"], 0) + 1
        echo(f"  {len(failed)} attempt(s) not kept (" + "; ".join(f"{n}x {r}" for r, n in reasons.items())
             + ("; their files were deleted)" if discard_failed else "; their files were left in place)"))
    return {"good": good, "failed": failed, "attempts": attempt}


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Regenerate a result from the provenance chain in a PNG or grid file.",
        epilog="Examples: hex_regen.py out.png | hex_regen.py out.png --new-seeds | "
               "hex_regen.py out.png --from render --grid g.txt --set render.thickness=1.3")
    p.add_argument("source", help="a PNG written by hex_render.py / hex_walksat.py, or a grid "
                                  "text file, carrying a recorded pipeline")
    p.add_argument("--new-seeds", action="store_true",
                   help="use a fresh random seed for every step that has one instead of the "
                        "recorded seeds: same settings, different result")
    p.add_argument("--count", type=int, default=None, metavar="N",
                   help="with --new-seeds: make N variants (each must pass the quality gate: "
                        "solver finished, single connected loop if there is a connect step); a "
                        "failed seed is retried with a new one")
    p.add_argument("--max-attempts", type=int, default=None, metavar="M",
                   help="with --new-seeds: give up after M attempts in all (default 3 x count)")
    p.add_argument("--discard-failed", action="store_true",
                   help="with --new-seeds: delete the files written by attempts that didn't "
                        "pass the gate (default: leave them)")
    p.add_argument("--from", dest="from_step", metavar="STEP",
                   help="start at this step (walksat, connect, render, or its number) instead of "
                        "the first; needs the input grid (the recorded file, or --grid)")
    p.add_argument("--to", dest="to_step", metavar="STEP", help="stop after this step")
    p.add_argument("--grid", metavar="PATH",
                   help="input grid for the first step when starting at connect or render")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="STEP.OPT=VALUE",
                   help="override an option of a step, e.g. render.thickness=1.3 or "
                        "walksat.n=40; repeatable")
    p.add_argument("--dry-run", action="store_true",
                   help="show the steps and their equivalent commands without running anything")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="hide each step's own output; print only the regeneration summary")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if ((args.count is not None or args.max_attempts is not None or args.discard_failed)
                and not args.new_seeds):
            raise UsageError("--count, --max-attempts and --discard-failed need --new-seeds: an "
                             "exact regeneration is deterministic, so repeating it gives the same result")
        if args.new_seeds:
            wanted = 1 if args.count is None else args.count
            res = variants(args.source, count=wanted, max_attempts=args.max_attempts,
                           discard_failed=args.discard_failed, quiet=args.quiet,
                           dry_run=args.dry_run, from_step=args.from_step, to_step=args.to_step,
                           grid=args.grid, overrides=args.overrides)
            if not args.dry_run and len(res["good"]) < wanted:
                sys.exit(1)
            return
        results = regenerate(args.source, new_seeds=args.new_seeds, from_step=args.from_step,
                             to_step=args.to_step, grid=args.grid, overrides=args.overrides,
                             dry_run=args.dry_run, quiet=args.quiet)
    except UsageError as e:
        sys.exit(str(e))
    if not args.dry_run and any(r["status"] == "different" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
