"""
Shared plumbing for hex_walksat.py, hex_connect.py and hex_render.py.

Keep this file next to the three programs. It holds only what they have in
common that isn't an algorithm:

  * provenance metadata - every program appends one JSON record describing
    what it did to a chain stored as trailing "# meta {json}" lines in the
    grid text file, copying forward whatever it found there; hex_render.py
    writes the whole chain into the PNG. Readers of the grid format only
    look at the first n+1 lines, so files with metadata read like files
    without. Each record carries the SHA-256 of the grid it produced, so the
    next program can tell whether the grid it was given is the one the
    record describes.
  * writing the grid text file and picking unused output names;
  * turning keyword options into a parsed-arguments object and back into the
    command line they correspond to, which is what lets a Python script run
    the same code as the command line and still leave a record you could
    paste into a shell.
"""

import argparse
import datetime
import hashlib
import json
import os
import shlex

META_PREFIX = "# meta "


class UsageError(ValueError):
    """A problem with how a program was called (bad option combination,
    unusable input file). The command-line wrappers turn it into an ordinary
    error message and exit status; scripts calling the functions directly
    get the exception."""


# ---------------------------------------------------------------------------
# Provenance metadata
# ---------------------------------------------------------------------------

def grid_digest(grid, n):
    """SHA-256 of the grid in its canonical text form (first line n, then n
    rows of 0/1) - independent of any metadata lines and of the file name, so
    a renamed file can still be matched to its record, and a hand-edited grid
    can be told apart from the one a record describes."""
    text = f"{n}\n" + "".join("".join("1" if c else "0" for c in row) + "\n" for row in grid)
    return hashlib.sha256(text.encode()).hexdigest()


def make_record(program, args, argv, seed_generated=False, result=None, extra=None):
    """One provenance record. `command` is the invocation as given (argv);
    `reproduce` equals it except when the seed was generated (none was
    given), in which case it is the command with that seed added, which is
    what actually repeats the run."""
    command = shlex.join(argv)
    rec = {
        "program": program,
        "command": command,
        "reproduce": (shlex.join(list(argv) + ["--seed", str(args.seed)])
                      if seed_generated else command),
        "seed_generated": bool(seed_generated),
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "args": vars(args),
    }
    if result:
        rec["result"] = result
    if extra:
        rec.update(extra)
    return rec


def read_meta(path):
    """The provenance records stored in a grid file (oldest first). Lines
    that aren't valid records are skipped; a file with none gives []."""
    out = []
    with open(path) as f:
        for line in f:
            if line.startswith(META_PREFIX):
                try:
                    rec = json.loads(line[len(META_PREFIX):])
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    return out


def pipeline_commands(records):
    """The commands in a chain, one per line, oldest first."""
    return "\n".join(r.get("reproduce") or r.get("command", "") for r in records)


def meta_pnginfo(records):
    """PNG text chunks holding a provenance chain: hex_pipeline (the full
    records as JSON) and hex_pipeline_commands (just the commands)."""
    from PIL import PngImagePlugin
    info = PngImagePlugin.PngInfo()
    info.add_text("hex_pipeline", json.dumps(records, sort_keys=True, default=str))
    info.add_text("hex_pipeline_commands", pipeline_commands(records))
    return info


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def write_grid_txt(grid, n, path, meta=None):
    """Write the solved grid as plain 0/1 text - decoupled from any
    particular rendering approach so other programs can consume it.

    Format: first line is n, followed by n lines of n characters each
    ('1' for live, '0' for dead), row-major (row i, then columns j in
    that row), matching the (i, j) indexing used everywhere else here.

    meta: optional list of provenance records (see make_record), written
    after the grid as "# meta {json}" lines. Anything after the first n+1
    lines is ignored by readers of the format.
    """
    with open(path, "w") as f:
        f.write(f"{n}\n")
        for i in range(n):
            f.write("".join("1" if grid[i][j] else "0" for j in range(n)))
            f.write("\n")
        for rec in (meta or []):
            f.write(META_PREFIX + json.dumps(rec, sort_keys=True, default=str) + "\n")
    return path


def unique_path(path):
    """Append _1, _2, _3, ... before the extension until an unused path is
    found, so repeated runs (e.g. with no --seed) don't clobber earlier
    output."""
    base, ext = os.path.splitext(path)
    n = 1
    candidate = f"{base}_{n}{ext}"
    while os.path.exists(candidate):
        n += 1
        candidate = f"{base}_{n}{ext}"
    return candidate


# ---------------------------------------------------------------------------
# Options <-> command line
# ---------------------------------------------------------------------------

def _option_actions(parser):
    return [a for a in parser._actions if a.dest != "help"]


def namespace_from(parser, options):
    """Parsed-arguments object for `parser` built from keyword options named
    after the argument's destination (--wall-boundary -> wall_boundary).
    Options not given take the parser's defaults. A string given for an
    option that has a type (e.g. kick_size="5%") is converted the way the
    command line would; unknown options and values outside an option's
    choices raise instead of being silently ignored."""
    actions = {a.dest: a for a in _option_actions(parser)}
    unknown = sorted(set(options) - set(actions))
    if unknown:
        raise TypeError("unknown option(s): " + ", ".join(unknown))
    values = {d: a.default for d, a in actions.items()}
    for dest, v in options.items():
        a = actions[dest]
        if isinstance(v, str) and a.type not in (None, str):
            v = a.type(v)
        if a.choices is not None and v not in a.choices:
            raise UsageError(f"{dest}: {v!r} is not one of {list(a.choices)}")
        values[dest] = v
    return argparse.Namespace(**values)


def canonical_argv(parser, args, program, formatters=None):
    """The command line (as a list, starting with `program`) that parses
    back to `args`: positional values first, then each option whose value
    differs from its default, spelled with its long name. `formatters` maps
    an option's type function to a function turning a parsed value back into
    text, for types that aren't plain str() round trips."""
    formatters = formatters or {}
    argv = [program]
    for a in _option_actions(parser):
        v = getattr(args, a.dest)
        if not a.option_strings:                 # positional
            argv.append(str(v))
            continue
        if v == a.default:
            continue
        opt = max(a.option_strings, key=len)     # --wall-boundary over -w
        if a.nargs == 0:                         # store_true and friends
            argv.append(opt)
        else:
            argv.extend([opt, formatters.get(a.type, str)(v)])
    return argv


def args_for_command(parser, argv):
    """Parse a command line as canonical_argv builds it, raising UsageError
    instead of exiting. The functions that run a program from Python use the
    arguments this returns, not the options they were handed, so a run from
    Python is by construction the same run as the recorded command - even
    where a value can't survive being written as text (a fraction written as
    a percentage can differ in its last bit)."""
    try:
        return parser.parse_args(argv[1:])
    except SystemExit:
        raise UsageError("the options could not be turned into a valid command line") from None
