#!/usr/bin/env python3
r"""Check that every citation resolves in refs.bib, and that no entry is dead.

Usage:  python Manuscript/verification/verify_refs.py [--strict] [--json]

Why this gate exists
--------------------
A ``\citep{key}`` with no matching ``@article{key,}`` is not a LaTeX error. The
document compiles, BibTeX emits one warning among dozens, and the page carries
``[?]`` where the reference belongs. Nothing in the build fails, so the defect
survives review -- and it survives longest in exactly the sentence that was
written to answer a reviewer asking for a citation.

The invariant is two-sided, and the two sides catch different mistakes:

  MISSING   a key cited in the text with no entry in the .bib. A build defect:
            the PDF prints ``[?]``.
  ORPHAN    an entry that no cited key refers to. A typo fills both margins at
            once: the mistyped key is MISSING, and the entry it was aiming at
            becomes ORPHAN. So this side does not detect the mistake, it names
            the reference the typo lost. It also flags entries kept for a
            section that does not exist yet.

MISSING fails the run. ORPHAN is always reported and fails only under
``--strict``: a reference kept for a section not yet written is legitimate and
must not block a build.

Both files are read as text. No LaTeX installation is needed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_TEX = HERE.parent / "main.tex"
DEFAULT_BIB = HERE.parent / "refs.bib"

#: A natbib citation: the command, an optional star, at most two bracketed
#: optional arguments (``\citep[see][p.~3]{key}``), then the key list.
CITE_RE = re.compile(
    r"\\(?:[Cc]ite[a-zA-Z]*|nocite)\*?(?:\s*\[[^\]]*\]){0,2}\s*\{([^}]*)\}"
)

#: A BibTeX entry header. ``@string``, ``@comment`` and ``@preamble`` are
#: directives rather than references.
ENTRY_RE = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,")
DIRECTIVES = {"comment", "preamble", "string"}

CITE_CMD_RE = re.compile(r"\\(?:[Cc]ite[a-zA-Z]*|nocite)")


def strip_comments(tex: str) -> str:
    """Remove TeX comments, so a disabled citation is not counted."""
    return re.sub(r"(?<!\\)%.*", "", tex)


def citation_keys(tex: str) -> set[str]:
    """Every key that a citation command in ``tex`` points at."""
    keys: set[str] = set()
    for match in CITE_RE.finditer(strip_comments(tex)):
        for raw in match.group(1).split(","):
            key = raw.strip()
            if key and key != "*":  # \nocite{*} means "everything"
                keys.add(key)
    return keys


def bib_keys(bib: str) -> set[str]:
    """Every entry key defined in ``bib``."""
    return {
        match.group(2)
        for match in ENTRY_RE.finditer(bib)
        if match.group(1).lower() not in DIRECTIVES
    }


def compare(tex: str, bib: str) -> dict:
    """Compare the two sides of the invariant and return both margins."""
    cited = citation_keys(tex)
    entries = bib_keys(bib)
    return {
        "n_commands": len(CITE_CMD_RE.findall(strip_comments(tex))),
        "n_cited": len(cited),
        "n_entries": len(entries),
        "missing": sorted(cited - entries),
        "orphan": sorted(entries - cited),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the gate; return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--bib", type=Path, default=DEFAULT_BIB)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also fail when an entry is never cited",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    try:
        tex_text = args.tex.read_text(encoding="utf-8")
        bib_text = args.bib.read_text(encoding="utf-8")
    except OSError as exc:
        print("cannot read: " + str(exc), file=sys.stderr)
        return 2

    rep = compare(tex_text, bib_text)
    fails = bool(rep["missing"]) or (args.strict and bool(rep["orphan"]))

    if args.json:
        payload = dict(rep, tex=str(args.tex), bib=str(args.bib), ok=not fails)
        print(json.dumps(payload, indent=2))
        return 1 if fails else 0

    print("refs gate: " + args.tex.name + " vs " + args.bib.name)
    print("  citation commands  " + str(rep["n_commands"]))
    print("  distinct keys      " + str(rep["n_cited"]))
    print("  bib entries        " + str(rep["n_entries"]))
    print()
    for label in ("missing", "orphan"):
        keys = rep[label]
        tail = "" if not keys else "  " + ", ".join(keys)
        print("  " + label.upper().ljust(8) + str(len(keys)) + tail)
    print()

    if rep["missing"]:
        print("FAIL  a cited key has no entry; the PDF prints [?] where it belongs.")
        return 1
    if args.strict and rep["orphan"]:
        print("FAIL  an entry is never cited (--strict).")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
