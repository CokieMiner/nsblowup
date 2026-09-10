"""Single command line entry point.

Parsing of the individual commands lives with the code they drive:

    reference   nsblowup.reference.cli      report, wave fit, frozen force table
    dns         nsblowup.run.cli            forced DNS run, movie and archive
    rerender    nsblowup.run.rerender       rebuild the movie from an archive
    compare     nsblowup.run.compare        post-hoc exponent comparison
    verify      nsblowup.verify...          manufactured-solution check

Usage:  python cli.py <command> [options]
"""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "reference": "nsblowup.reference.cli",
    "dns": "nsblowup.run.cli",
    "rerender": "nsblowup.run.rerender",
    "compare": "nsblowup.run.compare",
    "verify": "nsblowup.verify.manufactured_solution",
}

HELP = """nsblowup - forced Navier-Stokes study of a collapsing core

commands:
  reference   construction report, wave-sector fit, frozen force tables
  dns         forced DNS of the reference construction (movie + archive)
  rerender    rebuild the movie from an archive (no re-simulation)
  compare     post-hoc exponent comparison against the paper's predictions
  verify      manufactured-solution check of solver and forcing

run `python cli.py <command> --help` for the options of a command.
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(HELP)
        return 0
    command = argv[0]
    module_name = COMMANDS.get(command)
    if module_name is None:
        print(f"unknown command {command!r}\n\n{HELP}")
        return 2
    sys.argv = [f"nsblowup {command}", *argv[1:]]
    result = importlib.import_module(module_name).main()
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
