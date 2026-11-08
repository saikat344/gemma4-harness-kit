"""Run the compaction characterization suite against several google-adk venvs.

Usage (from the repo root):
  python3 harness/adk_spec/run_matrix.py [--out FILE.md] VENV...

Each VENV is a virtualenv directory with google-adk, litellm and pytest
installed (see docs/harness-contracts.md §7 in the release;
source/adk-compaction-tests.md §7 in the dev repo). Prints a markdown
pass/fail table, one column per venv, labelled by the installed google-adk
version.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(venv: Path) -> tuple[str, dict[str, str]]:
  py = venv / "bin" / "python"
  version = subprocess.run(
      [str(py), "-I", "-c", "from importlib.metadata import version; print(version('google-adk'))"],
      capture_output=True, text=True, check=True,
  ).stdout.strip()
  with tempfile.TemporaryDirectory() as tmp:
    xml = Path(tmp) / "r.xml"
    subprocess.run(
        [str(py), "-I", "-m", "pytest", "-c", str(HERE / "pytest.ini"), str(HERE),
         f"--junitxml={xml}"],
        capture_output=True, text=True,
    )
    results = {}
    for case in ET.parse(xml).iter("testcase"):
      if case.find("failure") is not None:
        outcome = "FAIL"
      elif case.find("error") is not None:
        outcome = "ERROR"
      elif case.find("skipped") is not None:
        outcome = "skip"
      else:
        outcome = "pass"
      results[case.get("name")] = outcome
  return f"{version} ({venv.name})", results


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("venvs", nargs="+", type=Path)
  ap.add_argument("--out", type=Path)
  args = ap.parse_args()
  cols = [run(v) for v in args.venvs]
  names = sorted({n for _, r in cols for n in r})
  lines = ["| test | " + " | ".join(c for c, _ in cols) + " |",
           "|---|" + "---|" * len(cols)]
  for n in names:
    lines.append(f"| `{n}` | " + " | ".join(r.get(n, "—") for _, r in cols) + " |")
  table = "\n".join(lines)
  print(table)
  if args.out:
    args.out.write_text(table + "\n")


if __name__ == "__main__":
  main()
