"""Run a command under macOS `/usr/bin/time -l` and record runtime and peak memory.

"peak memory footprint" is what Activity Monitor calls Memory; it includes unified
(GPU/Metal) allocations, which max RSS undercounts. It only covers the direct
child process, so wrap the final interpreter, never `uv run` / `uvx` (those spawn
a child and the footprint you get is uv's ~10 MB). Max RSS covers descendants and
is recorded as a cross-check.

CLI: python measure.py --out meta.json -- <cmd> [args...]
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from common import write_json

_TIME_RE = {
    "real": re.compile(r"^\s*([\d.]+)\s+real", re.MULTILINE),
    "max_rss": re.compile(r"^\s*(\d+)\s+maximum resident set size", re.MULTILINE),
    "footprint": re.compile(r"^\s*(\d+)\s+peak memory footprint", re.MULTILINE),
}


def measure(cmd: list[str], *, log: Path | None = None, **kwargs) -> dict:
    """Run ``cmd``; return runtime_s, peak_mem_mb, max_rss_mb.

    The command's stdout/stderr go to ``log`` (or are inherited). Raises
    CalledProcessError on a non-zero exit.
    """
    with tempfile.NamedTemporaryFile("r", suffix=".time") as tf:
        full = ["/usr/bin/time", "-l", "-o", tf.name, *cmd]
        if log is not None:
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w") as fh:
                proc = subprocess.run(full, stdout=fh, stderr=subprocess.STDOUT, check=False, **kwargs)  # noqa: S603
        else:
            proc = subprocess.run(full, check=False, **kwargs)  # noqa: S603
        report = Path(tf.name).read_text()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    values = {k: float(r.search(report).group(1)) for k, r in _TIME_RE.items()}
    return {
        "runtime_s": round(values["real"], 2),
        "peak_mem_mb": round(values["footprint"] / 2**20, 1),
        "max_rss_mb": round(values["max_rss"] / 2**20, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure runtime and peak memory of a command")
    parser.add_argument("--out", type=Path, help="write the measurement JSON here")
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    result = measure(cmd)
    if args.out:
        write_json(args.out, result)
    print(result, file=sys.stderr)


if __name__ == "__main__":
    main()
