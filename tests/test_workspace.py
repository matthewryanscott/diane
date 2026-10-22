"""Exercise startup paths in fresh processes, without patching workspace constants."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import wave
from datetime import datetime, timedelta
from pathlib import Path

import pytest


@pytest.mark.parametrize("layout", ["source-checkout", "relocated-package"])
def test_cli_uses_startup_workspace_not_package_location(tmp_path, layout):
    package_root = tmp_path / ("checkout" if layout == "source-checkout" else "site-packages")
    package_root.mkdir()
    shutil.copytree(Path(__file__).resolve().parent.parent / "diane", package_root / "diane")
    workspace = package_root if layout == "source-checkout" else tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    recordings = workspace / "Recordings"
    recordings.mkdir()
    filename = "20260402 1700 Workspace meeting.wav"
    source = recordings / filename
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\x00\x00" * 8000)
    original_audio = source.read_bytes()
    recording_id = hashlib.sha256(source.stem.encode()).hexdigest()[:8]
    birthtime = getattr(source.stat(), "st_birthtime", None)
    recorded = (
        datetime.fromtimestamp(birthtime).astimezone()
        if birthtime is not None
        else datetime(2026, 4, 2, 17).astimezone()
    )
    transcripts = workspace / "transcripts"
    transcripts.mkdir()
    transcript = transcripts / f"{recorded:%Y%m%dT%H%M%S}-{recording_id}.txt"
    transcript.write_text(
        f"---\nrecorded: '{recorded.isoformat()}'\nduration: 1\n"
        "title: Workspace meeting\nspeakers: [SPEAKER_00]\n---\n"
        "[00:00] SPEAKER_00: Workspace sentence.\n"
    )
    (workspace / "server-url.txt").write_text("http://localhost:11434\n")
    (workspace / "diane.toml").write_text('[[servers]]\nurl = "@server-url.txt"\n')
    if layout == "relocated-package":
        # The former __file__-based root would select these decoys instead.
        (package_root / "Recordings").mkdir()
        (package_root / "transcripts").mkdir()
        (package_root / "diane.toml").write_text('[[servers]]\nurl = "wrong-package-root"\n')
    env = {
        **os.environ,
        "PYTHONPATH": str(package_root),
        "PYTHONDONTWRITEBYTECODE": "1",
        "SDL_AUDIODRIVER": "dummy",
        "NO_COLOR": "1",
        "COLUMNS": "240",
    }

    def run(code, *arguments, cwd=workspace):
        return subprocess.run(
            [sys.executable, "-c", code, *arguments], cwd=cwd, env=env,
            capture_output=True, text=True, check=True,
        )

    paths = json.loads(run(
        "import json; from pathlib import Path; import diane.db as db; import diane.cli as cli; "
        "import diane.config as config; import diane.transcribe as transcribe; "
        "print(json.dumps([str(db.PROJECT_ROOT), str(db.RECORDINGS_DIR), "
        "str(cli.TRANSCRIPTS_DIR), str(config.CONFIG_PATH), "
        "str(transcribe.RECORDINGS_DIR), config.get_config().servers[0].url, str(Path(db.__file__).resolve())]))"
    ).stdout)
    assert paths == [
        str(workspace.resolve()), str(recordings.resolve()), str(transcripts.resolve()),
        str((workspace / "diane.toml").resolve()), str(recordings.resolve()),
        "http://localhost:11434", str((package_root / "diane" / "db.py").resolve()),
    ]
    cli = "from diane.cli import app; app()"
    listing = run(cli, "recording", "list", "--limit", "0")
    assert recording_id in listing.stdout
    assert "Workspace meeting" in listing.stdout
    pending = run(cli, "transcribe", "all", "--dry-run")
    assert "All recordings are already transcribed" in pending.stderr
    entries = [json.loads(line) for line in run(cli, "transcript", "list", "--json").stdout.splitlines()]
    assert [entry["path"] for entry in entries] == [str(transcript.resolve())]
    assert run(cli, "transcript", "path", recording_id).stdout.strip() == str(transcript.resolve())
    # This real CLI write must update the workspace file, never the package tree.
    run(cli, "transcript", "absolute", recording_id)
    assert "timestamps: absolute" in transcript.read_text()
    start = recorded.replace(microsecond=0, tzinfo=None)
    sliced = run(
        cli, "slice", "--start", start.isoformat(),
        "--end", (start + timedelta(seconds=1)).isoformat(),
    )
    assert "[transcripts] SPEAKER_00: Workspace sentence." in sliced.stdout
    assert source.read_bytes() == original_audio
    if layout == "relocated-package":
        assert list((package_root / "transcripts").iterdir()) == []

    # An unrelated cwd does not fall back to the checkout or installed package.
    empty = tmp_path / "empty-workspace"
    empty.mkdir()
    assert run(cli, "transcript", "list", "--json", cwd=empty).stdout == ""
    defaults = run(
        "from diane.config import load_config; from diane.db import PROJECT_ROOT; "
        "print(PROJECT_ROOT); print(len(load_config().servers))", cwd=empty,
    )
    assert defaults.stdout.splitlines() == [str(empty.resolve()), "0"]
