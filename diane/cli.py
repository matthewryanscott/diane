import json
import sys
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
from rich.console import Console
from rich.table import Table

from diane.db import (
    PROJECT_ROOT,
    RECORDINGS_DIR,
    Recording,
    canonical_transcript_path,
    find_recording,
    find_recording_transcript,
    find_transcript,
    get_recordings,
    list_transcripts,
    pending_recordings,
    transcribed_ids,
)
from diane.frontmatter import (
    build_output,
    ensure_frontmatter,
    frontmatter_from_recording,
    read_frontmatter_head,
    transcript_record,
)
from diane.voiceprints import IdentityDocument, load_identity, new_identity

_HELP = """\
Transcribe recurring meetings locally, with diarization and speaker names.
The intended use is recurring meetings with 2-6 people. Check results against audio.
Run commands from the workspace containing Recordings/ and transcripts/.

To read a recent meeting, run these in order:

  1. uv run diane transcribe all --dry-run
     Preview discovered recordings without transcripts. Discovery scans date-first
     or title-first filenames in Recordings/, not its database. Sources stay read-only.
  2. uv run diane transcribe all
     Transcribe the pending discovered recordings locally.
  3. uv run diane transcript list
     Newest first. Match the meeting by Date, Title and Speakers; take its ID.
     Add --json for one JSON object per transcript, for scripts.
  4. uv run diane transcript path <ID>
     Prints the transcript's absolute path. Read that file. A unique ID prefix
     is enough.

A transcript has YAML frontmatter (recorded, duration, title, speakers), then
speaker turns with "[MM:SS]" or "[HH:MM:SS]" offsets from "recorded".
`transcript absolute` converts offsets to full local timestamps.
SPEAKER_00-style labels are speakers not yet identified. Naming them
(`uv run diane speakers <ID>`) plays audio and asks for user input.
Use `uv run diane speakers audit <ID>` to review automatic voiceprint names.
"""

_EPILOGUE = """\
To read what was said during a time range:

    uv run diane transcript absolute <ID>
    uv run diane slice --start 2026-04-05T14:00 --end 2026-04-05T15:30

`diane slice` scans workspace-wide .txt files with absolute-timestamp metadata,
excluding hidden directories. It prints lines in time order, labeled with the
file's parent directory. Convert a transcript first with
`uv run diane transcript absolute <ID>` (or `... absolute all`).

Timestamps are YYYY-MM-DDTHH:MM[:SS], local time.

JSONL output:
  transcript list --json  {"id", "recorded", "duration_seconds", "title", "speakers", "path"}
"""

app = App(
    name="diane",
    help=_HELP,
    help_epilogue=_EPILOGUE,
    help_format="plaintext",
)

recording_app = App(name="recording", help="Recording operations.")
app.command(recording_app)

transcript_app = App(name="transcript", help="Transcript operations.")
app.command(transcript_app)

TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"


@recording_app.command(name="list")
def recording_list(*, limit: int = 25) -> None:
    """List recordings discovered from date-first or title-first filenames in Recordings/."""
    recs = get_recordings()
    if limit > 0:
        recs = recs[:limit]

    transcribed = transcribed_ids(TRANSCRIPTS_DIR)

    table = Table(show_edge=False, pad_edge=False)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Date", style="green", no_wrap=True)
    table.add_column("Duration", style="yellow", justify="right", no_wrap=True)
    table.add_column("Title")
    table.add_column("T", no_wrap=True)

    for r in recs:
        dur = r.duration.total_seconds()
        if dur >= 3600:
            hours = int(dur // 3600)
            mins = int((dur % 3600) // 60)
            dur_str = f"{hours}h {mins:02d}m"
        elif dur >= 60:
            mins = int(dur // 60)
            secs = int(dur % 60)
            dur_str = f"{mins}m {secs:02d}s"
        else:
            dur_str = f"{int(dur)}s"

        date_str = r.date.astimezone().strftime("%Y-%m-%d %H:%M") if r.date else ""

        has_transcript = "[green]Y[/green]" if r.short_id in transcribed else ""
        table.add_row(r.short_id, date_str, dur_str, r.title, has_transcript)

    console = Console()
    console.print(table)
    console.print(f"\n[dim]{len(recs)} recording(s) shown[/dim]")


@transcript_app.command(name="list")
def transcript_list(*, as_json: Annotated[bool, Parameter(name="--json")] = False) -> None:
    """List available transcripts. --json prints one JSON object per line, newest first."""
    if as_json:
        for entry in sorted(list_transcripts(TRANSCRIPTS_DIR), key=lambda t: t.path, reverse=True):
            print(json.dumps(transcript_record(entry.path)))
        return

    if not TRANSCRIPTS_DIR.exists():
        Console().print("[dim]No transcripts yet.[/dim]")
        return

    files = sorted(
        list_transcripts(TRANSCRIPTS_DIR, id_policy="delimited-only", include_recordings=True),
        key=lambda t: t.path,
        reverse=True,
    )
    if not files:
        Console().print("[dim]No transcripts yet.[/dim]")
        return

    table = Table(show_edge=False, pad_edge=False)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Date", style="green", no_wrap=True)
    table.add_column("Duration", style="yellow", justify="right", no_wrap=True)
    table.add_column("Title")
    table.add_column("Speakers", style="magenta")
    table.add_column("File", style="dim")

    for entry in files:
        f, short_id, rec = entry.path, entry.filename_id, entry.recording

        fm = read_frontmatter_head(f) or {}
        speakers = ", ".join(fm.get("speakers") or [])

        if rec:
            dur = rec.duration.total_seconds()
            if dur >= 3600:
                hours = int(dur // 3600)
                mins = int((dur % 3600) // 60)
                dur_str = f"{hours}h {mins:02d}m"
            elif dur >= 60:
                mins = int(dur // 60)
                secs = int(dur % 60)
                dur_str = f"{mins}m {secs:02d}s"
            else:
                dur_str = f"{int(dur)}s"
            date_str = rec.date.astimezone().strftime("%Y-%m-%d %H:%M") if rec.date else ""
            title = rec.title
        else:
            dur_str = ""
            date_str = ""
            title = ""

        table.add_row(short_id, date_str, dur_str, title, speakers, f.name)

    console = Console()
    console.print(table)
    console.print(f"\n[dim]{len(files)} transcript(s)[/dim]")


def _do_transcribe(rec: Recording, console: Console, *, progress=None) -> None:
    import time

    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

    from diane.transcribe import STEPS, format_diarized_transcript, transcribe_recording

    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    output_path = canonical_transcript_path(rec, TRANSCRIPTS_DIR)

    # Use provided progress (from transcribe-all) or create a standalone one
    own_progress = progress is None
    if own_progress:
        progress = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        )
        progress.start()

    task = progress.add_task(f"  {rec.title}", total=len(STEPS))

    def on_step(step: int, total: int) -> None:
        progress.update(task, completed=step, description=f"  {STEPS[step]}")

    start = time.monotonic()
    try:
        result = transcribe_recording(rec, on_step=on_step)
        elapsed = time.monotonic() - start
        progress.update(task, completed=len(STEPS), description="  Done")
    finally:
        progress.remove_task(task)
        if own_progress:
            progress.stop()

    document = new_identity(
        output_path,
        frontmatter_from_recording(rec),
        format_diarized_transcript(result.segments),
        result.clusters,
    )
    document.save()

    audio_dur = rec.duration.total_seconds()
    speedup = audio_dur / elapsed if elapsed > 0 else 0
    console.print(
        f"[green]Wrote[/green] {output_path.name}"
        f" [dim]({elapsed:.1f}s for {audio_dur:.0f}s audio, {speedup:.1f}x realtime)[/dim]"
    )
    if document.matches:
        named = ", ".join(f"{name} ({score:.2f})" for name, score in document.matches.values())
        console.print(f"  [dim]Auto-named from voiceprints: {named}[/dim]")


transcribe_app = App(name="transcribe", help="Transcribe recordings.")
app.command(transcribe_app)


@transcribe_app.default
def transcribe(recording_id: str, *, dry_run: bool = False) -> None:
    """Transcribe a recording with speaker diarization."""
    console = Console(stderr=True)

    try:
        rec = find_recording(recording_id)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    audio_path = RECORDINGS_DIR / rec.path
    if not audio_path.exists():
        console.print(f"[red]Error:[/red] Audio file not found: {audio_path}")
        sys.exit(1)

    output_path = canonical_transcript_path(rec, TRANSCRIPTS_DIR)
    if output_path.exists():
        console.print(f"[yellow]Transcript already exists:[/yellow] {output_path}")
        return

    if dry_run:
        console.print(f"Would transcribe [cyan]{rec.title}[/cyan] ({rec.short_id}) -> {output_path.name}")
        return

    _do_transcribe(rec, console)


@transcribe_app.command
def all(*, dry_run: bool = False) -> None:
    """Transcribe discovered recordings without transcripts. Preview with --dry-run first."""
    console = Console(stderr=True)
    pending = pending_recordings(TRANSCRIPTS_DIR)

    if not pending:
        console.print("[dim]All recordings are already transcribed.[/dim]")
        return

    console.print(f"[bold]{len(pending)}[/bold] recording(s) to transcribe")

    if dry_run:
        for r in pending:
            dur = r.duration.total_seconds()
            if dur >= 3600:
                dur_str = f"{int(dur // 3600)}h {int((dur % 3600) // 60):02d}m"
            elif dur >= 60:
                dur_str = f"{int(dur // 60)}m {int(dur % 60):02d}s"
            else:
                dur_str = f"{int(dur)}s"
            console.print(f"  {r.short_id}  {dur_str:>9}  {r.title}")
        return

    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

    total_duration = sum(r.duration.total_seconds() for r in pending)
    completed_duration = 0.0

    progress = Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        console=console,
    )

    failed: list[tuple[Recording, str]] = []

    with progress:
        overall = progress.add_task("Overall", total=total_duration)

        for i, r in enumerate(pending, 1):
            progress.update(overall, description=f"[{i}/{len(pending)}] {r.title}")
            try:
                _do_transcribe(r, console, progress=progress)
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]Failed[/red] {r.short_id} {r.title}: {e}")
                failed.append((r, str(e)))
            completed_duration += r.duration.total_seconds()
            progress.update(overall, completed=completed_duration)

    if failed:
        console.print(f"\n[red bold]{len(failed)} recording(s) failed:[/red bold]")
        for r, err in failed:
            console.print(f"  {r.short_id}  {r.title}: {err}")


# --- Speakers ---

speakers_app = App(name="speakers", help="Assign speaker names to transcripts.")
app.command(speakers_app)


def _process_speakers(
    document: IdentityDocument,
    rec: Recording,
    console: Console,
    *,
    on_speaker: object = None,
) -> bool:
    """Prompt for pending anonymous labels; the identity owner stages and saves answers."""
    from rich.rule import Rule

    from diane.speakers import (
        _is_absolute_body,
        extract_speaker_segments,
        first_excerpt,
        get_snippets,
        prompt_with_snippets,
    )

    body = document.body
    targets = document.manual_targets
    if not targets:
        console.print(f"  [dim]Skipping {document.path.name} — no unnamed speakers[/dim]")
        return False

    audio_path = RECORDINGS_DIR / rec.path
    if not audio_path.exists():
        console.print(f"  [red]Audio not found:[/red] {audio_path}")
        return False

    rec_start_seconds = None
    if _is_absolute_body(body):
        rd = rec.date.astimezone()
        rec_start_seconds = rd.hour * 3600 + rd.minute * 60 + rd.second + rd.microsecond / 1e6

    console.print(Rule(f"[bold]{document.path.name}[/bold]"))
    console.print(f"  [bold]{len(targets)}[/bold] speaker(s) found\n")
    console.print("  [dim]Enter skips this speaker; it remains available next time.[/dim]")
    answers: dict[str, str] = {}
    for i, target in enumerate(targets, 1):
        segments = extract_speaker_segments(body, target.label, rec_start_seconds=rec_start_seconds)
        console.print(f"  [bold]Speaker {i}/{len(targets)}:[/bold] [cyan]{target.label}[/cyan]")
        console.print(f'  [dim]"{first_excerpt(body, target.label)}"[/dim]')
        if target.shared:
            console.print("  [yellow]Playback combines turns with this rendered label.[/yellow]")
        answers[target.key] = prompt_with_snippets(
            target.label, get_snippets(segments), audio_path, console,
            body=body, rec_start_seconds=rec_start_seconds,
        )
        if on_speaker:
            on_speaker()
        console.print()

    for rejected in document.apply_manual(answers):
        console.print(f"  [yellow]{rejected.label}: {rejected.reason}[/yellow]")
    if document.save():
        console.print(f"\n  [green]Wrote[/green] {document.path.name}\n")
        return True
    console.print("  [dim]No identity changes saved.[/dim]")
    return False


@speakers_app.default
def speakers_cmd(*recording_ids: str) -> None:
    """Assign speaker names to transcripts by ID."""
    console = Console(stderr=True)

    if not recording_ids:
        console.print("[red]Error:[/red] Provide at least one recording ID.")
        sys.exit(1)

    targets: list[tuple[Path, Recording]] = []
    for rid in recording_ids:
        try:
            entry = find_recording_transcript(rid, TRANSCRIPTS_DIR)
            targets.append((entry.path, entry.recording))
        except (ValueError, FileNotFoundError) as e:
            console.print(f"[yellow]Warning:[/yellow] {e}")

    if not targets:
        console.print("[dim]No transcripts to process.[/dim]")
        return

    for idx, (path, rec) in enumerate(targets, 1):
        if len(targets) > 1:
            console.print(f"\n[bold dim][{idx}/{len(targets)}][/bold dim]")
        _process_speakers(load_identity(path, rec), rec, console)


@speakers_app.command(name="audit")
def speakers_audit(*recording_ids: str) -> None:
    """Review speakers named automatically from voiceprints.

    Plays each auto-named speaker's audio. Press Enter to confirm the name (it then
    feeds that person's voiceprint), or type the right name to replace it.
    """
    console = Console(stderr=True)
    wanted = {rid.lower() for rid in recording_ids}
    entries = list_transcripts(TRANSCRIPTS_DIR, include_recordings=True)
    targets = []
    for entry in sorted(entries, key=lambda t: t.path, reverse=True):
        path, short_id = entry.path, entry.filename_id
        if wanted and not any(short_id.startswith(w) for w in wanted):
            continue
        document = load_identity(path, entry.recording)
        if document.audit_targets:
            targets.append((document, entry.recording))

    if not targets:
        console.print("[dim]No auto-named speakers to review.[/dim]")
        return

    for document, rec in targets:
        if rec is None:
            console.print(f"  [yellow]Skipping {document.path.name}: recording not found[/yellow]")
            continue
        _audit_transcript(document, rec, console)


def _audit_transcript(document: IdentityDocument, rec: Recording, console: Console) -> None:
    """Collect explicit review answers without implementing identity transitions."""
    from diane.speakers import (
        _is_absolute_body,
        extract_speaker_segments,
        first_excerpt,
        get_snippets,
        prompt_with_snippets,
    )

    audio_path = RECORDINGS_DIR / rec.path
    body = document.body
    rec_start_seconds = None
    if _is_absolute_body(body):
        rd = rec.date.astimezone()
        rec_start_seconds = rd.hour * 3600 + rd.minute * 60 + rd.second + rd.microsecond / 1e6

    console.print(f"\n[bold]{document.path.name}[/bold]")
    answers: dict[str, str] = {}
    for target in document.audit_targets:
        segments = extract_speaker_segments(
            body, target.label, rec_start_seconds=rec_start_seconds,
        )
        score = f"{target.score:.2f}" if target.score is not None else "unknown"
        console.print(
            f"  [cyan]{target.label}[/cyan] [dim](voiceprint match {score},"
            f" {target.speech_s / 60:.1f} min)[/dim]"
        )
        console.print(f'  [dim]"{first_excerpt(body, target.label)}"[/dim]')
        if target.shared:
            console.print(
                "  [yellow]Playback and excerpts combine turns with this rendered label;"
                " they do not isolate this cluster.[/yellow]"
            )
        console.print("  [dim]Enter confirms; type a name to replace it.[/dim]")
        answers[target.key] = prompt_with_snippets(
            target.label, get_snippets(segments), audio_path, console,
            body=body, rec_start_seconds=rec_start_seconds,
        )

    for rejected in document.apply_audit(answers):
        console.print(f"  [yellow]{rejected.label}: {rejected.reason}[/yellow]")
    if document.save():
        console.print(f"  [green]Saved[/green] {document.path.name}")
    else:
        console.print("  [dim]No identity changes saved.[/dim]")


@speakers_app.command(name="all")
def speakers_all() -> None:
    """Assign names to remaining anonymous speakers, including previously skipped ones."""
    console = Console(stderr=True)

    if not TRANSCRIPTS_DIR.exists():
        console.print("[dim]No transcripts yet.[/dim]")
        return

    files = sorted(
        list_transcripts(TRANSCRIPTS_DIR, id_policy="delimited-only", include_recordings=True),
        key=lambda t: t.path,
        reverse=True,
    )

    targets: list[tuple[IdentityDocument, Recording, int]] = []
    for entry in files:
        rec = entry.recording
        if rec is None:
            continue
        document = load_identity(entry.path, rec)
        n_speakers = len(document.manual_targets)
        if n_speakers:
            targets.append((document, rec, n_speakers))

    if not targets:
        console.print("[dim]All transcripts already have speakers assigned.[/dim]")
        return

    total_speakers = sum(n for _, _, n in targets)
    completed_speakers = 0

    def _print_progress(title: str = "") -> None:
        bar_width = 20
        frac = completed_speakers / total_speakers if total_speakers else 1
        filled = int(bar_width * frac)
        bar = "█" * filled + "░" * (bar_width - filled)
        label = f" {title}" if title else ""
        console.print(
            f"\n[bold]{bar} {completed_speakers}/{total_speakers} speakers{label}[/bold]"
        )

    _print_progress()

    for idx, (document, rec, _n) in enumerate(targets, 1):

        def on_speaker() -> None:
            nonlocal completed_speakers
            completed_speakers += 1

        _process_speakers(
            document, rec, console, on_speaker=on_speaker,
        )
        _print_progress(f"[{idx}/{len(targets)}] done")


# --- Transcript operations ---


@transcript_app.command
def path(recording_id: str) -> None:
    """Print the absolute path of a transcript by ID."""
    console = Console(stderr=True)
    try:
        transcript_path = find_transcript(recording_id, TRANSCRIPTS_DIR).path
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    print(transcript_path.resolve())


absolute_app = App(name="absolute", help="Convert relative timestamps to absolute ISO timestamps.")
transcript_app.command(absolute_app)


def _do_absolute(path: Path, rec: Recording, console: Console) -> bool:
    """Convert one transcript to absolute timestamps. Returns True if converted."""
    from datetime import datetime

    from diane.timestamps import is_absolute, to_absolute

    fm, body = ensure_frontmatter(path, rec)

    if is_absolute(fm):
        return False

    # Always use recording birthtime (second precision) for conversion
    recorded = rec.date.astimezone()
    new_body = to_absolute(body, recorded)
    fm["timestamps"] = "absolute"
    fm["recorded"] = recorded.isoformat()
    output = build_output(fm, new_body)
    path.write_text(output)
    console.print(f"[green]Converted[/green] {path.name} (start: {recorded.isoformat()})")
    return True


def _fixup_absolute(path: Path, rec: Recording, console: Console) -> bool:
    """Fix an already-absolute transcript whose recorded time lacks second precision.

    Compares frontmatter recorded time with actual file birthtime.
    If there's a delta, shifts all timestamps and updates frontmatter.
    Returns True if fixed.
    """
    from datetime import datetime, timedelta

    from diane.timestamps import shift_timestamps

    fm, body = ensure_frontmatter(path, rec)

    recorded_str = fm.get("recorded")
    if not recorded_str:
        # No recorded time in frontmatter — add it from birthtime, but can't fix timestamps
        fm["recorded"] = rec.date.astimezone().isoformat()
        output = build_output(fm, body)
        path.write_text(output)
        console.print(f"[yellow]Added[/yellow] recorded time to {path.name}")
        return True

    old_recorded = datetime.fromisoformat(recorded_str)
    new_recorded = rec.date.astimezone()

    # Compare truncated to minute — if they match at minute level but differ
    # at second level, we have imprecise data that needs fixing
    old_minute = old_recorded.replace(second=0, microsecond=0)
    new_minute = new_recorded.replace(second=0, microsecond=0)
    if old_minute != new_minute:
        # Different minute entirely — don't auto-fix, could be a different issue
        return False

    delta_s = (new_recorded - old_recorded).total_seconds()
    if abs(delta_s) < 1.0:
        return False  # already accurate

    new_body = shift_timestamps(body, timedelta(seconds=delta_s))
    fm["recorded"] = new_recorded.isoformat()
    output = build_output(fm, new_body)
    path.write_text(output)
    console.print(
        f"[yellow]Fixed[/yellow] {path.name} (shifted {delta_s:+.0f}s,"
        f" {old_recorded.isoformat()} → {new_recorded.isoformat()})"
    )
    return True


@absolute_app.default
def absolute(recording_id: str) -> None:
    """Convert a transcript's timestamps to absolute."""
    console = Console(stderr=True)

    try:
        entry = find_recording_transcript(recording_id, TRANSCRIPTS_DIR)
        path, rec = entry.path, entry.recording
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    if not _do_absolute(path, rec, console):
        console.print(f"[dim]Timestamps already absolute in {path.name}[/dim]")


@absolute_app.command(name="all")
def absolute_all() -> None:
    """Convert all transcripts to absolute timestamps, and fix imprecise existing ones."""
    from diane.frontmatter import parse_frontmatter
    from diane.timestamps import is_absolute

    console = Console(stderr=True)

    if not TRANSCRIPTS_DIR.exists():
        console.print("[dim]No transcripts yet.[/dim]")
        return

    files = sorted(
        list_transcripts(TRANSCRIPTS_DIR, id_policy="delimited-only", include_recordings=True),
        key=lambda t: t.path,
    )

    converted = 0
    fixed = 0
    skipped = 0
    for entry in files:
        f, rec = entry.path, entry.recording
        if rec is None:
            continue
        text = f.read_text()
        fm, _ = parse_frontmatter(text)
        if is_absolute(fm or {}):
            if _fixup_absolute(f, rec, console):
                fixed += 1
            else:
                skipped += 1
            continue
        if _do_absolute(f, rec, console):
            converted += 1

    console.print(f"\n[dim]{converted} converted, {fixed} fixed, {skipped} already accurate[/dim]")


@app.command
def slice(*, start: str, end: str) -> None:
    """Print time-range lines from workspace-wide .txt files with absolute-timestamp metadata."""
    from datetime import datetime

    from diane.frontmatter import parse_frontmatter
    from diane.timestamps import (
        extract_timestamp,
        is_absolute,
        slice_transcript,
        timestamp_line_content,
    )

    console = Console(stderr=True)

    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)

    # Collect (datetime, line) pairs from all absolute-timestamped .txt files
    entries: list[tuple[datetime, str]] = []

    for txt_file in PROJECT_ROOT.rglob("*.txt"):
        # Skip hidden dirs, .venv, etc.
        if any(part.startswith(".") or part == ".venv" for part in txt_file.parts):
            continue

        text = txt_file.read_text()
        fm, body = parse_frontmatter(text)
        if not is_absolute(fm or {}):
            continue

        result = slice_transcript(body, start_dt, end_dt)
        if not result:
            continue

        # Label each line with the file's parent directory.
        source = txt_file.parent.name

        for line in result.splitlines():
            if not line.strip():
                continue
            ts = extract_timestamp(line)
            if ts and timestamp_line_content(line):
                # Inject [source] after the timestamp: [ts] content -> [ts] [source] content
                tagged = line.replace("]", f"] [{source}]", 1)
                entries.append((ts, tagged))

    if not entries:
        console.print("[dim]No lines in the given time range.[/dim]")
        return

    entries.sort(key=lambda e: e[0])
    for _, line in entries:
        print(line)
        print()  # blank line separator
