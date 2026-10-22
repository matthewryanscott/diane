"""End-to-end check: scribe's own diarize() + run_transcription() on AMI.

Writes the ASR and diarizer contract files under the name `scribe-e2e` so the
combo scorer can confirm scribe reproduces the experiment's cpWER:

    uv run python run_scribe_e2e.py [--variants sdm,ihm] [--meetings all]
    uv run python score.py combo --asr scribe-e2e --diar scribe-e2e --merge word
"""

import argparse
import sys
import time

from common import asr_run_dir, audio_path, diar_run_dir, meetings_for, wav_duration, write_json

import scribe  # resolved via the sys.path entry common adds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", default="sdm,ihm")
    parser.add_argument("--meetings", default="all")
    args = parser.parse_args()

    for variant in args.variants.split(","):
        asr_dir = asr_run_dir("scribe-e2e", variant)
        diar_dir = diar_run_dir("scribe-e2e", variant)
        asr_dir.mkdir(parents=True, exist_ok=True)
        diar_dir.mkdir(parents=True, exist_ok=True)
        for meeting in meetings_for(args.meetings):
            wav = audio_path(variant, meeting)
            audio_s = round(wav_duration(wav), 2)

            t0 = time.perf_counter()
            diarization = scribe.diarize(wav)
            diar_s = round(time.perf_counter() - t0, 2)
            lines = [
                f"SPEAKER {meeting} 1 {s.start:.3f} {s.end - s.start:.3f} <NA> <NA> {s.speaker} <NA> <NA>"
                for s in diarization
            ]
            (diar_dir / f"{meeting}.rttm").write_text("\n".join(lines) + "\n")
            write_json(diar_dir / f"{meeting}.meta.json", {"model": "scribe diarize()", "runtime_s": diar_s, "audio_s": audio_s})

            t0 = time.perf_counter()
            segments = scribe.run_transcription(wav)
            asr_s = round(time.perf_counter() - t0, 2)
            sentences = [
                {
                    "text": s.text,
                    "start": s.start,
                    "end": s.end,
                    "words": [{"text": w.text, "start": w.start, "end": w.end} for w in s.words],
                }
                for s in segments
            ]
            write_json(asr_dir / f"{meeting}.json", {
                "words": [w for s in sentences for w in s["words"]],
                "sentences": sentences,
                "meta": {"model": "scribe run_transcription()", "runtime_s": asr_s, "audio_s": audio_s},
            })
            print(f"scribe-e2e/{variant}/{meeting}: diarize {diar_s}s, asr {asr_s}s, audio {audio_s}s", file=sys.stderr)


if __name__ == "__main__":
    main()
