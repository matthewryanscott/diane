# Why the meeting workflow separates speech, speakers, and names

Diane is intended for recurring meetings with 2–6 people. That use case is a design target, not an accuracy guarantee.
The workflow keeps three questions separate: what was said, which speaker cluster said it, and which person that cluster represents.
Errors in one answer can affect the next. A familiar name in a transcript does not prove that the words or attribution are correct.

## Local inference still has network dependencies

Transcription, diarization, and voiceprint matching run locally. Setup is a separate concern.
Dependency installation and first-use model downloads can require network access.
Parakeet and Nemotron run through `uvx`, which prepares separate dependency environments.
Those environments isolate conflicting model dependencies. They do not make an uncached installation offline.

The evaluated default model downloads did not require an account. That observation does not cover every alternative model or future revision.
A successful online setup is not evidence that every later invocation will work without a network connection.
See [Scribe's requirements and model behavior](../scribe/README.md) for the supported execution environment.

Local execution limits where inference happens. It does not anonymize the source audio, transcripts, embeddings, or logs.
The output files need their own access and sharing decisions.

## Scribe processes audio; Diane maintains the meeting record

[Diane's transcription adapter](../diane/transcribe.py) passes the original recording path to Scribe with embeddings enabled.
[Scribe's public operation](../scribe/README.md#python-interface) owns the temporary audio and model execution:

1. Normalize the input to a temporary 16 kHz mono WAV.
2. Run Parakeet speech recognition alongside diarization, using Nemotron by default.
3. Compute CAM++ speaker embeddings from usable diarized speech windows.
4. Merge recognized words with speaker turns by their timestamps.
5. Remove temporary audio after the workers stop, including when an operation fails.

Diane then formats timestamped turns and manages names, metadata, transcripts, and voiceprint sidecars.
This boundary keeps audio preparation and worker lifetime in one place.
The source recording stays unchanged. `Recordings/` and its symlink targets must remain read-only.

Recognition and diarization provide different evidence. Recognition supplies text and word times. Diarization supplies speaker activity over time.
When word times exist, Scribe assigns each word to the speaker with the most overlap.
It uses timing fallbacks for gaps and smooths short speaker runs.
Without word times, it attributes the whole recognition segment.
These are attribution rules, not proof of who spoke.
A better diarizer cannot restore words that recognition omitted.

## A speaker cluster is not a person

Labels such as `SPEAKER_00` identify clusters within a recording, not verified people across meetings.
The diarizer can split one person into several clusters or mix people in one cluster.
Brief replies, overlapping speech, and inaccurate word times can also produce incorrect attribution.
Renaming a cluster changes its displayed name. It does not separate mixed voices or repair its words.

Nemotron does not supply the speaker embeddings used for naming.
Scribe uses senko's CAM++ embedding pipeline over Nemotron's turns instead.
It excludes overlapped or ambiguous embedding windows under its coverage rules.
A cluster can have no usable centroid, so some transcript speakers have no voiceprint to match.
See the [embedding and merge implementation](../scribe/scribe.py) for these rules.

## Confirmation controls enrollment

A voiceprint summarizes a name's enrolled cluster centroids. It is not a recording of that person's voice or a verified identity credential.
[Diane's identity owner](../diane/voiceprints.py) combines normalized centroids with equal cluster weight, not speech-duration weight.
It compares each new cluster independently against the enrolled names using cosine similarity.
The current threshold is **0.86**. A score of 0.86 does not mean an 86% probability of correct identity.
Several clusters can receive the same name. The matcher does not enforce one cluster per person.

Sidecars record how each name entered the workflow:

- **`manual`**: a user assigned or corrected the name.
- **`auto`**: matching proposed the name during transcription.
- **`confirmed`**: a user accepted the automatic name during audit.

Only named `manual` and `confirmed` clusters contribute to enrollment. Automatic proposals never enroll by themselves.
This prevents an unreviewed proposal from becoming enrollment evidence automatically.
A mistaken manual decision or confirmation can still affect future matches.
Enrollment reads sidecars in the transcript directory, excluding the new target itself.
It can include a sidecar even after its corresponding transcript has been removed.

The normal speaker-naming workflow addresses anonymous speakers. Audit addresses unconfirmed automatic names.
Confirmation retains the match score. A correction records a manual name and clears that score.
A transcript without a sidecar can still receive manual names, but typing a name does not create an embedding.

### Why some corrections are ambiguous

Several original clusters can share one displayed name. Their combined transcript turns do not identify each original cluster independently.
New sidecars therefore retain provenance: original-to-displayed labels and a fingerprint of the ordered label occurrences.
This evidence tracks attribution, not the truth of a person's identity.
Timestamp-only conversions and speech-text edits do not invalidate it.

Missing or stale provenance leaves attribution unknown. Sidecar cluster records alone cannot reconstruct which displayed turns belong to each cluster.
Audit can confirm an unchanged name, but rejects a correction when it cannot safely identify that cluster's turns.
A fresh transcription with provenance is safer than guessing how to split an ambiguous displayed group.

Transcript and sidecar saves are not one transaction. Diane writes the transcript first.
A later sidecar failure raises `PartialIdentityWriteError` and can leave the files inconsistent.
The workflow does not silently retry, roll back, or repair that state.

## Why these defaults are tradeoffs

The [tracked AMI evaluation](../eval/README.md) covers eight four-speaker meetings, about 3.9 hours per microphone variant.
Its distant-microphone and mixed-headset inputs test different conditions.
The reports describe historical model versions, settings, references, and scoring rules, not a universal model ranking.

The [Parakeet experiments](../eval/results/parakeet.md) explain the current 30-second windows with 5-second overlap.
Shorter windows reduced omissions and memory use against the historical 120/15-second baseline. They did not remove all dropped speech.
Diarization-derived recognition windows performed slightly better in that comparison, but made recognition wait for diarization.
Speech-only windows also lose anything the diarizer misses.
The current fixed windows preserve parallel recognition and diarization.

The [Nemotron comparison](../eval/results/nemotron3.md) found lower speaker confusion and better speaker-attributed word scores than senko under several AMI settings.
Other reference styles and scoring collars changed the ranking.
Nemotron remains the default diarizer, while senko remains an alternative and supplies CAM++ embeddings.
Historical runtime and memory measurements depend on hardware, model load, and machine contention. They are not turnaround guarantees.

Three evidence levels must stay distinct:

- **Intended use:** recurring meetings with 2–6 people.
- **Execution smoke:** the tutorial's [two-synthetic-voice walkthrough](tutorial.md#walkthrough-evidence) demonstrates execution, not real-meeting accuracy.
- **Measured evaluation:** the AMI reports measure specific recognition and diarization configurations against references.

Voiceprint matching is experimental. Its private evaluation helpers and reports are excluded from this publication candidate.
The public AMI recognition and diarization results do not establish cross-recording identity accuracy.
Synthetic tests likewise do not establish reliable unattended naming.
Important words, boundaries, and names still need review against the audio.

## Consent and sensitive outputs remain separate responsibilities

Obtain consent for the intended recording, transcription, and speaker-matching use before processing participants' voices.
Permission to record a meeting does not by itself establish permission for every later use of voiceprints or transcripts.

A `.voice.json` sidecar can contain voice centroids, speech durations, names, name sources, similarity scores, and attribution provenance.
Removing names from a transcript does not remove those data from its sidecar.
Treat recordings, transcripts, sidecars, embeddings, exports, and diagnostic logs as sensitive.
Ignored files are not anonymous files, and local processing does not make generated files safe to commit or publish.

The evaluation documentation's privacy gates also apply to derived artifacts, not just prose.
For the dependency and model inventory, source provenance, obligations, and unresolved release gates, see the [canonical licensing documentation](licensing/index.md).
That inventory does not supply participant consent or settle whether a particular recording or output may be shared.
