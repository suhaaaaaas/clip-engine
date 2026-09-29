"""Join ASR segments to diarization turns.

Whisper's segment boundaries and pyannote's turn boundaries never line up, so
each segment takes the speaker it overlaps most. Both lists are sorted, so walk
them together rather than comparing every pair.

Assumes both lists are sorted by start time and diarization turns don't
overlap each other, which holds for pyannote's output.
"""

from dataclasses import replace

from ..models import Segment, SpeakerTurn


def assign_speakers(segments: list[Segment], turns: list[SpeakerTurn]) -> list[Segment]:
    result = []
    i = 0  # index of the first turn that could still be relevant
    for seg in segments:
        while i < len(turns) and turns[i].end_ms <= seg.start_ms:
            i += 1

        best_speaker = None
        best_overlap = 0
        j = i
        while j < len(turns) and turns[j].start_ms < seg.end_ms:
            overlap = min(seg.end_ms, turns[j].end_ms) - max(seg.start_ms, turns[j].start_ms)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turns[j].speaker
            j += 1

        result.append(seg if best_speaker is None else replace(seg, speaker=best_speaker))
    return result
