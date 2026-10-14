"""Read audio.wav into an RMS loudness envelope for the audio_energy feature.

Lives in ingest, not scoring, so the scorer never touches a file: the CLI
computes this once and hands the array in.
"""

import wave
from pathlib import Path

import numpy as np


def rms_envelope(wav_path: Path, hop_ms: int = 100) -> np.ndarray:
    """RMS per hop_ms frame, on a 0-1 scale. Expects 16-bit mono (what ingest writes)."""
    with wave.open(str(wav_path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f"{wav_path}: expected 16-bit mono, run `clip ingest --force`")
        rate = w.getframerate()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

    frame = rate * hop_ms // 1000
    usable = len(samples) // frame * frame
    frames = samples[:usable].reshape(-1, frame).astype(np.float32) / 32_768
    return np.sqrt(np.mean(frames**2, axis=1))
