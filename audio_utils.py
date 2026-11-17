"""Audio decoding, validation, mono downmixing and 16 kHz resampling."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".webm"}


class AudioError(ValueError):
    """Raised when uploaded data is not usable audio."""


def load_audio(data: bytes, filename: str) -> tuple[np.ndarray, float]:
    """Return mono float32 audio at 16 kHz and original duration in seconds."""
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioError("Unsupported file type. Upload WAV, MP3, M4A, FLAC, OGG, AAC, or WEBM.")
    if not data:
        raise AudioError("The uploaded file is empty.")
    try:
        try:
            samples, rate = sf.read(BytesIO(data), dtype="float32", always_2d=True)
        except Exception:
            samples, rate = librosa.load(BytesIO(data), sr=None, mono=False, dtype=np.float32)
            if samples.ndim == 1:
                samples = samples[:, None]
            else:  # librosa channel-first
                samples = samples.T
    except Exception as exc:
        raise AudioError("Could not decode this audio. For MP3/M4A, install FFmpeg and try again.") from exc

    if rate <= 0 or samples.size == 0:
        raise AudioError("The audio contains no samples or has an invalid sample rate.")
    duration = float(len(samples) / rate)
    if duration > 3600:
        raise AudioError("Please use an audio file shorter than one hour.")
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if not np.any(mono):
        raise AudioError("The audio appears to be silent.")
    if rate != SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=int(rate), target_sr=SAMPLE_RATE)
    if not np.isfinite(mono).all() or mono.size == 0:
        raise AudioError("Audio preprocessing failed; please try another file.")
    return mono.astype(np.float32), duration
