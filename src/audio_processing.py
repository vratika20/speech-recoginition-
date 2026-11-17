"""Safe, model-ready audio loading utilities."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import librosa
import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 16_000
SUPPORTED_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".webm"}


class AudioValidationError(ValueError):
    """Raised when an audio upload cannot be decoded for Whisper."""


def validate_filename(filename: str | None) -> None:
    """Validate an upload name before attempting potentially expensive decoding."""
    if not filename:
        raise AudioValidationError("Please choose an audio file first.")
    if Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise AudioValidationError(f"Unsupported format. Use one of: {supported}.")


def _read_audio(source: str | Path | bytes | BinaryIO) -> tuple[np.ndarray, int]:
    """Decode audio with SoundFile first, then Librosa (which can use audioread)."""
    try:
        if isinstance(source, bytes):
            return sf.read(BytesIO(source), always_2d=False, dtype="float32")
        return sf.read(source, always_2d=False, dtype="float32")
    except Exception as soundfile_error:
        try:
            # librosa accepts filesystem paths; bytes need a seekable stream.
            input_value = BytesIO(source) if isinstance(source, bytes) else source
            samples, sample_rate = librosa.load(input_value, sr=None, mono=False)
            return samples.astype(np.float32), int(sample_rate)
        except Exception as librosa_error:
            raise AudioValidationError(
                "Could not read this audio file. Try WAV, MP3, M4A, FLAC, OGG, or WEBM. "
                "For MP3/M4A on some computers, install FFmpeg."
            ) from librosa_error


def load_audio(source: str | Path | bytes | BinaryIO) -> tuple[np.ndarray, int, float]:
    """Decode, downmix, resample, and validate audio for the Whisper pipeline.

    Returns a one-dimensional float32 waveform, its sample rate (always 16 kHz),
    and duration in seconds before resampling.
    """
    samples, sample_rate = _read_audio(source)
    if sample_rate <= 0 or samples.size == 0:
        raise AudioValidationError("The audio file is empty or has an invalid sample rate.")

    duration_seconds = float(samples.shape[-1] / sample_rate)
    if duration_seconds > 60 * 60:
        raise AudioValidationError("Please use audio shorter than one hour.")

    # SoundFile returns (frames, channels); librosa multi-channel audio is
    # (channels, frames). Reduce whichever axis is the channel axis.
    if samples.ndim == 2:
        samples = samples.mean(axis=1 if samples.shape[0] >= samples.shape[1] else 0)
    if samples.ndim != 1:
        raise AudioValidationError("This file does not contain a usable audio stream.")

    if sample_rate != TARGET_SAMPLE_RATE:
        samples = librosa.resample(
            samples, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE
        )
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if not np.any(samples):
        raise AudioValidationError("The audio appears to be silent.")
    return samples, TARGET_SAMPLE_RATE, duration_seconds
