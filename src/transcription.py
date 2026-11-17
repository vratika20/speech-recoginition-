"""Language settings and calls to the Hugging Face Whisper pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

LANGUAGES = {
    "Auto-detect": None,
    "English": "english",
    "Hindi": "hindi",
    "Tamil": "tamil",
    "Spanish": "spanish",
    "French": "french",
    "German": "german",
    "Japanese": "japanese",
}


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    task: str
    language_label: str


def transcribe(
    asr_pipeline: Any,
    waveform: np.ndarray,
    sample_rate: int,
    language_label: str = "Auto-detect",
    task: str = "transcribe",
) -> TranscriptionResult:
    """Run Whisper transcription or speech-to-English translation.

    ``task='transcribe'`` preserves the spoken language. ``task='translate'``
    asks Whisper to produce English text.
    """
    if language_label not in LANGUAGES:
        raise ValueError(f"Unknown language selection: {language_label}")
    if task not in {"transcribe", "translate"}:
        raise ValueError("task must be 'transcribe' or 'translate'.")

    generate_kwargs: dict[str, str] = {"task": task}
    language = LANGUAGES[language_label]
    if language:
        generate_kwargs["language"] = language
    output = asr_pipeline(
        {"raw": waveform, "sampling_rate": sample_rate},
        generate_kwargs=generate_kwargs,
        return_timestamps=False,
    )
    text = str(output.get("text", "")).strip()
    if not text:
        raise RuntimeError("Whisper returned no text. Try clearer or longer speech.")
    return TranscriptionResult(text=text, task=task, language_label=language_label)
