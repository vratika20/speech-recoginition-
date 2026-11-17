"""Whisper and SeamlessM4T model loading plus device selection."""

from __future__ import annotations

import torch
from transformers import pipeline

MODEL_ID = "openai/whisper-small"
SEAMLESS_MODEL_ID = "facebook/seamless-m4t-v2-large"


def get_device() -> tuple[int, torch.dtype, str]:
    """Return Transformers pipeline device, dtype, and friendly device label."""
    if torch.cuda.is_available():
        return 0, torch.float16, "CUDA GPU"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float32, "Apple Silicon GPU"
    return -1, torch.float32, "CPU"


def load_asr_pipeline():
    """Load the official Whisper Small automatic-speech-recognition pipeline."""
    device, dtype, _ = get_device()
    return pipeline(
        task="automatic-speech-recognition",
        model=MODEL_ID,
        device=device,
        torch_dtype=dtype,
    )


def load_seamless_asr_pipeline():
    """Load SeamlessM4T v2 once for automatic speech recognition.

    The Hugging Face ASR pipeline loads the speech-to-text task only.  It does
    not load a second Whisper instance or use SeamlessM4T's speech generator.
    """
    device, dtype, _ = get_device()
    return pipeline(
        task="automatic-speech-recognition",
        model=SEAMLESS_MODEL_ID,
        device=device,
        torch_dtype=dtype,
    )
