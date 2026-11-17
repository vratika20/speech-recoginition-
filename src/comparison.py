"""Utilities for a fair, per-file Whisper vs. SeamlessM4T comparison."""

from __future__ import annotations

import csv
import html
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from io import StringIO
from typing import Any, Iterable

import numpy as np
from src.transcription import transcribe


@dataclass(frozen=True)
class ModelRun:
    """One model's output and inference-only wall-clock duration."""

    model_name: str
    transcript: str
    seconds: float
    wer_score: float | None


MIN_LANGUAGE_SAMPLES = 2


def calculate_wer(reference: str, prediction: str) -> float:
    """Calculate WER when the optional evaluation dependency is installed."""
    try:
        from jiwer import wer
    except ImportError as error:
        raise RuntimeError("WER needs Jiwer. Install project dependencies with: pip install -r requirements.txt") from error
    return float(wer(reference, prediction))


def _run_seamless(asr_pipeline: Any, waveform: np.ndarray, sample_rate: int) -> str:
    """Run the Seamless ASR pipeline with exactly the same decoded waveform."""
    output = asr_pipeline(
        {"raw": waveform, "sampling_rate": sample_rate}, return_timestamps=False
    )
    text = str(output.get("text", "")).strip()
    if not text:
        raise RuntimeError("SeamlessM4T returned no text. Try clearer or longer speech.")
    return text


def compare_transcripts(
    whisper_pipeline: Any,
    seamless_pipeline: Any,
    waveform: np.ndarray,
    sample_rate: int,
    language_label: str,
    reference: str = "",
) -> tuple[ModelRun, ModelRun]:
    """Transcribe one waveform with both cached models and time inference only."""
    started = time.perf_counter()
    whisper_text = transcribe(
        whisper_pipeline, waveform, sample_rate, language_label, "transcribe"
    ).text
    whisper_seconds = time.perf_counter() - started

    started = time.perf_counter()
    seamless_text = _run_seamless(seamless_pipeline, waveform, sample_rate)
    seamless_seconds = time.perf_counter() - started

    normalized_reference = reference.strip()
    return (
        ModelRun(
            "OpenAI Whisper Small",
            whisper_text,
            whisper_seconds,
            calculate_wer(normalized_reference, whisper_text) if normalized_reference else None,
        ),
        ModelRun(
            "Meta SeamlessM4T v2 Large",
            seamless_text,
            seamless_seconds,
            calculate_wer(normalized_reference, seamless_text) if normalized_reference else None,
        ),
    )


def highlighted_difference_html(left: str, right: str) -> str:
    """Return a word-level visual diff; red was Whisper-only, green Seamless-only."""
    left_words, right_words = left.split(), right.split()
    pieces: list[str] = []
    matcher = SequenceMatcher(a=left_words, b=right_words)
    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == "equal":
            pieces.append(html.escape(" ".join(left_words[a0:a1])))
        if opcode in {"delete", "replace"}:
            pieces.append(
                f'<span style="background:#ffe1e1;text-decoration:line-through;">'
                f'{html.escape(" ".join(left_words[a0:a1]))}</span>'
            )
        if opcode in {"insert", "replace"}:
            pieces.append(
                f'<span style="background:#ddf7e5;">{html.escape(" ".join(right_words[b0:b1]))}</span>'
            )
    return " ".join(piece for piece in pieces if piece)


def transcript_difference_rows(left: str, right: str) -> list[dict[str, str]]:
    """Describe matching, added, omitted, and replaced word spans explicitly."""
    left_words, right_words = left.split(), right.split()
    rows: list[dict[str, str]] = []
    for opcode, a0, a1, b0, b1 in SequenceMatcher(a=left_words, b=right_words).get_opcodes():
        whisper_words = " ".join(left_words[a0:a1])
        seamless_words = " ".join(right_words[b0:b1])
        label = {
            "equal": "Match",
            "delete": "Only Whisper (omitted by SeamlessM4T)",
            "insert": "Only SeamlessM4T (added)",
            "replace": "Replacement",
        }[opcode]
        rows.append({"difference": label, "whisper_words": whisper_words, "seamless_words": seamless_words})
    return rows


def language_wise_summary(history: Iterable[dict[str, Any]]) -> list[dict[str, float | int | str]]:
    """Average only real referenced runs, and only after enough samples exist."""
    by_language: dict[str, list[dict[str, Any]]] = {}
    for record in history:
        language = str(record.get("language", ""))
        if language in {"English", "Hindi", "Tamil"} and record.get("whisper_wer") is not None:
            by_language.setdefault(language, []).append(record)

    summaries: list[dict[str, float | int | str]] = []
    for language in ("English", "Hindi", "Tamil"):
        records = by_language.get(language, [])
        if len(records) < MIN_LANGUAGE_SAMPLES:
            continue
        count = len(records)
        summaries.append(
            {
                "language": language,
                "samples": count,
                "whisper_wer": sum(float(row["whisper_wer"]) for row in records) / count,
                "seamless_wer": sum(float(row["seamless_wer"]) for row in records) / count,
                "whisper_seconds": sum(float(row["whisper_seconds"]) for row in records) / count,
                "seamless_seconds": sum(float(row["seamless_seconds"]) for row in records) / count,
            }
        )
    return summaries


def metric_winner(whisper_value: float, seamless_value: float) -> str:
    """Return the lower-valued model, or a tie, for time/WER metrics."""
    if whisper_value == seamless_value:
        return "Tie"
    return "Whisper Small" if whisper_value < seamless_value else "SeamlessM4T v2"


def comparison_csv(
    whisper: ModelRun,
    seamless: ModelRun,
    reference: str,
    language: str = "",
    differences: Iterable[dict[str, str]] = (),
    language_results: Iterable[dict[str, float | int | str]] = (),
) -> str:
    """Create an in-memory CSV report; no uploaded content is persisted."""
    buffer = StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["record_type", "language", "model", "processing_seconds", "wer", "reference", "transcript", "difference", "whisper_words", "seamless_words", "samples", "whisper_wer", "seamless_wer", "whisper_seconds", "seamless_seconds"],
    )
    writer.writeheader()
    for result in (whisper, seamless):
        writer.writerow(
            {
                "record_type": "per_file_result",
                "language": language,
                "model": result.model_name,
                "processing_seconds": f"{result.seconds:.3f}",
                "wer": "" if result.wer_score is None else f"{result.wer_score:.4f}",
                "reference": reference.strip(),
                "transcript": result.transcript,
            }
        )
    for difference in differences:
        writer.writerow({"record_type": "transcript_difference", "language": language, **difference})
    for result in language_results:
        writer.writerow({"record_type": "language_average", **result})
    return buffer.getvalue()


def comparison_text(
    whisper: ModelRun,
    seamless: ModelRun,
    reference: str,
    language: str = "",
    differences: Iterable[dict[str, str]] = (),
    language_results: Iterable[dict[str, float | int | str]] = (),
) -> str:
    """Create a human-readable in-memory TXT report."""
    lines = ["ASR Model Comparison", f"Selected language: {language or '[not selected]'}", f"Reference transcript: {reference.strip() or '[not provided]'}", ""]
    for result in (whisper, seamless):
        wer_text = "not calculated (reference required)" if result.wer_score is None else f"{result.wer_score:.4f}"
        lines.extend(
            [result.model_name, f"Processing time: {result.seconds:.3f} seconds", f"WER: {wer_text}", f"Transcript: {result.transcript}", ""]
        )
    difference_list = list(differences)
    if difference_list:
        lines.extend(["Transcript differences:"])
        lines.extend(f"- {row['difference']}: Whisper=[{row['whisper_words']}] | SeamlessM4T=[{row['seamless_words']}]" for row in difference_list)
    summary_list = list(language_results)
    if summary_list:
        lines.extend(["", "Language-wise averages (only languages with sufficient referenced samples):"])
        lines.extend(f"- {row['language']} ({row['samples']} samples): Whisper WER={row['whisper_wer']:.4f}, Seamless WER={row['seamless_wer']:.4f}, Whisper time={row['whisper_seconds']:.3f}s, Seamless time={row['seamless_seconds']:.3f}s" for row in summary_list)
    return "\n".join(lines)
