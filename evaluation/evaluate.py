"""Evaluate Whisper transcripts against references from a CSV manifest."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from jiwer import wer

from src.audio_processing import load_audio
from src.model import get_device, load_asr_pipeline
from src.transcription import transcribe


def evaluate(manifest_path: Path) -> int:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    needed = {"audio_path", "reference", "language"}
    if not needed.issubset(rows[0] if rows else {}):
        raise ValueError("CSV must contain: audio_path,reference,language")
    if not rows:
        print("No rows to evaluate. Add real audio paths and references to the CSV manifest.")
        return 0

    print(f"Loading Whisper Small on {get_device()[2]}…")
    asr = load_asr_pipeline()
    grouped: dict[str, list[float]] = {}
    for number, row in enumerate(rows, 1):
        audio_path = (manifest_path.parent / row["audio_path"]).resolve()
        if not audio_path.exists():
            print(f"[{number}] SKIPPED: audio not found: {audio_path}")
            continue
        started = time.perf_counter()
        waveform, rate, _ = load_audio(audio_path)
        result = transcribe(asr, waveform, rate, row["language"], "transcribe")
        seconds = time.perf_counter() - started
        score = wer(row["reference"], result.text)
        grouped.setdefault(row["language"], []).append(score)
        print(f"[{number}] {row['language']} | WER={score:.3f} | {seconds:.1f}s")
        print(f"  Reference: {row['reference']}\n  Predicted: {result.text}")
    for language, scores in grouped.items():
        print(f"{language}: mean WER={sum(scores) / len(scores):.3f} across {len(scores)} file(s)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate WER from an evaluation CSV.")
    parser.add_argument("--manifest", default="evaluation/test_data.csv", type=Path)
    args = parser.parse_args()
    raise SystemExit(evaluate(args.manifest))
