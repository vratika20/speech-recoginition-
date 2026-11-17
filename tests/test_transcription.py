import unittest
from unittest.mock import Mock

import numpy as np

from src.transcription import transcribe
from src.comparison import (
    compare_transcripts,
    comparison_csv,
    language_wise_summary,
    transcript_difference_rows,
)


class TranscriptionTests(unittest.TestCase):
    def test_uses_transcription_task_and_selected_language(self):
        pipeline = Mock(return_value={"text": "hello world"})
        result = transcribe(pipeline, np.array([0.1, 0.2], dtype=np.float32), 16000, "Hindi")
        self.assertEqual(result.text, "hello world")
        self.assertEqual(pipeline.call_args.kwargs["generate_kwargs"], {"task": "transcribe", "language": "hindi"})

    def test_auto_detection_does_not_force_language(self):
        pipeline = Mock(return_value={"text": "வணக்கம்"})
        transcribe(pipeline, np.array([0.1], dtype=np.float32), 16000)
        self.assertEqual(pipeline.call_args.kwargs["generate_kwargs"], {"task": "transcribe"})

    def test_comparison_reports_inference_times_and_report(self):
        whisper = Mock(return_value={"text": "hello world"})
        seamless = Mock(return_value={"text": "hello there"})
        first, second = compare_transcripts(whisper, seamless, np.array([0.1], dtype=np.float32), 16000, "English")
        self.assertIsNone(first.wer_score)
        self.assertGreaterEqual(first.seconds, 0.0)
        self.assertIn("OpenAI Whisper Small", comparison_csv(first, second, ""))

    def test_language_summary_requires_two_referenced_samples(self):
        one_sample = [{"language": "English", "whisper_wer": 0.1, "seamless_wer": 0.2, "whisper_seconds": 1.0, "seamless_seconds": 2.0}]
        self.assertEqual(language_wise_summary(one_sample), [])
        summary = language_wise_summary(one_sample * 2)
        self.assertEqual(summary[0]["language"], "English")
        self.assertEqual(summary[0]["samples"], 2)
        self.assertEqual(summary[0]["whisper_wer"], 0.1)

    def test_difference_rows_identify_replacements(self):
        rows = transcript_difference_rows("hello bright world", "hello small world")
        self.assertEqual(rows[0]["difference"], "Match")
        self.assertTrue(any(row["difference"] == "Replacement" for row in rows))


if __name__ == "__main__":
    unittest.main()
