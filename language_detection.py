"""Speech language identification using Meta MMS LID (126 languages)."""

from __future__ import annotations

import torch
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

from languages import LID_TO_SEAMLESS

LID_MODEL_ID = "facebook/mms-lid-126"


def load_lid_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    extractor = AutoFeatureExtractor.from_pretrained(LID_MODEL_ID)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(LID_MODEL_ID, torch_dtype=dtype)
    model.to(device).eval()
    return extractor, model, device


def detect_language(extractor, model, device, waveform) -> dict:
    """Return MMS top class and its raw softmax score (not a calibrated confidence)."""
    # Long recordings are sampled from beginning, middle and end to avoid feeding
    # extremely long files into the classifier; aggregate logits across windows.
    window = 20 * 16_000
    if waveform.size > window:
        starts = [0, max(0, (waveform.size - window) // 2), waveform.size - window]
        segments = [waveform[start : start + window] for start in starts]
    else:
        segments = [waveform]
    scores = []
    for segment in segments:
        batch = extractor(segment, sampling_rate=16_000, return_tensors="pt", padding=True)
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        with torch.inference_mode():
            scores.append(model(**batch).logits.float().softmax(dim=-1)[0])
    probabilities = torch.stack(scores).mean(dim=0)
    index = int(probabilities.argmax().item())
    label = model.config.id2label[index]
    return {"lid_code": label, "seamless_code": LID_TO_SEAMLESS.get(label), "score": float(probabilities[index].item())}
