"""Cached SeamlessM4T v2 loading for speech recognition and text translation."""

from __future__ import annotations

import torch
import numpy as np
from transformers import AutoProcessor, SeamlessM4Tv2Model

MODEL_ID = "facebook/seamless-m4t-v2-large"


def device_and_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32


def load_model():
    """Load the unified pretrained model once for ASR and text-to-text tasks."""
    device, dtype = device_and_dtype()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = SeamlessM4Tv2Model.from_pretrained(MODEL_ID, torch_dtype=dtype)
    model.to(device).eval()
    return processor, model, device


def transcribe(
    processor,
    model,
    device,
    waveform,
    language_code: str,
    calculate_attention: bool = False,
):
    """Run speech-to-text, optionally returning decoder cross-attention weights."""
    inputs = processor(audios=waveform, sampling_rate=16_000, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(**inputs, tgt_lang=language_code, generate_speech=False)
    text = _decode_generated(processor, generated)
    if not text:
        raise ValueError("The model returned an empty transcript. Try clearer speech or a shorter clip.")
    if calculate_attention:
        weights, token_labels = decoder_audio_attention(processor, model, inputs, generated)
        return text, weights, token_labels
    return text


def decoder_audio_attention(processor, model, audio_inputs, generated) -> tuple[np.ndarray, list[str]]:
    """Compute last-layer cross-attention from transcript tokens to audio positions.

    A second teacher-forced decoder pass is used so generation itself does not
    retain attention tensors at every decoding step. Heads are averaged. The
    result is a normalized attention-weight matrix, not a calibrated confidence.
    """
    sequences = getattr(generated, "sequences", None)
    if sequences is None or sequences.shape[-1] < 2:
        raise ValueError("Could not retrieve generated token IDs to calculate attention.")

    with torch.inference_mode():
        speech_encoding = model.speech_encoder(
            input_features=audio_inputs["input_features"],
            return_dict=True,
        )
        # The generated sequence begins with the target-language control token.
        # Shift it right so each decoder row predicts the corresponding next token.
        decoder_inputs = sequences[:, :-1]
        decoder_outputs = model.text_decoder(
            input_ids=decoder_inputs,
            encoder_hidden_states=speech_encoding.last_hidden_state,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    if not decoder_outputs.cross_attentions:
        raise RuntimeError("The model did not return cross-attention weights.")
    # Shape before averaging: batch x heads x output tokens x audio positions.
    weights = decoder_outputs.cross_attentions[-1][0].float().mean(dim=0)
    target_ids = sequences[0, 1:]
    if weights.shape[0] != target_ids.shape[0]:
        raise RuntimeError("Attention rows could not be aligned with generated tokens.")

    tokenizer = processor.tokenizer
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    keep = torch.tensor(
        [int(token_id) not in special_ids for token_id in target_ids.tolist()],
        dtype=torch.bool,
        device=weights.device,
    )
    if keep.any():
        weights = weights[keep]
        target_ids = target_ids[keep]
    token_labels = tokenizer.convert_ids_to_tokens(target_ids.tolist())
    if isinstance(token_labels, str):
        token_labels = [token_labels]
    return weights.cpu().numpy(), token_labels


def translate_text(processor, model, device, text: str, source_code: str, target_code: str) -> str:
    """Translate saved transcript text using SeamlessM4T v2's text-to-text path."""
    if not text.strip():
        raise ValueError("There is no transcript to translate. Transcribe audio first.")
    inputs = processor(text=text, src_lang=source_code, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(**inputs, tgt_lang=target_code, generate_speech=False)
    translated = _decode_generated(processor, generated)
    if not translated:
        raise ValueError("The model returned an empty translation. Try a shorter transcript.")
    return translated


def _decode_generated(processor, generated) -> str:
    """Decode text tokens across Transformers ModelOutput/tuple return forms."""
    if hasattr(generated, "sequences"):
        token_ids = generated.sequences[0]
    elif isinstance(generated, (tuple, list)):
        token_ids = generated[0]
        if getattr(token_ids, "ndim", 0) > 1:
            token_ids = token_ids[0]
    else:
        token_ids = generated[0] if getattr(generated, "ndim", 0) > 1 else generated
    return processor.decode(token_ids.tolist(), skip_special_tokens=True).strip()
