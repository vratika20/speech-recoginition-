"""Minimal Streamlit interface for multilingual, original-language ASR."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import streamlit as st

from audio_utils import AudioError, load_audio
from language_detection import detect_language, load_lid_model
from languages import LANGUAGES, TEXT_TARGET_LANGUAGES
from model import load_model, transcribe, translate_text

st.set_page_config(page_title="Speech to Text", page_icon="", layout="centered")
st.markdown(
    """<style>
    :root { color-scheme: light; }
    .stApp, [data-testid="stAppViewContainer"] { background: #fbfaf7; color: #242622; }
    .block-container { max-width: 820px; padding-top: 3rem; padding-bottom: 4rem; }
    .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label,
    .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"] {
        color: #242622;
    }
    .stApp input, .stApp textarea, .stApp [data-baseweb="select"] > div {
        color: #242622 !important; background-color: #fff !important; border-color: #c9cbc4 !important;
    }
    .stApp input::placeholder, .stApp textarea::placeholder { color: #62665f !important; opacity: 1 !important; }
    .stApp [data-baseweb="select"] *, .stApp [role="option"] { color: #242622 !important; }
    [data-testid="stFileUploader"] section { background: #fff; border: 1px solid #c9cbc4; }
    [data-testid="stFileUploader"] section, [data-testid="stFileUploader"] section * { color: #343831 !important; }
    [data-testid="stFileUploader"] button {
        background: #fff !important; color: #242622 !important; border: 1px solid #8b9087 !important;
    }
    [data-testid="stFileUploader"] button:hover { background: #f0f1ec !important; color: #171a16 !important; }
    .stApp button { color: #242622; }
    .stApp div.stButton > button, .stApp div.stDownloadButton > button {
        color: #242622 !important; background: #fff !important; border: 1px solid #8b9087 !important;
    }
    .stApp div.stButton > button[kind="primary"] {
        color: #fff !important; background: #476b5b !important; border: 1px solid #476b5b !important;
    }
    .stApp [data-testid="stBaseButton-primary"] {
        color: #fff !important; background: #476b5b !important; border: 1px solid #476b5b !important;
    }
    .stApp div.stButton > button[kind="primary"]:hover {
        color: #fff !important; background: #395647 !important; border-color: #395647 !important;
    }
    .stApp [data-testid="stBaseButton-primary"]:hover {
        color: #fff !important; background: #395647 !important; border-color: #395647 !important;
    }
    .stApp div.stButton > button:focus, .stApp div.stDownloadButton > button:focus,
    [data-testid="stFileUploader"] button:focus { outline: 3px solid #8da99a !important; outline-offset: 2px; }
.stApp div.stButton > button:disabled, .stApp div.stDownloadButton > button:disabled,
    .stApp [data-testid^="stBaseButton-"]:disabled {
        color: #555a53 !important; background: #e7e8e3 !important; opacity: 1 !important;
    }
    [data-testid="stAlert"] { color: #242622; border: 1px solid #c9cbc4; }
    [data-testid="stAlert"] * { color: #242622 !important; }
    [data-testid="stStatusWidget"] *, [data-testid="stSpinner"] * { color: #242622 !important; }
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def cached_asr():
    return load_model()


@st.cache_resource(show_spinner=False)
def cached_lid():
    return load_lid_model()


def main() -> None:
    st.title("Speech to text")
    st.write("Transcribe speech in its original language with Meta SeamlessM4T v2.")

    uploaded = st.file_uploader(
        "Choose an audio file", type=["wav", "mp3", "m4a", "flac", "ogg", "aac", "webm"]
    )
    if uploaded is None:
        st.caption("WAV, MP3, M4A, FLAC, OGG, AAC and WEBM are supported when the local decoder can read them.")
        return

    audio_bytes = uploaded.getvalue()
    fingerprint = hashlib.sha256(audio_bytes).hexdigest()
    try:
        waveform, duration = load_audio(audio_bytes, uploaded.name)
    except AudioError as exc:
        st.error(str(exc))
        return

    st.audio(audio_bytes)
    st.caption(f"Audio duration: {duration:.1f} seconds · Resampled to mono 16 kHz for recognition")

    if st.session_state.get("audio_fingerprint") != fingerprint:
        st.session_state.audio_fingerprint = fingerprint
        st.session_state.pop("language_detection", None)
        st.session_state.pop("transcript", None)
        st.session_state.pop("translated_transcript", None)
        st.session_state.pop("transcript_language_code", None)
        st.session_state.pop("transcription_attention", None)

    if "language_detection" not in st.session_state:
        try:
            with st.spinner("Identifying the spoken language…"):
                extractor, lid_model, lid_device = cached_lid()
                st.session_state.language_detection = detect_language(
                    extractor, lid_model, lid_device, waveform
                )
        except Exception as exc:
            st.session_state.language_detection_error = str(exc)

    detection = st.session_state.get("language_detection")
    if detection:
        detected_code = detection.get("seamless_code")
        detected_name = LANGUAGES.get(detected_code, detection["lid_code"])
        st.info(
            f"Detected language: **{detected_name}** · MMS model score: "
            f"{detection['score']:.1%} (raw, uncalibrated probability)"
        )
        if detected_code not in LANGUAGES:
            st.warning(
                f"The detected language ({detected_name}) is not supported for same-language text output "
                "by SeamlessM4T v2. Choose a supported language manually only if you know the audio is in that language."
            )
    else:
        st.warning("Automatic language identification is unavailable. Select the spoken language manually to continue.")
        if st.session_state.get("language_detection_error"):
            st.caption("Language identification model could not load. You can still transcribe with a manual selection.")

    options = ["Auto Detect"] + sorted(LANGUAGES.values(), key=str.casefold)
    choice = st.selectbox(
        "Spoken language",
        options,
        help="Auto Detect uses the language identifier above. Manual selection forces SeamlessM4T's same-language text output.",
    )
    selected_code = next((code for code, name in LANGUAGES.items() if name == choice), None)

    if selected_code and detection and detection.get("seamless_code") in LANGUAGES:
        if selected_code != detection["seamless_code"]:
            st.warning(f"Manual selection ({choice}) differs from the detected language ({LANGUAGES[detection['seamless_code']]}).")
    elif choice == "Auto Detect" and detection and detection.get("seamless_code") not in LANGUAGES:
        st.error("This detected language cannot be transcribed to same-language text by this model. Choose another supported source language only if appropriate.")

    calculate_attention = st.checkbox(
        "Calculate decoder attention matrix",
        value=True,
        help="Adds a second model pass to calculate how generated text tokens attend to audio encoder positions. This uses extra memory.",
    )

    if st.button("Transcribe audio", type="primary", use_container_width=True):
        effective_code = selected_code or (detection or {}).get("seamless_code")
        if not effective_code or effective_code not in LANGUAGES:
            st.error("Select a supported language before transcription. No fallback language was selected.")
            return
        try:
            with st.spinner("Loading SeamlessM4T v2 and transcribing… first model download may take several minutes."):
                processor, asr_model, device = cached_asr()
                result = transcribe(
                    processor,
                    asr_model,
                    device,
                    waveform,
                    effective_code,
                    calculate_attention=calculate_attention,
                )
            st.session_state.pop("transcription_attention", None)
            if calculate_attention:
                transcript, attention_weights, attention_tokens = result
                st.session_state.transcription_attention = {
                    "weights": attention_weights.tolist(),
                    "tokens": attention_tokens,
                }
            else:
                transcript = result
            st.session_state.transcript = transcript
            st.session_state.transcript_language = LANGUAGES[effective_code]
            st.session_state.transcript_language_code = effective_code
            st.session_state.pop("translated_transcript", None)
        except Exception as exc:
            st.error(f"Transcription failed: {exc}")
            st.caption("Check the first-run model download and available memory. You can turn off attention calculation or try a shorter clip.")

    transcript = st.session_state.get("transcript")
    if transcript:
        st.subheader(f"Transcript · {st.session_state.transcript_language}")
        st.text_area("Original-language transcript", value=transcript, height=220, label_visibility="collapsed")
        st.download_button(
            "Download Transcript",
            data=transcript.encode("utf-8"),
            file_name="transcript.txt",
            mime="text/plain; charset=utf-8",
        )

        attention_data = st.session_state.get("transcription_attention")
        if attention_data:
            render_attention_matrix(attention_data)

        st.markdown("### Translate Transcript")
        target_names = sorted(TEXT_TARGET_LANGUAGES.values(), key=str.casefold)
        default_target = TEXT_TARGET_LANGUAGES["eng"]
        target_index = target_names.index(default_target)
        target_name = st.selectbox(
            "Target language",
            target_names,
            index=target_index,
            key="translation_target_name",
        )
        target_code = next(code for code, name in TEXT_TARGET_LANGUAGES.items() if name == target_name)
        if st.button("Translate", type="primary", use_container_width=True):
            source_code = st.session_state.get("transcript_language_code")
            if not source_code or source_code not in LANGUAGES:
                st.error("The transcript source language is unavailable. Transcribe the audio again and retry.")
            elif source_code not in TEXT_TARGET_LANGUAGES:
                st.error("SeamlessM4T v2 does not support text input for this transcript language.")
            else:
                try:
                    with st.spinner(f"Translating to {target_name}…"):
                        processor, translation_model, device = cached_asr()
                        translated = translate_text(
                            processor,
                            translation_model,
                            device,
                            transcript,
                            source_code,
                            target_code,
                        )
                    st.session_state.translated_transcript = translated
                    st.session_state.translated_target_code = target_code
                except (MemoryError, RuntimeError) as exc:
                    st.error(f"Translation could not be completed because of a model or memory error: {exc}")
                    st.caption("Close other memory-heavy applications or try a shorter transcript.")
                except Exception as exc:
                    st.error(f"Translation failed for {st.session_state.transcript_language} to {target_name}: {exc}")

        translated = st.session_state.get("translated_transcript")
        if translated:
            previous_target = st.session_state.get("translated_target_code")
            translated_name = TEXT_TARGET_LANGUAGES.get(previous_target, target_name)
            st.subheader(f"Translation · {translated_name}")
            st.text_area(
                "Translated transcript",
                value=translated,
                height=220,
                label_visibility="collapsed",
                key="translated_transcript_display",
            )
            st.download_button(
                "Download Translation",
                data=translated.encode("utf-8"),
                file_name=f"translation_{previous_target or target_code}.txt",
                mime="text/plain; charset=utf-8",
            )


def render_attention_matrix(attention_data: dict) -> None:
    """Show and download head-averaged final decoder cross-attention weights."""
    weights = np.asarray(attention_data["weights"], dtype=np.float32)
    tokens = attention_data["tokens"]
    if weights.ndim != 2 or weights.shape[0] == 0 or weights.shape[1] == 0:
        st.warning("The model did not produce a displayable attention matrix.")
        return

    original_positions = weights.shape[1]
    displayed_bins = min(original_positions, 64)
    if displayed_bins < original_positions:
        edges = np.linspace(0, original_positions, displayed_bins + 1, dtype=int)
        display_weights = np.stack(
            [weights[:, edges[i] : edges[i + 1]].sum(axis=1) for i in range(displayed_bins)],
            axis=1,
        )
        columns = [f"Audio positions {edges[i] + 1}–{edges[i + 1]}" for i in range(displayed_bins)]
    else:
        display_weights = weights
        columns = [f"Audio position {i + 1}" for i in range(displayed_bins)]

    # Rows correspond to generated text tokens; columns are speech-encoder positions.
    table = pd.DataFrame(display_weights, index=tokens, columns=columns)
    with st.expander("Decoder attention matrix", expanded=False):
        st.caption(
            f"Last decoder layer, averaged over attention heads. Rows are transcript tokens; "
            f"columns are audio-encoder positions. Display shows {displayed_bins} bins from "
            f"{original_positions} positions; attention mass within each bin is summed."
        )
        st.dataframe(
            table.style.background_gradient(cmap="Blues", axis=None).format("{:.4f}"),
            use_container_width=True,
            height=420,
        )
        st.download_button(
            "Download attention matrix (.csv)",
            data=table.to_csv().encode("utf-8"),
            file_name="decoder_cross_attention.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
