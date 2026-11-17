# Multilingual Speech Recognition

A local Streamlit app that detects a spoken language, transcribes speech in that same language, and can translate the saved transcript using the pretrained Meta [`facebook/seamless-m4t-v2-large`](https://huggingface.co/facebook/seamless-m4t-v2-large) model. It does not train a model.

## Files

- `app.py` — Streamlit upload, language selection, transcript and download UI.
- `model.py` — cached unified SeamlessM4T v2 loading for transcription, translation and optional decoder attention extraction.
- `language_detection.py` — Meta MMS 126-language speech LID and its raw probability score.
- `audio_utils.py` — file validation, decoding, mono conversion and 16 kHz resampling.
- `languages.py` — verified speech transcription languages and text-translation target languages.

## Language coverage

The language picker includes only languages that the official SeamlessM4T v2 model card marks as both **source speech** and **target text**. This intersection is needed to produce a transcript in the input language. The listed Indian languages include English, Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Urdu and Odia. Listed international languages include Spanish, French, German, Portuguese, Italian, Russian, Arabic, Mandarin Chinese, Japanese, Korean, Indonesian, Thai, Turkish, Dutch, Polish and Vietnamese.

Some source speech languages do not have target text support in this checkpoint, so they are not offered for same-language transcription. The language list and SeamlessM4T codes come from the [official model card](https://huggingface.co/facebook/seamless-m4t-v2-large#supported-languages). Transcription sets `tgt_lang` to the selected or detected source-language code to request same-language text. Translation uses the same cached unified model with the saved transcript, `src_lang`, `tgt_lang` and `generate_speech=False`.

Automatic language identification uses Meta [`facebook/mms-lid-126`](https://huggingface.co/facebook/mms-lid-126). Its top-class softmax probability is displayed as a **raw, uncalibrated model score**, not a calibrated likelihood of correctness. MMS uses `ara` for Arabic while SeamlessM4T distinguishes Arabic varieties; the app maps it to Modern Standard Arabic (`arb`) because the identifier does not distinguish those varieties. If the detected language cannot be transcribed to same-language text, the app reports that and does not silently choose another language. A user can manually select the expected language.

## Install and run

Python 3.10–3.12 is recommended. PyTorch wheels and device acceleration vary by operating system. Install FFmpeg if the local audio decoder cannot read MP3 or M4A files.

```bash
cd multilingual-speech-recognition
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. The first language-detection request downloads MMS LID, and the first transcription downloads SeamlessM4T v2. Both models are cached in memory by Streamlit for later requests.

## Use

1. Upload WAV, MP3, M4A, FLAC, OGG, AAC or WEBM audio.
2. Review the detected language and its raw MMS model score.
3. Keep **Auto Detect** or select the known spoken language manually. A mismatch is shown as a warning.
4. Choose **Transcribe audio** and download the UTF-8 transcript as a TXT file.
5. Under **Translate Transcript**, choose a target language and click **Translate**. Change the target language and click again to translate the same transcript; the original transcript is preserved.
6. **Calculate decoder attention matrix** is enabled by default. The app makes one extra decoder pass and shows the final decoder layer's cross-attention averaged across heads. Matrix rows are generated text tokens and columns are audio-encoder time positions. The displayed matrix groups long audio axes into at most 64 bins; download the CSV for the displayed values. Turn this option off to reduce memory use.

## Hardware and limitations

SeamlessM4T v2 Large has about 2.3 billion parameters. MMS LID is also a large 1-billion-parameter checkpoint. Use a machine with substantial free RAM (16 GB or more is a practical baseline); a CUDA GPU with adequate memory can improve speed. CPU inference is supported but can be slow and may run out of memory. Attention extraction uses extra memory. The unified SeamlessM4T model is cached once and reused for transcription and translation. If loading or generation fails, close other memory-heavy applications, turn off attention calculation, or use a shorter transcript/audio clip. The model weights and generated transcripts remain local to this app, except for the initial Hugging Face model downloads.

Attention weights are diagnostic signals, not calibrated measures of how correct or important a word is. The confusion matrix for language identification is a separate evaluation: it requires a dataset with known correct language labels.

Accuracy depends on recording clarity, accent, vocabulary and language. LID softmax scores are not calibrated confidence estimates, and short or mixed-language recordings may be misidentified. Manual language selection can improve transcription when the source language is known. No numeric transcription confidence is available from this generation path, so the app does not report one. M4A/MP3 decoding may require FFmpeg depending on the operating system and installed codecs.

The pretrained SeamlessM4T model is distributed under the [CC-BY-NC 4.0 license](https://huggingface.co/facebook/seamless-m4t-v2-large). Review the model terms for your intended use.
