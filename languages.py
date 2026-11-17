"""Languages verified against the SeamlessM4T v2 model card.

Only entries supporting source speech (Sp) and target text (Tx) are offered.
The model card is at https://huggingface.co/facebook/seamless-m4t-v2-large
"""

# SeamlessM4T code -> display name. Filtered to Source includes Sp and Target
# includes Tx, because both are required for same-language speech recognition.
LANGUAGES = {
    "afr": "Afrikaans", "amh": "Amharic", "arb": "Arabic (Modern Standard)",
    "ary": "Arabic (Moroccan)", "arz": "Arabic (Egyptian)", "asm": "Assamese",
    "azj": "Azerbaijani (North)", "bel": "Belarusian", "ben": "Bengali",
    "bos": "Bosnian", "bul": "Bulgarian", "cat": "Catalan", "ceb": "Cebuano",
    "ces": "Czech", "ckb": "Kurdish (Central)", "cmn": "Chinese (Mandarin, Simplified)",
    "cmn_Hant": "Chinese (Mandarin, Traditional)", "cym": "Welsh", "dan": "Danish",
    "deu": "German", "ell": "Greek", "eng": "English", "est": "Estonian",
    "eus": "Basque", "fin": "Finnish", "fra": "French", "fuv": "Fulfulde (Nigerian)",
    "gaz": "Oromo (West Central)", "gle": "Irish", "glg": "Galician", "guj": "Gujarati",
    "heb": "Hebrew", "hin": "Hindi", "hrv": "Croatian", "hun": "Hungarian",
    "hye": "Armenian", "ibo": "Igbo", "ind": "Indonesian", "isl": "Icelandic",
    "ita": "Italian", "jav": "Javanese", "jpn": "Japanese", "kan": "Kannada",
    "kat": "Georgian", "kaz": "Kazakh", "khk": "Mongolian (Halh)", "khm": "Khmer",
    "kir": "Kyrgyz", "kor": "Korean", "lao": "Lao", "lit": "Lithuanian",
    "lug": "Ganda", "luo": "Luo", "lvs": "Latvian", "mai": "Maithili",
    "mal": "Malayalam", "mar": "Marathi",
    "mkd": "Macedonian", "mlt": "Maltese", "mni": "Meitei", "mya": "Burmese",
    "nld": "Dutch", "nno": "Norwegian (Nynorsk)", "nob": "Norwegian (Bokmål)",
    "npi": "Nepali", "nya": "Chichewa", "ory": "Odia", "pan": "Punjabi",
    "pbt": "Pashto (Southern)", "pes": "Persian (Western)", "pol": "Polish",
    "por": "Portuguese", "ron": "Romanian", "rus": "Russian", "slk": "Slovak",
    "slv": "Slovenian", "sna": "Shona", "snd": "Sindhi", "som": "Somali",
    "spa": "Spanish", "srp": "Serbian", "swe": "Swedish", "swh": "Swahili",
    "tam": "Tamil", "tel": "Telugu", "tgk": "Tajik", "tgl": "Tagalog",
    "tha": "Thai", "tur": "Turkish", "ukr": "Ukrainian", "urd": "Urdu",
    "uzn": "Uzbek (Northern)", "vie": "Vietnamese", "yue": "Cantonese",
    "yor": "Yoruba", "zul": "Zulu",
}

# Languages listed by the official model card as text generation targets (Tx).
# Text-to-text translation can use a broader target set than speech recognition.
TEXT_TARGET_LANGUAGES = {
    "afr": "Afrikaans", "amh": "Amharic", "arb": "Arabic (Modern Standard)",
    "ary": "Arabic (Moroccan)", "arz": "Arabic (Egyptian)", "asm": "Assamese",
    "azj": "Azerbaijani (North)", "bel": "Belarusian", "ben": "Bengali",
    "bos": "Bosnian", "bul": "Bulgarian", "cat": "Catalan", "ceb": "Cebuano",
    "ces": "Czech", "ckb": "Kurdish (Central)", "cmn": "Chinese (Mandarin, Simplified)",
    "cmn_Hant": "Chinese (Mandarin, Traditional)", "cym": "Welsh", "dan": "Danish",
    "deu": "German", "ell": "Greek", "eng": "English", "est": "Estonian",
    "eus": "Basque", "fin": "Finnish", "fra": "French", "fuv": "Fulfulde (Nigerian)",
    "gaz": "Oromo (West Central)", "gle": "Irish", "glg": "Galician", "guj": "Gujarati",
    "heb": "Hebrew", "hin": "Hindi", "hrv": "Croatian", "hun": "Hungarian",
    "hye": "Armenian", "ibo": "Igbo", "ind": "Indonesian", "isl": "Icelandic",
    "ita": "Italian", "jav": "Javanese", "jpn": "Japanese", "kan": "Kannada",
    "kat": "Georgian", "kaz": "Kazakh", "khk": "Mongolian (Halh)", "khm": "Khmer",
    "kir": "Kyrgyz", "kor": "Korean", "lao": "Lao", "lit": "Lithuanian",
    "lug": "Ganda", "luo": "Luo", "lvs": "Latvian", "mai": "Maithili",
    "mal": "Malayalam", "mar": "Marathi", "mkd": "Macedonian", "mlt": "Maltese",
    "mni": "Meitei", "mya": "Burmese", "nld": "Dutch", "nno": "Norwegian (Nynorsk)",
    "nob": "Norwegian (Bokmål)", "npi": "Nepali", "nya": "Chichewa", "ory": "Odia",
    "pan": "Punjabi", "pbt": "Pashto (Southern)", "pes": "Persian (Western)",
    "pol": "Polish", "por": "Portuguese", "ron": "Romanian", "rus": "Russian",
    "slk": "Slovak", "slv": "Slovenian", "sna": "Shona", "snd": "Sindhi",
    "som": "Somali", "spa": "Spanish", "srp": "Serbian", "swe": "Swedish",
    "swh": "Swahili", "tam": "Tamil", "tel": "Telugu", "tgk": "Tajik",
    "tgl": "Tagalog", "tha": "Thai", "tur": "Turkish", "ukr": "Ukrainian",
    "urd": "Urdu", "uzn": "Uzbek (Northern)", "vie": "Vietnamese",
    "yue": "Cantonese", "yor": "Yoruba", "zsm": "Malay (Standard)", "zul": "Zulu",
}

# MMS LID uses ISO-639-3 labels. Arabic `ara` is mapped to the model card's
# Modern Standard Arabic output code; Arabic dialect distinctions are not made.
LID_TO_SEAMLESS = {code: code for code in LANGUAGES}
LID_TO_SEAMLESS.update({"ara": "arb"})
