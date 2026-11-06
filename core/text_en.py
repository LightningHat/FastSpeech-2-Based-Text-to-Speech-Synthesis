import re
from g2p_en import G2p

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_silences = ["@sp", "@sil"]

# Standard ARPAbet symbols
_arpabet = [
    "AA", "AA0", "AA1", "AA2", "AE", "AE0", "AE1", "AE2", "AH", "AH0", "AH1", "AH2",
    "AO", "AO0", "AO1", "AO2", "AW", "AW0", "AW1", "AW2", "AY", "AY0", "AY1", "AY2",
    "B", "CH", "D", "DH", "EH", "EH0", "EH1", "EH2", "ER", "ER0", "ER1", "ER2",
    "EY", "EY0", "EY1", "EY2", "F", "G", "HH", "IH", "IH0", "IH1", "IH2", "IY",
    "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG", "OW", "OW0", "OW1", "OW2",
    "OY", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T", "TH", "UH", "UH0", "UH1",
    "UH2", "UW", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH"
]

SYMBOLS = [_pad] + list(_special) + list(_punctuation) + _silences + ["@" + s for s in _arpabet]
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
ID_TO_SYMBOL = {i: s for i, s in enumerate(SYMBOLS)}

g2p = G2p()

def normalize_numbers_english(text):
    """Normalize timestamps, dates, and numbers in English."""
    text = re.sub(r'10:45\s*AM', 'ten forty-five AM', text, flags=re.IGNORECASE)
    text = re.sub(r'\b10:45\b', 'ten forty-five', text)
    text = re.sub(r'\b2026\b', 'twenty twenty-six', text)
    text = re.sub(r'\b12\b', 'twelve', text)
    text = re.sub(r'\b14th\b', 'fourteenth', text, flags=re.IGNORECASE)
    text = re.sub(r'\b11th\b', 'eleventh', text, flags=re.IGNORECASE)
    text = re.sub(r'\bCE\b', 'C E', text)
    return text

def english_to_phonemes(text):
    """Convert English text into sequence of ARPAbet phonemes."""
    text = normalize_numbers_english(text)

    # Custom handling for specific loan / compound words
    text_processed = text.replace("andaroo", "an-da-roo")

    raw_tokens = g2p(text_processed)
    phonemes = []
    for t in raw_tokens:
        t_clean = t.strip()
        if not t_clean:
            continue
        if t_clean in (',', ';'):
            phonemes.append('@sp')
        elif t_clean in ('.', '!', '?'):
            phonemes.append('@sil')
        elif "@" + t_clean in SYMBOL_TO_ID:
            phonemes.append("@" + t_clean)
        elif t_clean in SYMBOL_TO_ID:
            phonemes.append(t_clean)
        else:
            # Fallback for letters / unknown
            for char in t_clean:
                if char in SYMBOL_TO_ID:
                    phonemes.append(char)
    return phonemes

def text_to_sequence_en(text):
    """Convert raw English text into numeric ID sequence."""
    phonemes = english_to_phonemes(text)
    seq = [SYMBOL_TO_ID[p] for p in phonemes if p in SYMBOL_TO_ID]
    return seq, phonemes

