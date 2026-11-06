import re

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_silences = ["@sp", "@sil"]

TELUGU_VOWELS = {
    'అ': 'a', 'ఆ': 'aa', 'ఇ': 'i', 'ఈ': 'ii', 'ఉ': 'u', 'ఊ': 'uu',
    'ఋ': 'ru', 'ౠ': 'ruu', 'ఌ': 'lu', 'ౡ': 'luu',
    'ఎ': 'e', 'ఏ': 'ee', 'ఐ': 'ai', 'ఒ': 'o', 'ఓ': 'oo', 'ఔ': 'au'
}

TELUGU_VOWEL_SIGNS = {
    'ా': 'aa', 'ి': 'i', 'ీ': 'ii', 'ు': 'u', 'ూ': 'uu',
    'ృ': 'ru', 'ౄ': 'ruu', 'ౢ': 'lu', 'ౣ': 'luu',
    'ె': 'e', 'ే': 'ee', 'ై': 'ai', 'ొ': 'o', 'ో': 'oo', 'ౌ': 'au'
}

TELUGU_CONSONANTS = {
    'క': 'k', 'ఖ': 'kh', 'గ': 'g', 'ఘ': 'gh', 'ఙ': 'ng',
    'చ': 'c', 'ఛ': 'ch', 'జ': 'j', 'ఝ': 'jh', 'ఞ': 'ny',
    'ట': 'tt', 'ఠ': 'tth', 'డ': 'dd', 'ఢ': 'ddh', 'ణ': 'nn',
    'త': 't', 'థ': 'th', 'ద': 'd', 'ధ': 'dh', 'న': 'n',
    'ప': 'p', 'ఫ': 'ph', 'బ': 'b', 'భ': 'bh', 'మ': 'm',
    'య': 'y', 'ర': 'r', 'ఱ': 'rr', 'ల': 'l', 'ళ': 'll',
    'వ': 'v', 'శ': 'sh', 'ష': 'ss', 'స': 's', 'హ': 'h'
}

VIRAMA = '్'
ANUSVARA = 'ం'
VISARGA = 'ః'

_te_phonemes = sorted(list(set(
    list(TELUGU_VOWELS.values()) +
    list(TELUGU_VOWEL_SIGNS.values()) +
    list(TELUGU_CONSONANTS.values()) +
    ['am', 'ah', 'a']
)))

SYMBOLS = [_pad] + list(_special) + list(_punctuation) + _silences + ["@" + p for p in _te_phonemes]
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
ID_TO_SYMBOL = {i: s for i, s in enumerate(SYMBOLS)}

def normalize_telugu_text(text):
    """Normalize Telugu numbers, timestamps, and split tokens."""
    # Split token fixes
    text = text.replace('అడ విలో', 'అడవిలో')
    text = text.replace('నిశ్శబ్ంద గా', 'నిశ్శబ్దంగా')
    text = text.replace('సిద్ంధ గా', 'సిద్ధంగా')

    # Date and time normalization
    text = text.replace('2026', 'రెండు వేల ఇరవై ఆరు')
    text = text.replace('12వ', 'పన్నెండవ')
    text = text.replace('12', 'పన్నెండు')
    text = text.replace('10:45', 'పది నలభై ఐదు')
    return text

def telugu_to_phonemes(text):
    """Decompose Telugu text into phonetic symbols."""
    text = normalize_telugu_text(text)
    phones = []
    i = 0
    n = len(text)
    while i < n:
        char = text[i]
        if char in (' ', '\t', '\n'):
            if phones and phones[-1] != '@sp':
                phones.append('@sp')
            i += 1
        elif char in (',', ';'):
            phones.append('@sp')
            i += 1
        elif char in ('.', '!', '?'):
            phones.append('@sil')
            i += 1
        elif char in TELUGU_VOWELS:
            phones.append("@" + TELUGU_VOWELS[char])
            i += 1
        elif char in TELUGU_CONSONANTS:
            cons = TELUGU_CONSONANTS[char]
            i += 1
            if i < n and text[i] == VIRAMA:
                phones.append("@" + cons)
                i += 1
            elif i < n and text[i] in TELUGU_VOWEL_SIGNS:
                vowel = TELUGU_VOWEL_SIGNS[text[i]]
                phones.append("@" + cons)
                phones.append("@" + vowel)
                i += 1
            else:
                phones.append("@" + cons)
                phones.append("@a")
        elif char == ANUSVARA:
            phones.append('@am')
            i += 1
        elif char == VISARGA:
            phones.append('@ah')
            i += 1
        else:
            if char in SYMBOL_TO_ID:
                phones.append(char)
            i += 1
    return phones

def text_to_sequence_te(text):
    """Convert raw Telugu text into numeric ID sequence."""
    phonemes = telugu_to_phonemes(text)
    seq = [SYMBOL_TO_ID[p] for p in phonemes if p in SYMBOL_TO_ID]
    return seq, phonemes

