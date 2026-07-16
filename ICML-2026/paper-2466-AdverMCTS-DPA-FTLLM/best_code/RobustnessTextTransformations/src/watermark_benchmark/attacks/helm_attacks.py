import random
import re
from .utils import ATTACK_LIST, Attack
from typing import Optional, List
import json

# Global vars
GLOBALS = {}
CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'll": "I will",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "ma'am": "madam",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there would",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what're": "what are",
    "what's": "what is",
    "when's": "when is",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}
CONTRACTION_PATTERN = re.compile(r"\b({})\b".format("|".join(CONTRACTION_MAP.keys())), flags=re.IGNORECASE | re.DOTALL, )
REVERSE_CONTRACTION_MAP = {value: key for key, value in CONTRACTION_MAP.items()}
REVERSE_CONTRACTION_PATTERN = re.compile(r"\b({})\b ".format("|".join(REVERSE_CONTRACTION_MAP.keys())),flags=re.IGNORECASE | re.DOTALL,)


def setup(config):
    with open(config.misspellings, "r") as f:
        GLOBALS['correct_to_misspelling'] = json.load(f)
        GLOBALS['mispelling_pattern'] = re.compile(r"\b({})\b".format("|".join(GLOBALS['correct_to_misspelling'].keys())))


def match_case(source_word: str, target_word: str) -> str:
    """Return a version of the target_word where the case matches the source_word."""
    # Check for all lower case source_word
    if all(letter.islower() for letter in source_word):
        return target_word.lower()
    # Check for all caps source_word
    if all(letter.isupper() for letter in source_word):
        return target_word.upper()
    # Check for capital source_word
    if source_word and source_word[0].isupper():
        return target_word.capitalize()
    return target_word


class MisspellingAttack(Attack):
    """ Using HELM code """

    def __init__(self, prob: float):
        super().__init__('MisspellingAttack_{}'.format(prob))
        self.prob: float = prob

    @staticmethod
    def get_param_list():
        basename = "MisspellingAttack_{}"
        raw_params = [(0.25,),(0.5,)]
        return [(basename.format(p), p) for p in raw_params]

    def warp(self, text: str, input_encodings: Optional[List] = None) -> str: 
        def mispell(match: re.Match) -> str:
            word = match.group(1)
            if random.random() < self.prob:
                mispelled_word = str(random.choice(GLOBALS['correct_to_misspelling'][word]))
                return match_case(word, mispelled_word)
            else:
                return word

        return GLOBALS['mispelling_pattern'].sub(mispell, text)


class TypoAttack(Attack):
    """ From HELM """
    def __init__(self, prob):
        super().__init__('TypoAttack_{}'.format(prob))
        self.prob = prob
        key_approx = {}
        key_approx["q"] = "was"
        key_approx["w"] = "qesad"
        key_approx["e"] = "wsdfr"
        key_approx["r"] = "edfgt"
        key_approx["t"] = "rfghy"
        key_approx["y"] = "tghju"
        key_approx["u"] = "yhjki"
        key_approx["i"] = "ujklo"
        key_approx["o"] = "iklp"
        key_approx["p"] = "ol"

        key_approx["a"] = "qwsz"
        key_approx["s"] = "weadzx"
        key_approx["d"] = "erfcxs"
        key_approx["f"] = "rtgvcd"
        key_approx["g"] = "tyhbvf"
        key_approx["h"] = "yujnbg"
        key_approx["j"] = "uikmnh"
        key_approx["k"] = "iolmj"
        key_approx["l"] = "opk"

        key_approx["z"] = "asx"
        key_approx["x"] = "sdcz"
        key_approx["c"] = "dfvx"
        key_approx["v"] = "fgbc"
        key_approx["b"] = "ghnv"
        key_approx["n"] = "hjmb"
        key_approx["m"] = "jkn"
        self.key_approx = key_approx


    @staticmethod
    def get_param_list():
        basename = "TypoAttack_{}"
        raw_params = [(0.05,),(0.1,)]
        return [(basename.format(p), p) for p in raw_params]

    def warp(self, text: str, input_encodings: Optional[List] = None) -> str: 
        perturbed_texts = ""
        for letter in text:
            lcletter = letter.lower()
            if lcletter not in self.key_approx.keys():
                new_letter = lcletter
            else:
                if random.random() < self.prob:
                    new_letter = random.choice(list(self.key_approx[lcletter]))
                else:
                    new_letter = lcletter
            # go back to original case
            if not lcletter == letter:
                new_letter = new_letter.upper()
            perturbed_texts += new_letter
        return perturbed_texts


class LowercaseAttack(Attack):
    """ From HELM """
    def __init__(self):
        super().__init__('LowercaseAttack')

    @staticmethod
    def get_param_list():
        return [("LowercaseAttack", ())]

    def warp(self, text: str, input_encodings: Optional[List] = None) -> str: 
        return text.lower()


class ContractionAttack(Attack):
    """ From HELM """
    def __init__(self):
        super().__init__('ContractionAttack')

    @staticmethod
    def get_param_list():
        return [("ContractionAttack", ())]

    def warp(self, text: str, input_encodings: Optional[List] = None) -> str: 
        def cont(possible):
            match = possible.group(1)
            expanded_contraction = REVERSE_CONTRACTION_MAP.get(
                match, REVERSE_CONTRACTION_MAP.get(match.lower())
            )
            return match_case(match, expanded_contraction) + " "

        return REVERSE_CONTRACTION_PATTERN.sub(cont, text)


class ExpansionAttack(Attack):
    """ From HELM """
    def __init__(self):
        super().__init__('ExpansionAttack')

    @staticmethod
    def get_param_list():
        return [("ExpansionAttack", ())]

    def warp(self, text: str, input_encodings: Optional[List] = None) -> str: 
        def expand_match(contraction):
            match = contraction.group(0)
            expanded_contraction = CONTRACTION_MAP.get(match, CONTRACTION_MAP.get(match.lower()))
            return match_case(match, expanded_contraction)

        return CONTRACTION_PATTERN.sub(expand_match, text)


def init_helm_attacks(names_only=False):
    dest = {}
    for att in (MisspellingAttack, TypoAttack, ContractionAttack, LowercaseAttack):
        for name, params in att.get_param_list():
            dest[name] = att(*params) if not names_only else True

    return dest
        
