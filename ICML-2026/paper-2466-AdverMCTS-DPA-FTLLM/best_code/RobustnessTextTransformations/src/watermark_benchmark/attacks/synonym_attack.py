import math
import random
import re
import sys
from multiprocessing import Queue

import nltk
import numpy as np
import textattack
import torch
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize as wt
from pyinflect import getAllInflections, getInflection
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

from .utils import Attack

CONFIG = {
    "server": None,
    "synonym_lem": WordNetLemmatizer(),
    "prefix": "",
    "suffix": "",
}
local_cache = {}


class SynonymAttack(Attack):

    def __init__(
        self,
        p=1.0,
        confidence=0.01,
        generation_queue=None,
        resp_queue=None,
        cache=None,
    ):
        super().__init__("SynonymAttack")
        self.p = p
        self.confidence = confidence
        self.no_replace_list = set("as")
        self.regex = re.compile("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s")
        self.queue = generation_queue
        self.resp_queue = resp_queue
        self.cache = cache if cache else local_cache

        # Running fast mode
        self.mode = "fast"
        self.window = 0

    @staticmethod
    def get_param_list():
        return [
            ("SynonymAttack_{}".format(i), (i,)) for i in [0.25, 0.5, 0.75, 1]
        ]

    def warp(self, text, input_str=None):
        text = text.replace("<pad>", "").replace("</s>", "")
        self.generate_synonyms_fast(text)

        separators = [
            match.end() for match in re.finditer(self.regex, text)
        ] + [len(text)]
        rslt = self.warp_sentence(text[: separators[0]], input_str)
        for i in range(len(separators) - 1):
            rslt += self.warp_sentence(
                text[separators[i] : separators[i + 1]], input_str
            )
        return rslt

    def inflect(self, word, tag):
        inflections = getInflection(
            CONFIG["synonym_lem"].lemmatize(word), tag=tag
        )
        if not inflections or not len(inflections):
            return word
        else:
            return inflections[0]

    def wordNetSynonyms(self, word, pos):
        return list(
            set(s.lemmas()[0].name() for s in wordnet.synsets(word, pos=pos))
        )

    def generate_synonyms_fast(self, text):

        cont_sent_token = wt(text)
        cont_sent_tag = nltk.pos_tag(cont_sent_token)
        len_sentence = len(cont_sent_token)

        for i in reversed(range(len_sentence)):
            # Filter non replacable words
            if not self.is_replacable(*cont_sent_tag[i]):
                continue

            word = cont_sent_token[i]
            w, t = cont_sent_tag[i]
            context = self.get_context(i, cont_sent_token, t)
            if (context, word) in self.cache:
                continue

            pos = self.convert_to_pos(t)
            options = [
                self.inflect(word, t)
                for word in set(self.wordNetSynonyms(w, pos))
                if textattack.shared.utils.is_one_word(word)
                and not textattack.shared.utils.check_if_punctuations(word)
                and not self.check_if_present(word, context, pos)
                and not "_" in word
            ]

            self.cache[(context, word)] = options

    def generate_synonyms_full(self, text):

        # Old code. This needs to be adapted to new servers
        pass

        # Break down into sentences
        separators = (
            [0]
            + [match.end() for match in re.finditer(self.regex, text)]
            + [len(text)]
        )
        sentences = [
            text[separators[i] : separators[i + 1]]
            for i in range(len(separators) - 1)
        ]
        questions, queries, wn_syns, tags = [], [], [], []
        word_replacements = {}

        for sentence in sentences:

            # Parse sentence
            cont_sent_token = wt(sentence)
            cont_sent_tag = nltk.pos_tag(cont_sent_token)
            len_sentence = len(cont_sent_token)

            for i in reversed(range(len_sentence)):
                # Filter non replacable words
                if not self.is_replacable(*cont_sent_tag[i]):
                    continue

                w, t = cont_sent_tag[i]
                word = cont_sent_token[i]
                context = self.get_context(i, cont_sent_token, t)
                if (context, word) in self.cache:
                    continue

                questions.append(
                    "Human: List 5 synonyms for the word '{}' in the following context: '{}'\nAssistant: 1.".format(
                        word, " ".join(context)
                    )
                )
                queries.append((context, cont_sent_token[i]))

                pos = self.convert_to_pos(t)
                wn_syns.append(self.wordNetSynonyms(w, pos))
                tags.append(t)

        request = {
            "temperature": 1.0,
            "num_return_sequences": 1,
            "max_new_tokens": 64,
            "echo_prompt": False,
            "remove_special_tokens": True,
        }

        if not len(questions):
            return

        if self.queue:
            self.queue.put(
                (
                    "generation",
                    (questions, request, None, False, None, self.resp_queue),
                )
            )
            resp = []
            while len(resp) < len(questions):
                resp.extend(self.resp_queue.get(block=True)[0])
        else:
            resp = [
                v[0]
                for v in call_local_server(
                    CONFIG["server"], questions, request, None, None, None
                )
            ][0]

        for pair in zip(queries, resp, wn_syns, tags):
            query, results, wn_syn_list, tag = pair
            text_result = results.text
            word_list = [
                re.sub("[0-9\"'`]", "", v).replace(".", "").strip().lower()
                for v in text_result.split("Human:")[0]
                .split("###")[0]
                .strip()
                .split("\n")
                if len(v)
            ]
            full_word_list = []
            for w in word_list:
                if ":" in w:
                    for tok in w.split(":")[-1].split(","):
                        full_word_list.append(tok.strip())
                elif len(w) > 32:
                    continue
                else:
                    full_word_list.append(w)

            context, orig_word = query
            options = [
                self.inflect(word, tag)
                for word in set(full_word_list + wn_syn_list)
                if textattack.shared.utils.is_one_word(word)
                and not textattack.shared.utils.check_if_punctuations(word)
                and not self.check_if_present(
                    word, context, self.convert_to_pos(tag)
                )
                and not "_" in word
            ]

            self.cache[query] = options

    def is_replacable(self, word, tag):
        if textattack.shared.utils.check_if_punctuations(word):
            return False

        if len(word) <= 3:
            return False

        if word in self.no_replace_list:
            return False

        if tag not in (
            "JJ",
            "JJR",
            "JJS",
            "RB",
            "RBR",
            "RBS",
            "VB",
            "VBD",
            "VBG",
            "VBN",
            "VBP",
            "VBZ",
            "NN",
            "NNS",
        ):
            return False

        if self.convert_to_pos(tag) == "v":
            w = CONFIG["synonym_lem"].lemmatize(word, pos="v")
            if w in ("be", "have", "do"):
                return False

        return True

    def is_same_word(self, a, b, pos):
        return CONFIG["synonym_lem"].lemmatize(a.lower(), pos=pos) == CONFIG[
            "synonym_lem"
        ].lemmatize(b.lower(), pos=pos)

    def check_if_present(self, word, word_list, pos):
        return any(self.is_same_word(word, w, pos) for w in word_list)

    def convert_to_pos(self, tag):
        if tag in ("JJ", "JJR", "JJS"):
            return "a"
        if tag in ("NN", "NNS", "NNP", "NNPS"):
            return "n"
        if tag in ("RB", "RBR", "RBS"):
            return "r"
        if tag in ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"):
            return "v"
        else:
            return None

    def get_context(self, i, tokens, pos):
        if not self.window:
            return pos
        len_sentence = len(tokens)
        return tokens[
            max(0, i - self.window) : min(len_sentence, i + self.window)
        ]

    def warp_sentence(self, sentence, input_text=""):

        cont_sent_token = wt(sentence)
        cont_sent_tag = nltk.pos_tag(cont_sent_token)
        entries = []
        real_index = len(sentence) - 1

        for i in reversed(range(len(cont_sent_token))):
            word = cont_sent_token[i]
            w, t = cont_sent_tag[i]
            context = self.get_context(i, cont_sent_token, t)

            if (context, word) not in self.cache:
                continue

            # Find index in sentence
            while not sentence[real_index:].startswith(word):
                real_index -= 1
                if real_index < 0:
                    raise Exception("Synonym parsing error")

            entries.append(((context, word), real_index))

        for entry in entries:
            cache_entry, index = entry
            if not len(self.cache[cache_entry]):
                continue

            options = self.cache[cache_entry]
            if self.p < 1.0 and random.random() > self.p:
                continue

            word = random.choice(options)
            sentence = (
                sentence[:index]
                + word
                + sentence[index + len(cache_entry[1]) :]
            )

        return sentence


def main():

    Syn = SynonymAttack(p=0.5)
    for line in sys.stdin:
        if "q" == line.rstrip():
            break
        else:
            input_encodings = CONFIG["synonym_tokenizer"](
                line.strip(), return_tensors="pt"
            )
            output = CONFIG["synonym_llm"].generate(
                input_encodings.input_ids,
                GenerationConfig(do_sample=True, max_new_tokens=256),
            )
            txt = CONFIG["synonym_tokenizer"].batch_decode(
                output, skip_special_tokens=True
            )[0]
            # print(Syn.score(CONFIG['synonym_llm'], encodings))
            print(Syn.warp(txt, input_encodings))
