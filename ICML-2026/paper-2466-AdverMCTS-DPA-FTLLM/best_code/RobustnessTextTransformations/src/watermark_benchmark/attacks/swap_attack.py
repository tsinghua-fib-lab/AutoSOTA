import numpy as np
import re
import random
from .utils import Attack

class SwapAttack(Attack):

    def __init__(self, word_repeat_prob=0.5, word_delete_prob=0.5, word_swap_probability=0.5, sentence_swap_probability=0.5):
        super().__init__("SwapAttack")
        self.word_repeat_prob = word_repeat_prob
        self.word_delete_prob = word_delete_prob
        self.word_swap_probability = word_swap_probability
        self.sentence_swap_probability = sentence_swap_probability
        self.sentence_regex = re.compile("((?![.!?\n\s])[^.!?\n\"]*(?:\"[^\n\"]*[^\n\".!?]\"[^.!?\n\"]*)*(?:\"[^\"\n]+[.!?]\"|\.|\?|!|(?=$)|(?=\n))\s*)")
        self.word_regex = re.compile("([^.!?,:\"\s]+(?=\s|,|\.|:|\"|!|\?))")


    @staticmethod
    def get_param_list():
        params = [(0.05, 0.05, 0, 0), (0, 0, 0.05, 0), (0, 0, 0.1, 0)]
        basename = "SwapAttack_{}_{}_{}_{}"
        return [(basename.format(*p), p) for p in params]

    def warp(self, text, input_encodings=None):
        text = text.replace("</s>", "").replace("<pad>", "")
        sentences=[]
        for g in re.findall(self.sentence_regex, text):
            if len(g):
                sentences.append(g)
        if self.sentence_swap_probability > 0:
            for i in range(len(sentences)-1):
                if random.random() < self.sentence_swap_probability:
                    sentences[i], sentences[i+1] = sentences[i+1], sentences[i]

        if self.word_swap_probability > 0:
            for i in range(len(sentences)):
                words = [(match.group(), match.start(), match.end()) for match in re.finditer(self.word_regex, sentences[i])]
                if len(words) < 2:
                    continue
                delimiters = [sentences[i][:words[0][1]]]
                for j in range(len(words)):
                    if j+1 == len(words):
                        delimiters.append(sentences[i][words[-1][2]:])
                    else:
                        delimiters.append(sentences[i][words[j][2]:words[j+1][1]])
                for j in range(len(words)-1):
                    if random.random() < self.word_swap_probability:
                        words[j], words[j+1] = words[j+1], words[j]
                sentences[i] = delimiters[0]
                for j in range(len(words)):
                    sentences[i] += words[j][0] + delimiters[j+1]

        text = "".join(sentences)
        words = [(match.group(), match.start(), match.end()) for match in re.finditer(self.word_regex, text)]
        if not len(words):
            return text
        delimiters = [text[:words[0][1]]]
        for j in range(len(words)):
            if j+1 == len(words):
                delimiters.append(text[words[-1][2]:])
            else:
                delimiters.append(text[words[j][2]:words[j+1][1]])

        if self.word_delete_prob > 0:
            for i in range(len(words)):
                if random.random() < self.word_delete_prob:
                    words[i] = ("",words[i][1], words[i][2])
        
            text = delimiters[0]
            for i in range(len(words)):
                text += words[i][0] + delimiters[i+1]

            words = [(match.group(), match.start(), match.end()) for match in re.finditer(self.word_regex, text)]
            if len(words) > 1:
                delimiters = [text[:words[0][1]]]
                for j in range(len(words)):
                    if j+1 == len(words):
                        delimiters.append(text[words[-1][2]:])
                    else:
                        delimiters.append(text[words[j][2]:words[j+1][1]])

        if self.word_repeat_prob > 0:
            i = len(words)-1
            while i >= 0:
                if random.random() < self.word_repeat_prob:
                    words.insert(i, words[i])
                    delimiters.insert(i+1, " ")
                i-=1

            text = delimiters[0]
            for i in range(len(words)):
                text += words[i][0] + delimiters[i+1]

        return text.replace("  "," ")
