"""Implement paraphrase attack."""

from __future__ import annotations

import logging
import os
import pprint
import shutil
import sys
import time
from multiprocessing import Queue

import argostranslate.package
import argostranslate.translate
import nltk
import numpy as np
import requests
import torch
from lingua import Language, LanguageDetectorBuilder
from nltk.tokenize import sent_tokenize
from transformers import T5ForConditionalGeneration, T5Tokenizer

from watermark_benchmark.utils.apis import call_openai, openai_process

if __name__ == "__main__":

    class Attack:
        """Dummy Attack class for testing."""

        def __init__(self, name: str) -> None:
            """Initialize Attack."""
            self.name = name

else:
    from watermark_benchmark.attacks.utils import Attack


logger = logging.getLogger(__name__)


class ResponseError(Exception):
    """Raised when online API returns an error response."""


def loop(f):
    retry = 0
    while retry < 3:
        try:
            return f()
        except openai.InvalidRequestError as e:
            return None
        except Exception as e:
            print(e)
            time.sleep(1)
            retry += 1
            continue
    return None


class ParaphraseAttack(Attack):
    """Paraphrase attack using automatic tools."""

    def __init__(
        self,
        paraphrase_method: str = "default",
        hf_model: str | None = None,
        openai_model: str | None = None,
        dipper_model: bool = False,
        use_google_translate: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        translate_lang: str = "French",
        queue: dict | None = None,
        resp_queue: Queue | None = None,
    ) -> None:
        """Initialize ParaphraseAttack.

        Args:
            paraphrase_method: Choice of paraphrase method to use as an attack.
                Defaults to "default".
            hf_model: HuggingFace model to use for paraphrasing. Example:
                "google/t5-v1_1-base". Defaults to None.
            use_google_translate: If True, use Google Translate for paraphrasing
                by back-translation. Defaults to False.
            openai_model: OpenAI model to use for paraphrasing. Options are
                "gpt-4", "gpt-3.5-turbo", and models in `_MAX_TOKENS_BY_MODEL`.
                Defaults to None.
            translate_lang: Language to translate to and back from. Defaults to
                "French".

        Raises:
            NotImplementedError: Paraphrase tool not available.
        """
        super().__init__("ParaphraseAttack")
        self.available_paraphrase_methods = ["default", "translate"]
        if paraphrase_method not in self.available_paraphrase_methods:
            raise NotImplementedError(
                f"Paraphrase tool {paraphrase_method} not available! "
                f"Available tools: {self.available_paraphrase_methods}"
            )
        self._paraphrase_method = paraphrase_method
        self._hf_model = hf_model
        self._openai_model = openai_model
        self._dipper = dipper_model
        self._use_google_translate = use_google_translate

        # Sampling params
        self._temperature = temperature
        self._top_p = top_p
        # OpenAI params
        self._presence_penalty = presence_penalty
        self._frequency_penalty = frequency_penalty
        self._translate_lang = translate_lang
        if self._openai_model is not None:
            self._is_chat = "gpt" in self._openai_model

        # Only one model can be used at a time
        models = [hf_model, openai_model, use_google_translate, dipper_model]
        num_models = sum(
            int(model is not None and model is not False) for model in models
        )
        if num_models != 1:
            # models must be exclusive
            raise ValueError(
                f"Only one model can be used at a time ({num_models} given)!"
            )

        # Reading API key from file
        self._hf_api_key: str = os.getenv("HF_API_KEY")

        self._dipper_initialized = False

        # Setup multiprocessing queues
        self._queue = queue
        self._resp_queue = resp_queue

        if self._use_google_translate and queue is None:
            self._init_translate(translate_lang)

    def _init_translate(self, lang):
        from_code = "en"
        to_code = lang
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        package_to_install = next(
            filter(
                lambda x: (x.from_code == to_code and x.to_code == from_code),
                available_packages,
            )
        )
        argostranslate.package.install_from_path(package_to_install.download())
        package_to_install = next(
            filter(
                lambda x: (x.from_code == from_code and x.to_code == to_code),
                available_packages,
            )
        )
        argostranslate.package.install_from_path(package_to_install.download())

    def _init_dipper(self):
        if self._dipper and not self._dipper_initialized:
            self._dipper_initialized = True
            # Setting up DIPPER if needed
            self.tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
            try:
                self.model = T5ForConditionalGeneration.from_pretrained(
                    "kalpeshk2011/dipper-paraphraser-xxl"
                ).cuda()
                self.device = "cuda"
            except:
                self.model = T5ForConditionalGeneration.from_pretrained(
                    "kalpeshk2011/dipper-paraphraser-xxl"
                )
                self.device = "cpu"
            self.model.eval()

    def _get_prompt(self, back_translate: bool) -> str:
        """Get prompt for the paraphrase model."""
        # TODO(feature): Add options to prompt
        if self._paraphrase_method == "default":
            return "Paraphrase the following text: "
        if self._paraphrase_method == "translate":
            lang = "English" if back_translate else self._translate_lang
            return f"Translate the following text to {lang}: "
        raise NotImplementedError(
            f"Paraphrase tool {self._paraphrase_method} not available! "
            f"Available tools: {self.available_paraphrase_methods}"
        )

    def _query_hf(self, text: str, back_translate: bool = False) -> str:
        url = f"https://api-inference.huggingface.co/models/{self._hf_model}"
        logger.debug("Querying Hugging Face API at %s...", url)
        headers = {"Authorization": f"Bearer {self._hf_api_key}"}
        payload = {"inputs": f"{self._get_prompt(back_translate)}{text}"}
        num_trials = 0
        # Have to retry because sometimes HuggingFace model has to be loaded
        while True:
            response = requests.post(
                url, headers=headers, json=payload, timeout=60
            )
            try:
                response = response.json()
                logger.warning("Given payload: %s", payload)
                logger.warning("Response: %s", response.text)
            except requests.JSONDecodeError:
                logger.error("Response: %s", response.text)
                raise
            if (
                "error" in response
                and "is currently loading" in response["error"]
            ):
                # Wait 50% longer than the estimated time
                wait_time = int(response["estimated_time"] * 1.5)
                logger.info(
                    "%s, waiting %d seconds", response["error"], wait_time
                )
                if wait_time > 60 or num_trials >= 3:
                    raise ResponseError(
                        f"Model took too long to load ({num_trials} retries)!"
                    )
                time.sleep(wait_time)
                num_trials += 1
            elif isinstance(response, list) and "generated_text" in response[0]:
                break
            else:
                raise ResponseError(
                    f"Failed response from HuggingFace API!\n{response}"
                )
        return response[0]["generated_text"]

    def _query_openai(self, text: str, back_translate: bool = False) -> str:
        logger.debug("Querying OpenAI API (%s)...", self._openai_model)

        prompt = f"{self._get_prompt(back_translate)}{text}"
        system_prompt = "You are a helpful assistant."

        if not self._queue:
            completion = call_openai(
                self._openai_model,
                self._temperature,
                self._top_p,
                self._presence_penalty,
                self._frequency_penalty,
                prompt,
                system_prompt,
                1024,
                10,
                None,
                False,
            )
        else:
            self._queue["openai"].put(
                (
                    self._openai_model,
                    self._temperature,
                    self._top_p,
                    self._presence_penalty,
                    self._frequency_penalty,
                    prompt,
                    system_prompt,
                    1024,
                    10,
                    None,
                    False,
                    self._resp_queue,
                )
            )
            completion = self._resp_queue.get(block=True)

        if completion is None:
            return text

        if "gpt" in self._openai_model:
            answer = completion.choices[0].message.content
        else:
            answer = completion.choices[0].text
        answer.strip()

        finish_reason = completion.choices[0].finish_reason
        if finish_reason != "stop":
            logger.warning(
                'finish_reason is "%s" instead of "stop". Something went wrong.',
                finish_reason,
            )

        if self._paraphrase_method == "translate" and not back_translate:
            # Query again to translate back to English
            # NOTE: We separate this into two queries so the model cannot
            # depend on the original English text for back translation.
            logger.debug("Text in %s:\n%s", self._translate_lang, answer)

            # Try to detect language of current answer
            target_lang = Language[self._translate_lang.upper()]
            detector = LanguageDetectorBuilder.from_languages(
                Language.ENGLISH, target_lang
            ).build()
            detected_lang = detector.detect_language_of(answer)
            if detected_lang != target_lang:
                logger.warning(
                    "Detected language is %s, not %s. Try another language next"
                    " time. Still proceed anyway...",
                    detected_lang,
                    target_lang,
                )

            answer = self._query_openai(answer, back_translate=True)
            return answer.strip()

        # TODO(feature): Sanitize answer?
        return answer.strip()

    def _query_google_translate(
        self, text: str, back_translate: bool = False
    ) -> str:
        lang = "en" if back_translate else self._translate_lang
        from_lang = self._translate_lang if back_translate else "en"

        if self._queue is None:
            answer = argostranslate.translate.translate(text, from_lang, lang)
        else:
            self._queue["translate"].put(
                (text, from_lang, lang, self._resp_queue)
            )
            answer = self._resp_queue.get(block=True)

        if not back_translate:
            logger.debug("Text in %s:\n%s", self._translate_lang, answer)
            answer = self._query_google_translate(answer, back_translate=True)
        return answer

    def _run_dipper(
        self,
        text: str,
        lex_diversity=40,
        order_diversity=60,
        prefix="",
        sent_interval=4,
        **kwargs,
    ):
        """Paraphrase a text using the DIPPER model.

        Args:
        input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
        lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
        order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
        **kwargs: Additional keyword arguments like top_p, top_k, max_length.
        """
        assert lex_diversity in [
            0,
            20,
            40,
            60,
            80,
            100,
        ], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [
            0,
            20,
            40,
            60,
            80,
            100,
        ], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        if self._queue is None:
            self._init_dipper()

        input_text = " ".join(text.split())
        sentences = sent_tokenize(input_text)
        output_text = ""
        queries = []

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(
                sentences[sent_idx : sent_idx + sent_interval]
            )
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"
            queries.append(final_input_text)

        if self._queue is None:
            # Not supported
            raise NotImplementedError(
                "Dipper is only accessible through the API"
            )
        else:
            self._queue["dipper"].put((queries, self._resp_queue))
            output = self._resp_queue.get(block=True)

        return output

    @staticmethod
    def get_param_list(reduced=False):
        """See Attack.get_param_list."""
        if not reduced:
            return [
                (
                    "ParaphraseAttack_GPT3.5",
                    (
                        "default",
                        None,
                        "gpt-3.5-turbo",
                        False,
                        False,
                        1.0,
                        1.0,
                        None,
                        None,
                        "fr",
                    ),
                ),
                #(
                #    "TranslationAttack_French",
                #    (
                #        "translate",
                #        None,
                #        None,
                #        False,
                #        True,
                #        1.0,
                #        1.0,
                #        None,
                #        None,
                #        "fr",
                #    ),
                #),
                #(
                #    "TranslationAttack_Russian",
                #    (
                #        "translate",
                #        None,
                #        None,
                #        False,
                #        True,
                #        1.0,
                #        1.0,
                #        None,
                #        None,
                #        "ru",
                #    ),
                #),
                (
                    "ParaphraseAttack_Dipper",
                    (
                        "default",
                        None,
                        None,
                        True,
                        False,
                        1.0,
                        1.0,
                        None,
                        None,
                        "fr",
                    ),
                ),
            ]
        else:
            return [
                (
                    "TranslationAttack_French",
                    (
                        "translate",
                        None,
                        None,
                        False,
                        True,
                        1.0,
                        1.0,
                        None,
                        None,
                        "fr",
                    ),
                ),
                (
                    "TranslationAttack_Russian",
                    (
                        "translate",
                        None,
                        None,
                        False,
                        True,
                        1.0,
                        1.0,
                        None,
                        None,
                        "ru",
                    ),
                ),
            ]

    def warp(self, text: str, input_encodings=None) -> str:
        """See Attack.warp."""
        _ = input_encodings  # Unused
        # TODO(feature): Break up text into smaller chunks if longer than
        # context windows size.
        if self._openai_model is not None:
            return self._query_openai(text)
        if self._hf_model is not None:
            return self._query_hf(text)
        if self._use_google_translate:
            return self._query_google_translate(text)
        if self._dipper:
            return self._run_dipper(text)
        raise NotImplementedError("No model specified!")


# _TEST_TEXT = """
# Sad Frog is a cartoon drawing of a depressed-looking frog, often accompanied by the text "Feels Bad Man" or "You Will Never X". It is used to denote feelings of failure or disappointment, either by posting the image or using the phrase "feelsbadman.jpg." Sad Frog may be seen as the antithesis of Feels Good Man.
# """
# _TEST_TEXT = """What is the purpose of life?"""
# _TEST_TEXT = "Bob is competing in a hotdog-eating competition and probably wins the first prize."
_TEST_TEXT = """
AI systems with human-competitive intelligence can pose profound risks to society and humanity, as shown by extensive research and acknowledged by top AI labs. As stated in the widely-endorsed Asilomar AI Principles, Advanced AI could represent a profound change in the history of life on Earth, and should be planned for and managed with commensurate care and resources. Unfortunately, this level of planning and management is not happening, even though recent months have seen AI labs locked in an out-of-control race to develop and deploy ever more powerful digital minds that no one – not even their creators – can understand, predict, or reliably control.
"""
_TEST_TEXT = _TEST_TEXT.strip()


def test():
    """Test ParaphraseAttack. Now just testing google translate"""
    attack = ParaphraseAttack(
        *("translate", None, None, False, True, 1.0, 1.0, None, None, "ru")
    )
    print(_TEST_TEXT)
    print(attack.warp(_TEST_TEXT))


if __name__ == "__main__":
    test()
