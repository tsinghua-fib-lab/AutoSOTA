import numpy as np
import json
import logging
from pprint import pprint


class DataLoader:
    def __init__(
            self,
            dataset_name: str = "2wiki",
            data_directory: str = "../data/",
            sample_size: int = 1000
    ):
        self.dataset_name = dataset_name
        self.data_directory = data_directory
        self.sample_size = sample_size
        self.logger = logging.getLogger()

    def _load_2wiki(
            self,
    ):
        """
        Read 2wiki dataset from a JSON file and return question and data lists.
        """
        data_file = self.data_directory + '2wiki.json'
        with open(data_file, "r") as f:
            data = json.load(f)

        print(len(data))
        data = data[0:self.sample_size]

        qa_list = [{'question': item['question'], 'answer': item['answer']} for item in data]
        context_list = []
        for item in data:
            context = item['context']
            context_l = [{'title': ct[0], 'text': " ".join(ct[1])} for ct in context]
            context_list.extend(context_l)
        return qa_list, context_list

    def _load_hotpotqa(
            self,
    ):
        """
        Read hotpotqa dataset from a JSON file and return question and data lists.
        """
        data_file = self.data_directory + 'hotpotqa.json'
        with open(data_file, "r") as f:
            data = json.load(f)

        data = data[0:self.sample_size]

        qa_list = [{'question': item['question'], 'answer': item['answer']} for item in data]
        context_list = []
        for item in data:
            context = item['context']
            context_l = [{'title': ct[0], 'text': " ".join(ct[1])} for ct in context]
            context_list.extend(context_l)
        return qa_list, context_list

    def _load_musique(
            self,
    ):
        """
        Read musique dataset from a JSON file and return question and data lists.
        """
        data_file = self.data_directory + 'musique.json'
        with open(data_file, "r") as f:
            data = json.load(f)

        data = data[0:self.sample_size]

        qa_list = [{'question': item['question'], 'answer': item['answer']} for item in data]
        context_list = []
        for item in data:
            context = item['paragraphs']
            context_l = [{'title': ct['title'], 'text': ct['paragraph_text']} for ct in context]
            context_list.extend(context_l)
        return qa_list, context_list

    def load(self):
        match self.dataset_name:
            case "2wiki":
                return self._load_2wiki()
            case "hotpotqa":
                return self._load_hotpotqa()
            case "musique":
                return self._load_musique()
            case _:
                self.logger.error(f"Unknown dataset name: {self.dataset_name}")
                return None, None


if __name__ == "__main__":
    qa, context = DataLoader("2wiki", sample_size=1000).load()
