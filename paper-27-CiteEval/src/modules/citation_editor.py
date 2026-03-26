"""Load CiteEval - Citation Editor.
"""
import re
from collections import defaultdict
from common_utils.eval_utils import load_model
from common_utils.logging_utils import get_logger
from common_utils.templates import ConfigurableTemplatedExample
from src.common_utils.eval_utils import run_api_predictions

logger = get_logger(__name__)


class CitationEditor:
    def __init__(self, model, max_len, max_request, citeeval_config, template_config_dir, verbose=False, n_threads=1) -> None:
        self.config = citeeval_config

        self.edit2rating = self.config.get("edit2rating", None)
        self.allowed_edits = self.config.get("allowed_edits", None)
        
        self.model = self.get_model(model_name=model, max_len=max_len)
        self.max_request = max_request

        self.template_config_dir = template_config_dir
        self.template_config_name = self.config["prompt_templates"]["ce"]

        self.citation_xml_tag = 'citation_rating'
        self.generation_config = {'temperature': 0.0, 'top_p': 1.0}
        self.verbose = verbose
        self.n_threads = n_threads
        
    def get_model(self, model_name, max_len):
        model = load_model(model_name, max_new_tokens=max_len)

        return model

    def build_example(self, params):
        template = ConfigurableTemplatedExample(self.template_config_name, config_dir=self.template_config_dir, **params)
        example_input = template.render()[0]
        if self.verbose:
            logger.info(f'Prompt: {example_input}')
        
        example = {"input": example_input, "id": "", "answers": ""}
        return example
        
    def parse_edits(self, response):
        """Parse sentence-level edits from response.
        Example: 
            <editing sentence_id=1> 
                <DELETE citation=4> DELETE REASON 2 </DELETE> 
                <ADD citation=5> ADD REASON 3 </ADD> 
                <ADD citation=7> ADD REASON 1 </ADD> 
            </editing>
        """
        sentence_edit_matches = []
        sent_id2edits = defaultdict(list)

        sentence_level_regex = f'<editing sentence_id="(\d+)">[\s\n]*((.|\n)*?)[\s\n]*<\/editing>'
        citation_level_regex = f'<(DELETE|ADD) citation="(\d+)">[\s\n]*(?:DELETE|ADD)*\s*REASON\s*(\d+)[\s\n]*<\/(?:DELETE|ADD)>'

        sentence_edit_matches = re.findall(sentence_level_regex, response)
        if not sentence_edit_matches:
            return {}

        for sentence_edit_output in sentence_edit_matches:
            sent_id = sentence_edit_output[0]
            sentence_edit_text = sentence_edit_output[1]

            edit_matches = re.findall(citation_level_regex, sentence_edit_text)
            if not edit_matches:
                continue

            for edit_match in edit_matches:
                assert len(edit_match) == 3, f"Invalid edit_match: {edit_match}"
                op, citation, reason = edit_match  # edit_match example: ('ADD', '1', '3')

                edit = f"{op}_{reason}"
                assert edit in self.allowed_edits, f"Invalid edit: {edit} from edit_match: {edit_match}"

                edit = {
                    "edit": edit,
                    "citation": citation
                }
                sent_id2edits[sent_id].append(edit)
        
        return sent_id2edits
    
    def parse_ratings(self, response):
        """Parse sentence-level ratings from response.
        """
        regex = f'<rating sentence_id="(\d+)">[\s\n]*(\d+)[\s\n]*<\/rating>'
        sent_id2rating = {}

        sentence_rating_matches = re.findall(regex, response)
        if not sentence_rating_matches:
            return {}
        
        for sentence_rating_output in sentence_rating_matches:
            sent_id = sentence_rating_output[0]
            if sentence_rating_output[1].isdigit():
                sent_rating = int(sentence_rating_output[1])
                sent_id2rating[sent_id] = sent_rating
        
        return sent_id2rating

    def parse_response(self, response):
        """For each response sentence, parse the model-generated edits and (optionally) ratings.
        """
        sent_id2edits = self.parse_edits(response)
        sent_id2rating = self.parse_ratings(response)

        return sent_id2edits, sent_id2rating

    def run(self, examples, data=None):
        """Generate response for the input dialog.
        """
        examples = [self.build_example(example) for example in examples]
        responses = run_api_predictions(data=examples, model=self.model, generation_config=self.generation_config, max_request=self.max_request, n_jobs=self.n_threads)

        all_res = []
        for idx in range(len(responses)):
            response = responses[idx]["prediction"]

            if self.verbose:
                logger.info(response)

            sent_id2edits, sent_id2rating = self.parse_response(response)

            res = {
                'output': response,
                'sent_id2edits': sent_id2edits,
                "sent_id2rating": sent_id2rating,
            }
            all_res.append(res)

        return all_res