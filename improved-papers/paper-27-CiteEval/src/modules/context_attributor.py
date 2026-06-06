"""Load CiteEval - Context Attributor.
"""
import re
from common_utils.eval_utils import load_model, run_api_predictions
from common_utils.logging_utils import get_logger
from common_utils.templates import ConfigurableTemplatedExample
from collections import defaultdict

logger = get_logger(__name__)


class ContextAttributor:
    def __init__(
            self, 
            model, max_len, max_request, 
            citeeval_config,
            template_config_dir,
            verbose=False, 
            n_threads=1
        ) -> None:
        
        self.config = citeeval_config

        self.model = self.get_model(model_name=model, max_len=max_len)
        self.max_request = max_request
        self.template_config_dir = template_config_dir
        self.template_config_name = self.config["prompt_templates"]["ca"]
        self.context_types = self.config["context_types"]
        self.context_types2citation_requirement = self.config["context_types2citation_requirement"]

        self.default_context_type = self.context_types["default"]
        assert self.default_context_type in self.context_types, f"Default context type not found in context types: {self.default_context_type}"

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
    
    def parse_response(self, response):
        """Parse a give response into a mapping from sentence id to context type.
        Example: 
            <category sentence_id="1"> 2 </category>
        """
        sent_id2type = defaultdict(str)

        sentence_level_regex = f'<category sentence_id="(\d+)">[\s\n]*(\d+)[\s\n]*<\/category>'

        sentence_type_matches = re.findall(sentence_level_regex, response)
        for sentence_type_output in sentence_type_matches:
            sent_id = sentence_type_output[0]
            category = sentence_type_output[1]

            if category not in self.context_types:
                category = category[0]  # the first char from "3. STATEMENT_TYPE"
            
            if category not in self.context_types:
                category = self.default_context_type
                print(f"Invalid statement category: {category} from sentence: {sent_id}. Replaced with default category: {self.default_context_type}")
            
            # assert category.isdigit() and 1 <= int(category) <= 5, f"Invalid statement category: {category} from sentence: {sent_id}"
            sent_id2type[sent_id] = category
        
        return sent_id2type

    def run(self, examples, validate_optional=False, data=None):
        """Generate response for the input dialog.
        """
        examples = [self.build_example(example) for example in examples]
        responses = run_api_predictions(data=examples, model=self.model, generation_config=self.generation_config, max_request=self.max_request, n_jobs=self.n_threads)

        all_res = []
        for idx in range(len(responses)):
            response = responses[idx]["prediction"]
            sent_info = data[idx]['sent_info']

            if self.verbose:
                logger.info(response)

            sent_id2pred = self.parse_response(response)

            sent_id2type = {}
            for sent_idx in range(len(sent_info)):
                raw_ca_pred = sent_id2pred.get(str(sent_idx+1), "")
                ca_pred = raw_ca_pred
                
                if raw_ca_pred == "":
                    ca_pred = self.default_context_type
                
                context_type = {
                    "raw_ca_pred": raw_ca_pred,
                    "ca_pred": ca_pred
                }
                sent_id2type[str(sent_idx+1)] = context_type
            
            res = {
                'output': response,
                'sent_id2type': sent_id2type
            }
            all_res.append(res)

        return all_res
