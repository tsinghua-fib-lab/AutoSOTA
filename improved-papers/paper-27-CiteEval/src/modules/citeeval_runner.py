from collections import Counter
from src.modules.citeeval_loader import CiteEvalLoader
from modules.format_utils import construct_span, build_sentence_xml, build_passage_xml, build_citation_xml
from data.split_response_into_sentences_and_citations import is_no_answer_prediction


class CiteEvalRunner:
    def __init__(self, data, version, citing_sentences_only, n_threads, model_name):
        self.data = data
        self.version = version
        self.citing_sentences_only = citing_sentences_only
        self.n_threads = n_threads
        self.model_name = model_name

        self.citeeval_loader = CiteEvalLoader(version=version, model_name=model_name, n_threads=n_threads)

    def run_citeeval_ca(self, context_attributor):
        """Run context attribution.
        """
        citeeval_ca_inputs = []
        n_sentences = 0.0

        for item in self.data:
            sent_info = item['sent_info']
            
            n_sentences += len(sent_info)
            span_info = construct_span(sent_info, citeeval_config=context_attributor.config)
            response_unit_name = context_attributor.config.get("response_unit_name", "sentence")
            ca_use_citations = context_attributor.config.get("ca_use_citations", True)

            sentences = build_sentence_xml(span_info, response_unit_name=response_unit_name, use_citations=ca_use_citations)
            passages = build_passage_xml(item, unit_name="passage")

            example = {
                'question': item['question'],
                'passages': passages,
                'sentences': sentences,
            }

            for key in ("id", "human_attribution_judgement"):
                if key in item:
                    example[key] = item[key]
            citeeval_ca_inputs.append(example)

        citeeval_ca_preds = context_attributor.run(citeeval_ca_inputs, data=self.data)
        
        citeeval_ca_outputs = []
        
        raw_ca_distribution = Counter()
        ca_distribution = Counter()
        
        for res, example in zip(citeeval_ca_preds, citeeval_ca_inputs):
            raw_ca_distribution.update([pred["raw_ca_pred"] for pred in res['sent_id2type'].values()])
            ca_distribution.update([pred["ca_pred"] for pred in res['sent_id2type'].values()])

            example['output'] = res['output']
            example['sent_id2type'] = res['sent_id2type']
            citeeval_ca_outputs.append(example)

        citeseval_ca_res = {
            'n_samples': len(citeeval_ca_outputs),
            'n_sentences': n_sentences,
            'raw_ca_distribution': raw_ca_distribution,
            'ca_distribution': ca_distribution,
        }
        
        return citeeval_ca_outputs, citeseval_ca_res

    def run_citeeval_ce(self, citation_editor):
        """Run citation editing. 

        Return:
            citeeval_ce_outputs: list of jsons, where each json consists of citeeval model inputs and outputs
            citeeval_ce_res: json, aggregated results on citeeval metrics
        Explanation for some of the keys in citeeval_res:
            no_citation_rate_span: % of spans that do not have citation
            uncited_spans_rate_response: % of responses that contain at least one uncited span
            no_citation_rate_response: % of responses that do not contain citations at all, and are not "no answer" responses
            no_citation_rate_response_raw: % of responses that do not contain citations at all, irrespective of whether the response is "no answer"
        """
        citeeval_ce_inputs = []
        n_spans = 0
        no_cite_spans = 0
        responses_with_uncited_span = 0
        no_cite_responses_raw = 0
        no_cite_responses = 0
        n_samples = len(self.data)

        for item in self.data:
            sent_info = item['sent_info']

            span_info = construct_span(
                sent_info, 
                citeeval_config=citation_editor.config
            )
            response_unit_name = citation_editor.config.get("response_unit_name", "sentence")
            citations = build_citation_xml(span_info=span_info, response_unit_name=response_unit_name)
            passages = build_passage_xml(item, unit_name="passage")

            # calculate citation stats
            no_cite_spans += len([span for span in span_info if len(span['citations']) == 0])
            responses_with_uncited_span += 1 if any([span for span in span_info if len(span['citations']) == 0]) else 0
            n_spans += len(span_info)
            no_cite_responses_raw += 1 if all([len(span['citations']) == 0 for span in span_info]) else 0

            example = {
                'question': item['question'],
                'passages': passages,
                'answer': ' '.join([span["span"] for span in span_info]),
                'citations': citations,
            }

            no_cite_responses += 1 if all([len(span['citations']) == 0 for span in span_info]) and not is_no_answer_prediction(example['answer']) else 0

            for key in ("id", "human_attribution_judgement"):
                if key in item:
                    example[key] = item[key]
            citeeval_ce_inputs.append(example)

        citeeval_ce_preds = citation_editor.run(citeeval_ce_inputs, data=self.data)
        edit_distribution = Counter()
        citeeval_ce_outputs = []

        res_keys = ["output", "sent_id2edits", "sent_id2rating", "sent_id2sufficiency"]
        for res, example in zip(citeeval_ce_preds, citeeval_ce_inputs):
            """
                res = {
                    'output': response,
                    'sent_id2edits': sent_id2edits,
                    "sent_id2rating": sent_id2rating
                }
            """
            all_edits = []
            for sent_edits in res['sent_id2edits'].values():
                for edit in sent_edits:
                    all_edits.append(edit['edit'])

            edit_distribution.update(all_edits)
            for k in res_keys:
                if k not in res:
                    continue
                example[k] = res[k]

            citeeval_ce_outputs.append(example)

        citeeval_ce_res = {
            "edit_distribution": edit_distribution,
            'no_citation_rate.sentence': no_cite_spans / n_spans,
            'no_citation_rate.response': no_cite_responses / n_samples,
            'no_citation_rate.response_raw': no_cite_responses_raw / n_samples,
            'no_full_citation_rate.response': responses_with_uncited_span / n_samples
        }

        return citeeval_ce_outputs, citeeval_ce_res

    def run_citeeval_cr(self, citation_rater):
        """Run citation rating.
        """
        citeeval_cr_preds = citation_rater.run(data=self.data)
        sentence_rating_distribution = Counter()

        response_ratings = []
        for res in citeeval_cr_preds:
            """
                res = {
                    "sent_id2type": sent_id2type,
                    "sent_id2model_rating": sent_id2model_rating,
                    "sent_id2rating": sent_id2rating,
                    "answer_rating": answer_rating
                }
            """
            sentence_ratings = list(res['sent_id2rating'].values())
            sentence_rating_distribution.update(sentence_ratings)
            response_ratings.append(res['answer_rating'])

        response_rating_distribution = Counter(response_ratings)
        
        assert len(response_ratings) > 0
        response_rating = sum(response_ratings) / float(len(response_ratings))
        citeeval_cr_res = {
            "response_rating": response_rating,
            "response_rating_distribution": response_rating_distribution,
            "sentence_rating_distribution": sentence_rating_distribution,
        }
        return citeeval_cr_preds, citeeval_cr_res

    def run(self, module_name=None, ca_output_file=None, ce_output_file=None):
        """When citing_sentences_only is True, only sentences with citations will be evaluated.
        """
        n_cites = 0.0
        n_sentence, n_citing_sentence = 0.0, 0.0

        for i, item in enumerate(self.data):
            n_sentence += len(item['sent_info'])
            citing_sentence_info = []
            for sinfo in item['sent_info']:
                _nc = len(sinfo['citations'])
                n_cites += _nc
                if _nc > 0:
                    citing_sentence_info.append(sinfo)

            n_citing_sentence += len(citing_sentence_info)
            if self.citing_sentences_only:
                self.data[i]['sent_info'] = citing_sentence_info
        
        citeeval_module = self.citeeval_loader.load_module(
            module_name=module_name,
            ca_output_file=ca_output_file, 
            ce_output_file=ce_output_file,
        )

        if module_name == "ca":
            citeeval_outputs, citeeval_res = self.run_citeeval_ca(context_attributor=citeeval_module)
        
        elif module_name.startswith("ce"):
            citeeval_outputs, citeeval_res = self.run_citeeval_ce(citation_editor=citeeval_module)
        
        elif module_name.startswith("cr"):
            citeeval_outputs, citeeval_res = self.run_citeeval_cr(citation_rater=citeeval_module)
        
        else:
            raise ValueError(f"Invalid module name: {module_name}")
        
        results = {}
        metrics = []
        for key, value in citeeval_res.items():
            metric = f"{key}.{self.version}"
            results[metric] = value
            metrics.append(metric)
        
        if self.citing_sentences_only:
            citing_only_affix = 'citing_only'
            metrics = list(results.keys())
            for key in metrics: 
                results[f'{key}_{citing_only_affix}'] = results.pop(key)

        results['average_num_citations_per_sentence'] = n_cites / n_sentence if n_sentence else 0.0
        results['average_num_citations_per_sentence_citing_only'] = n_cites / n_citing_sentence if n_citing_sentence else 0.0
        return results, citeeval_outputs
