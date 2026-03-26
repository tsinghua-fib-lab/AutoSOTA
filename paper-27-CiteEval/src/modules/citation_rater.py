"""Load CiteEval - Citation Rater.
"""
import io
import json
from tqdm import tqdm
from common_utils.logging_utils import get_logger
from common_utils.rating_utils import rate

logger = get_logger(__name__)


class BaseCitationRater:
    def __init__(
            self, 
            citeeval_config,
            ca_output_file, 
            ce_output_file,
            verbose=False
        ) -> None:
        self.config = citeeval_config
        self.prompt_version = self.config.get("prompt_version", "")

        self.ca_output_file = ca_output_file
        self.ce_output_file = ce_output_file
        self.verbose = verbose

        with io.open(self.ce_output_file) as f:
            self.ce_output = json.load(f)

        if self.ca_output_file:
            with io.open(self.ca_output_file) as f:
                self.ca_output = json.load(f)
            assert len(self.ca_output) == len(self.ce_output), f"CA and CE outputs have different #record. CA: {len(self.ca_output)}, CE: {len(self.ce_output)}"
        else:
            self.ca_output = None

        self.context_types2citation_requirement = self.config["context_types2citation_requirement"]
    
    def rate(self, sent_info, sent_id2type, sent_id2model_rating, normalized_model_rating=False):
        return rate(
            sent_info=sent_info, 
            sent_id2type=sent_id2type, 
            sent_id2model_rating=sent_id2model_rating,
            context_types2citation_requirement=self.context_types2citation_requirement,
            normalized_model_rating=normalized_model_rating
        )


class LightCitationRater(BaseCitationRater):
    """
    Citation Rater based on citation ratings generated jointly with citation edits from Citation Editor.
    """
    def __init__(
            self, 
            citeeval_config,
            ca_output_file,
            ce_output_file, 
            verbose) -> None:
        """
        rating_algo: 
            - rule: rule based rating with pre-defined citation-level distance
            - joint: ce_output_file contains ratings jointly predicted with citation edits
            - posthoc: perform a separate LLM call for citation rating, based on citation edits
        """
        super().__init__(
            citeeval_config=citeeval_config,
            ca_output_file=ca_output_file, 
            ce_output_file=ce_output_file,
            verbose=verbose
        )

    def run(self, data):
        """Calculate citation ratings for each example. 
        
        Each sample requires:
            - sent_id2type, from ca output
            - sent_id2edits, from ce output
            - sent_id2rating, from ce output
        """
        all_res = []
        for idx in tqdm(range(len(data))):
            
            sent_info = data[idx]['sent_info']
            sent_id2type = self.ca_output[idx]["sent_id2type"]
            sent_id2edits = self.ce_output[idx]["sent_id2edits"]
            assert self.ca_output[idx]["id"] == self.ce_output[idx]["id"]

            assert "sent_id2rating" in self.ce_output[idx]
            sent_id2model_rating = self.ce_output[idx]["sent_id2rating"]

            res = self.rate(
                sent_info=sent_info, 
                sent_id2type=sent_id2type, 
                sent_id2model_rating=sent_id2model_rating
            )

            res = {
                "id": self.ce_output[idx]["id"],
                **res
            }
            all_res.append(res)

        return all_res


class DistanceBasedCitationRater(BaseCitationRater):
    """Citation Rater based on edit distance and heuristic assignment of citation distances.
    """
    def __init__(
            self, 
            citeeval_config, 
            ca_output_file, 
            ce_output_file, 
            distance_type,
            verbose
        ) -> None:
        super().__init__(
            citeeval_config=citeeval_config,
            ca_output_file=ca_output_file, 
            ce_output_file=ce_output_file, 
            verbose=verbose
        )
        
        self.allowed_edits = self.config.get("allowed_edits", None)

        self.distance_type = distance_type
        if distance_type == "estimated":
            self.edit2rating = self.config.get("edit2esitamted_rating", None)
        elif distance_type == "heuristic":
            self.edit2rating = self.config.get("edit2heuristic_rating", None)
        else:
            raise ValueError(f"Invalid distance type: {distance_type}")

    def get_citation2rating_for_sentence(self, sent_info, edits):
        def _init_citation_rating():
            if self.distance_type == "estimated":
                init_rating = self.edit2rating["KEEP"]
            elif self.distance_type == "heuristic":
                init_rating = 1.0
            else:
                raise ValueError(f"Invalid distance type: {self.distance_type}")
            
            return {
                "rating": init_rating,
                "deleted": False,
                "added": False
            }
        
        citation2rating = {}  # related citations for the current sentence
        for cite in sent_info["citations"]:
            # citations in sent_info start from 0; add 1 to be consistent with doc ID
            citation2rating[str(cite+1)] = _init_citation_rating()

        for edit in edits:
            op, _ = edit["edit"].split("_")
            citation = edit["citation"]
            if op == "DELETE":
                if citation not in citation2rating:  # skip if the citation to delete does not exist
                    continue
                
                # remove if the citation was added then deleted
                # the citation to delete should not be added in the first place
                if citation2rating[citation]["added"]:  
                    del citation2rating[citation]
                    continue

                assert edit["edit"] in self.allowed_edits, f"Invalid edit: {edit['edit']}"
                citation2rating[citation] = {
                    "rating": self.edit2rating[edit["edit"]],
                    "deleted": True,
                    "added": False
                }
            
            elif op == "ADD":
                if citation in citation2rating:
                    # re-init if the citation was deleted and then added back
                    # the citation to add should not be deleted in the first place
                    if citation2rating[citation]["deleted"]:
                        citation2rating[citation] =  _init_citation_rating()
                    
                    # otherwise, skip it
                    # the citation to add already exist in the original set
                    continue

                citation2rating[citation] =  {
                    "rating": self.edit2rating[edit["edit"]],
                    "deleted": False,
                    "added": True
                }
            else:
                raise ValueError(f"Invalid op: {edit['edit']}")
        
        return citation2rating


    def rate_with_heuristics(self, sent_id2edits, sent_info, min_rating=0.0, max_rating=1.0):
        """Heuristics to convert provided edits into a rating between 0 and 1.

        Return sent_id2model_rating, rating (1-5)
        """
        
        def _aggregate_citation_ratings(citation2rating):
            """Return sentence-level rating via averaging citation-level ratings. 
            """
            if len(citation2rating) == 0:
                return max_rating
            
            if self.distance_type == "estimated":
                total = 0.0
                for rating in citation2rating.values():
                    total += rating["rating"]

                    if rating["deleted"] or rating["added"]:
                        all_keep = False
                
                total = total / len(citation2rating) + self.edit2rating["INTERCEPT"]  # start with the bias term
                return total

            elif self.distance_type == "heuristic":
                total = 0.0
                for rating in citation2rating.values():
                    total += rating["rating"]
                
                return total / len(citation2rating)
            else:
                raise ValueError(f"Invalid distance type: {self.distance_type}")
            
        
        sent_id2model_rating = {}
        for sid, sinfo in enumerate(sent_info):
            sid = str(sid+1)
            if sid not in sent_id2edits:  # no editing needed
                rating = max_rating

            else:
                edits = sent_id2edits[sid]
                citation2rating = self.get_citation2rating_for_sentence(
                    sent_info=sinfo,
                    edits=edits
                )

                rating = _aggregate_citation_ratings(citation2rating)

            sent_id2model_rating[sid] = rating

        return sent_id2model_rating

    def run(self, data):
        """Calculate citation ratings for each example. 
        """
        all_res = []
        for idx in range(len(data)):
            sample_id = self.ca_output[idx]["id"]  # TBC
            
            sent_info = data[idx]['sent_info']
            sent_id2type = self.ca_output[idx]["sent_id2type"]
            
            if self.ce_output:
                sent_id2edits = self.ce_output[idx]["sent_id2edits"]
            elif self.cce_output:
                sent_id2edits = self.cce_output[idx]["sent_id2edits"]
            else:
                raise ValueError("One of the following output files has to be provoded: ce_output, cce_output")

            sent_id2model_rating = self.rate_with_heuristics(sent_id2edits, sent_info, max_rating=1.0)

            res = self.rate(
                sent_info=sent_info, 
                sent_id2type=sent_id2type, 
                sent_id2model_rating=sent_id2model_rating,
                normalized_model_rating=True  # rating is between 0 and 1
            )
            res = {
                "id": sample_id,
                **res
            }
            all_res.append(res)

        return all_res
