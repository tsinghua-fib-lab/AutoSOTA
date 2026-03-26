from .logging_utils import get_logger
logger = get_logger(__name__)


def get_op(edit):
    """edit example: DELETE_3
    """
    return edit["edit"].split("_")[0]


def aggregate_sentence_ratings_to_answer_rating(sent_id2rating, all_none_rating=1.0):
    """
    sent_id2rating: value: rating (can be None which is to skip)
    all_none_rating: default to 0.0 (assuming at least one citation-worthy statement should be provided)

    Return answer_rating.
    """
    sanitized_sent_id2rating = [rating for rating in sent_id2rating.values() if rating is not None]
    
    if len(sanitized_sent_id2rating):
        answer_rating = sum(sanitized_sent_id2rating) / len(sanitized_sent_id2rating)
    else:
        answer_rating = all_none_rating
    
    return answer_rating


def rate(
        sent_info, 
        sent_id2type, 
        sent_id2model_rating,
        context_types2citation_requirement,
        normalized_model_rating=False
    ):
    """
    Rating function.

    sent_info: a list of dicts wit keys: citations
    sent_id2type: a dict mapping sent_idx to context type prediction
    sent_id2model_rating: a dict mapping sent_idx to model-predicted rating

    return a result dict including sent_id2rating and answer_rating
    """
    def _get_citation_requirement(sent_idx, sent_id2type, context_types2citation_requirement, has_citations):
        """All cited statements need to be evalauted.
        """
        if has_citations:
            return "yes"
        
        if not sent_id2type:  # for ablation
            return "yes"

        context_type = sent_id2type[str(sent_idx+1)]["ca_pred"]
        return context_types2citation_requirement.get(context_type, "yes")

    sent_id2rating = {}
    context_attribution_ablation = False

    for sent_idx in range(len(sent_info)):
        has_citations = True if len(sent_info[sent_idx]["citations"])>0 else False

        if context_attribution_ablation:
            citation_req = "n/a"  # does not affect predicted rating
            logger.warning("Citation requirement is set to N/A for ALL INSTANCES for Citation Attribution ablation")
        else:
            citation_req = _get_citation_requirement(sent_idx, sent_id2type, context_types2citation_requirement, has_citations=has_citations)

        if citation_req == "no":
            rating = None
        
        elif citation_req == "yes":
            if has_citations:
                sent_id = str(sent_idx+1)
                if sent_id not in sent_id2model_rating:
                    logger.warning(f"{sent_id} not found in sent_id2model_rating, set to default (1.0)")
                    rating = 1.0
                else:
                    if normalized_model_rating:
                        rating = sent_id2model_rating[sent_id]
                    else:
                        rating = (sent_id2model_rating[sent_id]-1) * 0.25
            else:
                rating = 0.0
        
        elif citation_req == "n/a":  # for CA is disabled for ablation study
            sent_id = str(sent_idx+1)
            if sent_id not in sent_id2model_rating:
                logger.warning(f"{sent_id} not found in sent_id2model_rating, set to default (1.0)")
                rating = 1.0
            else:
                if normalized_model_rating:
                    rating = sent_id2model_rating[sent_id]
                else:
                    rating = (sent_id2model_rating[sent_id]-1) * 0.25
            
        else:
            raise ValueError(f"Invalid citation requirement: {citation_req}")
        
        sent_id2rating[str(sent_idx+1)] = rating

    answer_rating = aggregate_sentence_ratings_to_answer_rating(sent_id2rating, all_none_rating=1.0)

    res = {
        "sent_id2type": sent_id2type,
        "sent_id2model_rating": sent_id2model_rating,
        "sent_id2rating": sent_id2rating,
        "answer_rating": answer_rating
    }

    return res
