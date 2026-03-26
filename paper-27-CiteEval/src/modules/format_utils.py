from pathlib import Path
import json
import io
import math
from common_utils.templates import TemplatedExample
from common_utils.logging_utils import get_logger


logger = get_logger(__name__)


class TemplatedCitation(TemplatedExample):
    def __init__(self, idx, span, citation_seq, unit_name="span"):
        super().__init__('<citation {x.unit_name}_id="{x.idx}" {x.unit_name}="{x.span}"> {x.citation_seq} </citation>')
        self.idx = idx
        self.unit_name = unit_name
        self.span = span
        self.citation_seq = citation_seq


class TemplatedSentence(TemplatedExample):
    def __init__(self, idx, span, citation_seq, unit_name="sentence", use_citations=True):
        if use_citations:
            super().__init__('<{x.unit_name} {x.unit_name}_id="{x.idx}" citation="{x.citation_seq}"> {x.span} </{x.unit_name}>')
        else:
            super().__init__('<{x.unit_name} {x.unit_name}_id="{x.idx}"> {x.span} </{x.unit_name}>')
        
        self.idx = idx
        self.unit_name = unit_name
        self.span = span
        self.citation_seq = citation_seq


class TemplatedDocument(TemplatedExample):
    def __init__(self, idx, text):
        super().__init__('<document id={x.idx}>\n{x.text}\n</document>')
        self.idx = idx
        self.text = text


class TemplatedPassage(TemplatedExample):
    def __init__(self, idx, text, title):
        super().__init__('<passage id="{x.idx}" title="{x.title}">\n{x.text}\n</passage>')
        self.idx = idx
        self.text = text
        self.title = title


def construct_span(sent_info, citeeval_config=None):
    if citeeval_config:
        sentence_level_spans = citeeval_config.get("sentence_level_spans", False)
        last_sentences_as_one_group = citeeval_config.get("last_sentences_as_one_group", False)
        max_sentence_group_size = citeeval_config.get("max_sentence_group_size", math.inf)

    buffer = []
    span_info = []

    for sent_info in sent_info:
        if len(sent_info['citations']) == 0:
            buffer.append(sent_info['clean_sent'])
            continue

        buffer.append(sent_info['clean_sent'])

        # merging logic
        # the last max_sentence_group_size sentences are treated as one group, with original citations
        # the rest: each sentence is a group, w/o citations
        if len(buffer) > max_sentence_group_size:
            for i in range(len(buffer)-max_sentence_group_size):
                span_info.append({'span': buffer[i], 'citations': []})
            buffer = buffer[-max_sentence_group_size:]
            
        if sentence_level_spans:
            for s in buffer:
                span_info.append({'span': s, 'citations': sent_info['citations']})
        else:
            span = ' '.join(buffer)
            span_info.append({'span': span, 'citations': sent_info['citations']})
        buffer = []

    if buffer:
        if last_sentences_as_one_group and not sentence_level_spans:
            span = ' '.join(buffer)
            span_info.append({'span': span, 'citations': []})
        else:
            # treat each sentence as an individual group
            for s in buffer:
                span_info.append({'span': s, 'citations': []})

    return span_info


def build_citation_xml(span_info, response_unit_name):
    citation_xmls = []

    for idx, info in enumerate(span_info):
        citations = info["citations"]
        if citations:
            citation_seq = ", ".join([str(cite) for cite in citations])
        else:
            citation_seq = "None"

        xml = TemplatedCitation(idx=idx+1, span=info["span"], citation_seq=citation_seq, unit_name=response_unit_name)
        citation_xmls.append(xml.render()[0])

    return '\n'.join(citation_xmls)


def build_sentence_xml(span_info, response_unit_name, use_citations=True):
    sentence_xmls = []

    for idx, info in enumerate(span_info):
        citations = info["citations"]
        if citations:
            citation_seq = ", ".join([str(cite) for cite in citations])
        else:
            citation_seq = "None"

        xml = TemplatedSentence(idx=idx+1, span=info["span"], citation_seq=citation_seq, unit_name=response_unit_name, use_citations=use_citations)
        sentence_xmls.append(xml.render()[0])

    return '\n'.join(sentence_xmls)


def build_passage_xml(item, unit_name="document"):
    passage_xmls = []
    for idx, doc in enumerate(item['docs']):
        if unit_name == "document":
            xml = TemplatedDocument(idx=idx+1, text=doc["text"])
        elif unit_name == "passage": 
            xml = TemplatedPassage(idx=idx+1, text=doc["text"], title=doc.get("title", ""))
        else:
            raise ValueError(f"Unsupported unit name: {unit_name}")
        
        passage_xmls.append(xml.render()[0])

    return '\n\n'.join(passage_xmls)
