"""
WordNet-enhanced class alias expansion for VIP Pascal VOC classes.
Extracts synonyms, direct hypernyms, and direct hyponyms.
"""
from nltk.corpus import wordnet as wn

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
]

def get_wordnet_expansions(word, max_per_type=5):
    syns = wn.synsets(word.replace(" ", "_"), pos=wn.NOUN)
    if not syns:
        syns = wn.synsets(word.replace(" ", ""), pos=wn.NOUN)
    
    synonyms = set()
    hypernyms = set()
    hyponyms = set()
    
    for syn in syns[:3]:
        for lemma in syn.lemmas()[:max_per_type]:
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower():
                synonyms.add(name.lower())
        
        for hyper in syn.hypernyms()[:3]:
            for lemma in hyper.lemmas()[:3]:
                name = lemma.name().replace("_", " ")
                hypernyms.add(name.lower())
        
        for hypo in syn.hyponyms()[:5]:
            for lemma in hypo.lemmas()[:3]:
                name = lemma.name().replace("_", " ")
                hyponyms.add(name.lower())
    
    return synonyms, hypernyms, hyponyms

# Read original aliases
original_aliases = {}
with open("configs/cls_voc21.txt") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        aliases = [a.strip() for a in line.split(",")]
        original_aliases[aliases[0]] = aliases[1:]

all_expanded = []

for i, class_name in enumerate(VOC_CLASSES):
    if class_name == "background":
        expanded = [class_name] + original_aliases.get(class_name, [])
        all_expanded.append(expanded)
        continue
    
    expanded = set([class_name])
    if class_name in original_aliases:
        expanded.update(original_aliases[class_name])
    
    synonyms, hypernyms, hyponyms = get_wordnet_expansions(class_name)
    expanded.update(synonyms)
    
    bad = {"entity", "object", "physical object", "whole", "artifact", 
           "organism", "living thing", "matter", "substance", "thing",
           "unit", "abstraction", "instrumentality", "device"}
    for hyp in hypernyms:
        if hyp not in bad:
            expanded.add(hyp)
    expanded.update(hyponyms)
    
    all_expanded.append(list(expanded))
    new = len(expanded) - 1 - len(original_aliases.get(class_name, []))
    print(f"# Class {i}: {class_name} ({len(expanded)} aliases, +{new})")

# Output the alias file
for aliases_tmp in all_expanded:
    print(", ".join(aliases_tmp))
