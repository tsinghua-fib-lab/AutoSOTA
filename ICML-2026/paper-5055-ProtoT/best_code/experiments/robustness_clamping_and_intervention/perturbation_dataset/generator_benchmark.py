# generator_benchmark.py
# Robust perturbation dataset generator
# 7 categories × 500 pairs (3500 total) with natural variance + fallback pools
# Synonym slice: balanced between 1/2/3 replacements.

import json, random, re
from datasets import load_dataset
from nltk.corpus import wordnet as wn
import nltk

random.seed(42)

DATASET_FILE = "perturbation_benchmark.jsonl"

CATEGORIES = [
    "synonym",
    "typo",
    "spelling",
    "morphology",
    "contraction",
    "punctuation",
    "abbreviation"
]

# Ensure NLTK resources
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# --- Load corpora ---
def load_wikitext(n=8000):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    sentences = []
    for line in ds["text"]:
        line = line.strip()
        if 5 <= len(line.split()) <= 20 and line[0].isupper():
            sentences.append(line)
    return sentences[:n]

def load_dialog(n=8000):
    ds = load_dataset("daily_dialog", split="train")
    sentences = []
    for row in ds:
        for utt in row["dialog"]:
            s = utt.strip()
            if 3 <= len(s.split()) <= 15:
                sentences.append(s)
    random.shuffle(sentences)
    return sentences[:n]

def load_agnews(n=4000):
    ds = load_dataset("ag_news", split="train[:4000]")
    sentences = []
    for row in ds:
        s = row["text"].strip().split(".")[0]
        if 5 <= len(s.split()) <= 20:
            sentences.append(s)
    return sentences[:n]

def load_mixed_sentences(n=20000):
    sents = load_wikitext(n//2) + load_dialog(n//3) + load_agnews(n//6)
    random.shuffle(sents)
    return sents[:n]

# --- Pools ---
US_UK = {
    "color":"colour","organize":"organise","analyze":"analyse",
    "center":"centre","theater":"theatre","gray":"grey",
    "favorite":"favourite","defense":"defence","honor":"honour"
}
ABBREV_POOL = [
    ("Doctor Smith arrived.", "Dr. Smith arrived."),
    ("Professor Miller will talk.", "Prof. Miller will talk."),
    ("Mister Brown is waiting.", "Mr. Brown is waiting."),
    ("Saint John’s church is old.", "St. John’s church is old."),
    ("United States is large.", "U.S. is large."),
    ("United Kingdom is small.", "U.K. is small."),
    ("European Union meets today.", "E.U. meets today."),
    ("January is cold.", "Jan. is cold."),
    ("California is sunny.", "CA is sunny."),
    ("Doctor Adams was late.", "Dr. Adams was late.")
]
CONTRACTION_POOL = [
    ("They do not agree.", "They don’t agree."),
    ("It is fine.", "It’s fine."),
    ("We are ready.", "We’re ready."),
    ("You are late.", "You’re late."),
    ("He will not come.", "He won’t come."),
    ("I will wait.", "I’ll wait."),
    ("She has finished.", "She’s finished."),
    ("That is true.", "That’s true."),
    ("I have done it.", "I’ve done it."),
    ("They are happy.", "They’re happy.")
]
SPELLING_POOL = [
    ("I like this color.", "I like this colour."),
    ("Please organize the files.", "Please organise the files."),
    ("We must analyze the data.", "We must analyse the data."),
    ("The city center is crowded.", "The city centre is crowded."),
    ("The theater is closed.", "The theatre is closed."),
    ("Gray clouds cover the sky.", "Grey clouds cover the sky."),
    ("This is my favorite song.", "This is my favourite song."),
    ("The defense was strong.", "The defence was strong."),
    ("He received an honor.", "He received an honour."),
    ("The traveler is tired.", "The traveller is tired.")
]
MORPHOLOGY_POOL = [
    ("The cat is sleeping.", "The cats are sleeping."),
    ("The dog is barking.", "The dogs are barking."),
    ("The car is fast.", "The cars are fast."),
    ("The book is interesting.", "The books are interesting."),
    ("The house is big.", "The houses are big."),
    ("The apple is red.", "The apples are red."),
    ("The boy runs quickly.", "The boys run quickly."),
    ("The girl sings well.", "The girls sing well."),
    ("The tree is tall.", "The trees are tall."),
    ("The bird is flying.", "The birds are flying.")
]
PUNCTUATION_POOL = [
    ("I like apples", "I like apples."),
    ("Let’s eat grandma", "Let’s eat, grandma"),
    ("Where are you", "Where are you?"),
    ("This is amazing", "This is amazing!"),
    ("He said hello", "He said, hello"),
    ("The sky is blue", "The sky is blue."),
    ("Can you help me", "Can you help me?"),
    ("She was late however she still joined", "She was late; however, she still joined."),
    ("It is raining", "It is raining."),
    ("Stop", "Stop!")
]

# --- Perturbation functions ---
def synonym_swap(sent, max_replacements=1):
    words = sent.split()
    candidates = list(range(len(words)))
    random.shuffle(candidates)
    replaced = 0
    for idx in candidates:
        w = words[idx]
        syns = wn.synsets(w)
        for synset in syns:
            lemmas = [l.name().replace("_", " ") for l in synset.lemmas()]
            safe = [s for s in lemmas if s.lower() != w.lower() and " " not in s and abs(len(s)-len(w))<4]
            if safe:
                words[idx] = random.choice(safe)
                replaced += 1
                break
        if replaced >= max_replacements:
            break
    if replaced > 0:
        return " ".join(words)
    return None

def typo_noise(sent):
    words = sent.split()
    j = random.randrange(len(words))
    w = words[j]
    if len(w) > 3:
        i = random.randrange(1, len(w)-1)
        c = random.choice("abcdefghijklmnopqrstuvwxyz")
        words[j] = w[:i] + c + w[i+1:]
        return " ".join(words)
    return None

def spelling_variant(sent):
    for us, uk in US_UK.items():
        if us in sent:
            return sent.replace(us, uk, 1)
    return None

def morphology_variant(sent):
    for w in ["cat","dog","car","book","house","tree","apple","boy","girl","bird"]:
        if w in sent:
            return sent.replace(w, w+"s", 1)
    return None

def contraction_variant(sent):
    s2 = re.sub(r"\bdo not\b", "don’t", sent)
    s2 = re.sub(r"\bis not\b", "isn’t", s2)
    s2 = re.sub(r"\bcan not\b", "can’t", s2)
    s2 = re.sub(r"\bwill not\b", "won’t", s2)
    s2 = re.sub(r"\bthey are\b", "they’re", s2)
    s2 = re.sub(r"\bit is\b", "it’s", s2)
    if s2 != sent:
        return s2
    return None

def punctuation_variant(sent):
    if "," in sent:
        return sent.replace(",", "")
    elif not sent.endswith((".", "?", "!")):
        return sent + "."
    return None

def abbreviation_variant(sent):
    for long, short in [("Doctor ", "Dr. "),("Professor ","Prof. "),("Mister ","Mr. ")]:
        if long in sent:
            return sent.replace(long, short, 1)
    return None

# --- Generator with rebalanced synonym slice ---
def generate_dataset(n_per_slice=500, out_file=DATASET_FILE):
    sentences = load_mixed_sentences(30000)
    data = []
    for category in CATEGORIES:
        fn = {
            "synonym": synonym_swap,
            "typo": typo_noise,
            "spelling": spelling_variant,
            "morphology": morphology_variant,
            "contraction": contraction_variant,
            "punctuation": punctuation_variant,
            "abbreviation": abbreviation_variant,
        }[category]

        pool = {
            "synonym": [],
            "typo": [],
            "spelling": SPELLING_POOL,
            "morphology": MORPHOLOGY_POOL,
            "contraction": CONTRACTION_POOL,
            "punctuation": PUNCTUATION_POOL,
            "abbreviation": ABBREV_POOL,
        }[category]

        used = 0

        if category == "synonym":
            # enforce 50% with 1 replacement, 30% with 2, 20% with 3
            quotas = {1: int(n_per_slice*0.5), 2: int(n_per_slice*0.3), 3: n_per_slice - int(n_per_slice*0.5) - int(n_per_slice*0.3)}
            counts = {1: 0, 2: 0, 3: 0}
            for s in sentences:
                if all(counts[k] >= quotas[k] for k in quotas):
                    break
                for k in [1,2,3]:
                    if counts[k] < quotas[k]:
                        pert = synonym_swap(s, max_replacements=k)
                        if pert and pert != s:
                            data.append({"slice": category, "original": s, "perturbed": pert})
                            counts[k] += 1
                            break
            print(f"✅ {category}: {sum(counts.values())} pairs (1→{counts[1]}, 2→{counts[2]}, 3→{counts[3]})")
        else:
            for s in sentences:
                if used >= n_per_slice:
                    break
                pert = fn(s)
                if pert and pert != s:
                    data.append({"slice": category, "original": s, "perturbed": pert})
                    used += 1
            # Fallback pool
            while used < n_per_slice:
                a, b = random.choice(pool)
                data.append({"slice": category, "original": a, "perturbed": b})
                used += 1
            print(f"✅ {category}: {used} pairs")

    with open(out_file, "w") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n🎉 Wrote {len(data)} pairs to {out_file}")

if __name__ == "__main__":
    generate_dataset()