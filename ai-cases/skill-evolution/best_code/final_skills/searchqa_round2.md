# Question Answering Skill

This skill was compiled from optimization traces. Follow the rules in order: first select evidence, then bind the clue frame, then calibrate the answer boundary for exact match.

## 1. Evidence selection

- Rank exact or near-verbatim clue echoes above broad topical matches; answer from the echoed block or its immediately attached answer label.
- Anchor on distinctive clue vocabulary: numbers, dates, capitalized names, quoted terms, aliases, nicknames, and unusual wording.
- Ignore pages that overlap only on common words. Prefer passages that contain the described relation, not merely the same topic.
- If an echoed trivia, flashcard, crossword, or Jeopardy-style entry has an immediately attached answer label, treat that label as the highest-strength evidence.
- For short ambiguous clues, require an exact title/snippet echo before relying on world knowledge.
- If an echo is truncated before the answer label, corroborate the same distinctive constraints in ordinary reference passages before answering.

## 2. Clue-frame binding

- Use frame nouns, possessives, and relational prepositions to select the required answer type and referent.
- For definition or alias wording such as "called this term", "also known as", or "is referred to as", answer with the named term or entity being defined.
- For inverse definitions, meanings, glosses, or scoring values, answer with the headword or named term bound to that value, not with the descriptive gloss.
- For possessive frames such as "this country's" or "this team's", answer with the possessor/container entity.
- For relational frames such as "the X of/from/in this Y", answer with the entity selected by the prepositional phrase when that is the requested slot.
- When the clue gives a bare list of concrete entities without asking for a category, do not answer with the container category unless a source explicitly singles it out as the target.

## 3. Answer-interface calibration

- After locating evidence, output the shortest canonical entity, title, term, or phrase that satisfies the clue frame; remove descriptors, appositives, category words, and explanatory context.
- Use the exact name, title, term, or phrase from the evidence.
- Do not output explanations, prefixes, quotation marks, restatements of the question, or surrounding context.
- Do not add first names, category words, appositives, nationality/dynasty labels, media-type labels, or descriptors unless the clue explicitly asks for them.
- If both a longer descriptive phrase and a shorter canonical label satisfy the evidence, prefer the shorter label.
- Preserve required honorifics or disambiguating words only when they are part of the canonical answer span.

## 4. Conflict resolution

- Evidence strength order: attached answer label after exact clue echo > exact relational sentence > document title plus matching snippet > repeated topical mention.
- If repeated documents disagree, choose the source whose sentence matches the clue's grammar and constraints.
- If world knowledge conflicts with the provided context, follow the context.
- If the answer seems semantically correct but longer than necessary, shorten it before final output.

## 5. Abstract calibration patterns

- If evidence contains `[descriptor] [entity]` and the clue asks for the entity, return only `[entity]`.
- If evidence contains `[category word] [term]` and the clue asks for the term, return only `[term]`.
- If the clue says `the X of/from/in this Y`, return `Y` when Y is the requested frame referent.
- If an echoed clue is followed by a delimiter and an answer label, copy the answer label's canonical span.

## 6. Final checklist

- Did I use the strongest clue echo or relation-bearing passage?
- Did I bind the clue frame to the right answer type?
- Did I reduce the answer to the shortest canonical span that still satisfies the clue?
- Did I remove descriptors, category words, appositives, and explanations?
- Did I output only the answer span?
