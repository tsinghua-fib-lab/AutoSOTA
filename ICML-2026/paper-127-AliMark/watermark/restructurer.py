

class ReStructurer:
    def __init__(self):
        pass

    def _split_sentence_at_middle(self, text):
        if len(text) < 10:  # Too short to split
            return [text]
        
        mid = len(text) // 2
        # Find closest spaces to the midpoint
        left_space = text.rfind(' ', 0, mid)
        right_space = text.find(' ', mid)

        pivot = -1
        if left_space == -1 and right_space == -1:
            return [text]
        elif left_space == -1:
            pivot = right_space
        elif right_space == -1:
            pivot = left_space
        else:
            # Pick the closer one
            if (mid - left_space) < (right_space - mid):
                pivot = left_space
            else:
                pivot = right_space
        
        return [text[:pivot] + '.', text[pivot+1:]]

    def gen_re_merge_candidates(self, original_sentences):
        candidates = []
        if len(original_sentences) > 1:
            for i in range(len(original_sentences) - 1):
                candidates.append(
                    original_sentences[:i] + 
                    [original_sentences[i] + " " + original_sentences[i+1]] + 
                    original_sentences[i+2:]
                )
        return candidates

    def gen_re_split_candidates(self, original_sentences):
        candidates = []
        for i in range(len(original_sentences)):
            split_parts = self._split_sentence_at_middle(original_sentences[i])
            if len(split_parts) == 2:
                candidates.append(
                    original_sentences[:i] + split_parts + original_sentences[i+1:]
                )
        return candidates

    def gen_one_resplit_and_one_remerge_candidates(self, original_sentences):
        candidates = []
        split_candidates = self.gen_re_split_candidates(original_sentences)
        for split in split_candidates:
            merge_candidates = self.gen_re_merge_candidates(split)
            candidates.extend(merge_candidates)

        return candidates


if __name__ == "__main__":
    re_structurer = ReStructurer()
    original_sentences = ["This is the first sentence.", "This is the second sentence.", "This is the third sentence."]
    print("Original Sentences:", original_sentences)

    merge_candidates = re_structurer.gen_re_merge_candidates(original_sentences)
    print("\nMerge Candidates:")
    for candidate in merge_candidates:
        print(candidate)

    split_candidates = re_structurer.gen_re_split_candidates(original_sentences)
    print("\nSplit Candidates:")
    for candidate in split_candidates:
        print(candidate)

    resplit_and_remerge_candidates = re_structurer.gen_one_resplit_and_one_remerge_candidates(original_sentences)
    print("\nResplit and Remerge Candidates:")
    for candidate in resplit_and_remerge_candidates:
        print(candidate)