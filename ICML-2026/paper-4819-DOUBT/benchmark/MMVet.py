from datasets import load_dataset


class MMVet:

    def __init__(self):
        self.ds = load_dataset("whyu/mm-vet")

    def obtain_size(self):
        return len(self.ds['test'])

    def retrieve(self, idx):
        row = self.ds['test'][idx]
        question = f"{row['question']}\nNOTE: Provide only the final answer. Do not provide unrelated details."
        result = {
            'idx': idx,
            'img': row['image'],
            'question': question,
            'gt_ans': row['answer'],
        }
        return result

if __name__ == "__main__":
    benchmark = MMVet()
    print(benchmark.retrieve(0))