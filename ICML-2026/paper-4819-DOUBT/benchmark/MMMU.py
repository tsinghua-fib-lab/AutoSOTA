from datasets import load_dataset
import ast


class MMMU:

    def __init__(self):
        self.categories = ['Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology']
        self.NUM_CATEGORY = 30
        self.NUM_SAMPLES_EACH_CATEGORY = 30
        self.ds_map = {}
        for c in self.categories:
            self.ds_map[c] = load_dataset('MMMU/MMMU', c)

    def obtain_size(self):
        return self.NUM_CATEGORY * self.NUM_SAMPLES_EACH_CATEGORY

    def retrieve(self, idx):
        ds_id = int(idx / self.NUM_SAMPLES_EACH_CATEGORY)
        real_index = idx % self.NUM_SAMPLES_EACH_CATEGORY
        ds_cur = self.ds_map[self.categories[ds_id]]
        row = ds_cur['validation'][real_index]
        if '<image 2>' in row['question'] or row['question_type'] == 'open':
            return None
        question = row['question']
        question += '\n'
        choices = ""
        choice_numbers = ""
        options = ast.literal_eval(row['options'])
        for i, c in enumerate(options):
            choices += f'({i}): {c}\n'
            choice_numbers += f'{i}, '
        choice_numbers = choice_numbers[:-2]
        question += choices
        question += '\n'
        question += f'This is a single choice question, answer only with choice number in {choice_numbers}.'
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' 
        mapping = {L: N for N, L in enumerate(letters)}
        gt_ans = mapping[row['answer']]
        result = {
            'idx': idx,
            'img': row['image_1'],
            'question': question,
            'gt_ans': gt_ans,
            'num_c': len(options),
        }
        return result


if __name__ == "__main__":
    benchmark = MMMU()
    print(benchmark.retrieve(0))