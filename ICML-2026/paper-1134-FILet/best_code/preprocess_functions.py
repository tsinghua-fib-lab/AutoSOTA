instructions = { 
    "arcc": "Answer the question based on the given choices.", 
    "arce": "Answer the question based on the given choices.", 
    "hellaswag": "Choose the most appropriate ending for the given context.", 
    "obqa": "Please choose the correct answer to the question: ", 
    "siqa": "Answer the question based on the provided context and given choices.", 
    "piqa": "Choose the solution that best achieves the goal.", 
    "winogrande": "Choose the correct option that completes the sentence.", 
    "boolq": "Decide if the question can be answered with True or False."
    }


def data_to_sequence_two_choices(examples, sentence1_key, choice1_key, choice2_key, args):
    sentence1 = examples[sentence1_key]
    choices1 = examples[choice1_key]
    choices2 = examples[choice2_key]

    for i in range(len(choices1)):
        sentence1[i] = f"{instructions[args.task_name]} Question: {sentence1[i]} \n\nChoices: \n"
        sentence1[i] += f"A: {choices1[i]} \nB: {choices2[i]} \n" + "Answer: "

    return (sentence1, )

def data_to_sequence_three_choices(examples, sentence1_key, sentence2_key, args):
    sentence1 = examples[sentence1_key]
    sentence2 = examples[sentence2_key]
    for i in range(len(sentence1)):
        s = f"{instructions[args.task_name]} Context: {sentence1[i]} Question: {sentence2[i]} \n\n Choices: \n"
        s += f"A: {examples['answerA'][i]} \n"
        s += f"B: {examples['answerB'][i]} \n"
        s += f"C: {examples['answerC'][i]} \n"
        sentence1[i] = s + "Answer: "
    return (sentence1, )


def data_to_sequence_four_choices(examples, sentence1_key, sentence2_key, args):
    sentence1 = examples[sentence1_key]
    if args.task_name == "obqa":
        return (sentence1,)
    sentence2 = examples[sentence2_key]
    for i in range(len(sentence2)):
        s = ""
        for j in range(len(sentence2[i])):
            if args.task_name == "obqa":
                s += f"Choice {j+1}: {sentence2[i]['text'][j]} "
            else:
                s += f"Choice {j+1}: {sentence2[i][j]} "
        if args.task_name == "obqa":
            sentence2[i] = f"{instructions[args.task_name]} Question: " + sentence1[i] + " \n" + s + "Answer: "
        else:
            sentence2[i] = f"{instructions[args.task_name]} Context: " + sentence1[i] + " \n" + s + "Answer: "
    return (sentence2, )

def data_to_sequence_arc(examples, sentence1_key, sentence2_key, label_key, args):
    sentence1 = examples[sentence1_key]
    sentence2 = examples[sentence2_key]
    # five_choices_idx = [i for i in range(len(sentence2)) if len(sentence2[i]['label']) == 5]
    # sentence1 = [sentence1[i] for i in range(len(sentence1)) if i not in five_choices_idx]
    # sentence2 = [sentence2[i] for i in range(len(sentence2)) if i not in five_choices_idx]
    for i in range(len(sentence2)):
        s = "" 
        for j in range(len(sentence2[i]['label'])):
            # if args.task_name == "arcc" and j == 4:
            #     break
            s += f"Choice {j}: {sentence2[i]['text'][j]}; "
        sentence1[i] = f"{instructions[args.task_name]} Question: " + sentence1[i] + s + "Answer: "

    labels = []
    for i in range(len(sentence2)):
        label_list = {sentence2[i]['label'][j]: j for j in range(len(sentence2[i]['label']))}
        labels.append(label_list[examples[label_key][i]])
    return (sentence1, ), labels


def data_to_sequence_classification(examples, sentence1_key, sentence2_key, args):
    # sentence1 = examples[sentence1_key]
    sentence2 = examples[sentence2_key]
    # sentences = [f"{instructions[args.task_name]} Passage: {sentence1[i]} Question: {sentence2[i]}" for i in range(len(sentence1))]
    sentences = [f"{instructions[args.task_name]} Question: {sentence2[i]}\nAnswer:" for i in range(len(sentence2))]
    # print(sentences)
    # exit()
    return (sentences, )