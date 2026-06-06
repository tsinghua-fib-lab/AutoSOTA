import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json as json
from tqdm import tqdm
from framework_construction.getMoralValue import *
from scipy.stats import spearmanr
import random
from gensim.models import KeyedVectors
from collections import Counter, defaultdict
from scipy.stats import median_abs_deviation as mad
from scipy import stats
import argparse

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def varying_alpha():
    df = pd.read_csv('./data/emfd_scoring.csv')
    
    # Apply softmax to probability columns
    probability_columns = ['care_p', 'fairness_p', 'loyalty_p', 'authority_p', 'sanctity_p']
    df[probability_columns] = df[probability_columns].apply(softmax, axis=1)
    
    # Calculate the new scores by multiplying _p and _sent columns
    df['care'] = df['care_p'] * df['care_sent']
    df['fairness'] = df['fairness_p'] * df['fairness_sent']
    df['loyalty'] = df['loyalty_p'] * df['loyalty_sent']
    df['authority'] = df['authority_p'] * df['authority_sent']
    df['sanctity'] = df['sanctity_p'] * df['sanctity_sent']

    with open("./data/human_association.json", 'r', encoding='utf-8') as file:
        human = json.load(file)
    
    with open("./data/llama_2.1_association.json", 'r', encoding='utf-8') as file:
        llms = json.load(file)
    alphas = [0.99, 0.95]
    dim = ['care', 'fairness', 'loyalty', 'authority', 'sanctity']
    mfd = get_mfd('./data/human_association.json')
    intersected_word = set(human.keys()).intersection(set(llms.keys())).intersection(set(df['word']))

    with open("./data/mag_words.json", 'r', encoding='utf-8') as file:
        mag = json.load(file)
    words = set()
    for key, value in mag.items():
        for v in value:
            words.add(v)
    used_words = intersected_word.difference(words)
    intersected_word = used_words
    value = 0.9
    while value >= 0.45:
        alphas.append(value)
        value -= 0.05
    
    correlation_dict = {}
    emfd_value = []
    for word in intersected_word:
        value = 0
        for d in dim:
            value += df[df['word'] == word][d].values[0]
        emfd_value.append(value)
    count = 0
    for value in tqdm(alphas,desc='Alpha Values'):
        correlation_dict[value] = {'Human':[], 'LLMs':[]}
        human_vocab_dict, human_information_matrix = stereotype_propagation("./data/human_association.json",value)
        llms_vocab_dict, llms_information_matrix = stereotype_propagation("./data/llama_2.1_association.json",value)

        for word in intersected_word:
            human_index = human_vocab_dict[word]
            llms_index = llms_vocab_dict[word]
            human_list = human_information_matrix[human_index].tolist()
            llms_list = llms_information_matrix[llms_index].tolist()
            if word in mfd:
                human_score = 0
                llms_score = 0
                for i, d in enumerate(dim):
                    human_score += human_list[i] - mfd[word][d]
                    llms_score += llms_list[i] - mfd[word][d]
                correlation_dict[value]['Human'].append(human_score)
                correlation_dict[value]['LLMs'].append(llms_score)
            else:
                correlation_dict[value]['Human'].append(sum(human_list))
                correlation_dict[value]['LLMs'].append(sum(llms_list))
        count += 1
    alphas = list(correlation_dict.keys())
    
    # Lists to store Spearman correlations
    human_spearman_corr = []
    llms_spearman_corr = []
    
    # Iterate over each alpha and calculate Spearman correlation
    for alpha in alphas:
        # Calculate Spearman correlation between human scores and EMFD values
        human_corr, _ = spearmanr(correlation_dict[alpha]['Human'], emfd_value)
        llms_corr, _ = spearmanr(correlation_dict[alpha]['LLMs'], emfd_value)
        
        human_spearman_corr.append(human_corr)
        llms_spearman_corr.append(llms_corr)
    
    # Assuming `alphas` is the list of alpha values
    alpha_min, alpha_max = min(alphas), max(alphas)
    alpha_ticks = np.arange(alpha_min, 1, 0.05)  # Creating x-ticks with increments of 0.5

    #plt.rcParams.update({'font.size': 16})  # Adjust this value for a larger overall font size

    # Plot the results
    plt.figure(figsize=(8, 6))

    plt.plot(alphas, human_spearman_corr, label='GMN-H Correlation', marker='o', color='blue')
    plt.plot(alphas, llms_spearman_corr, label='GMN-L Correlation', marker='o', color='green')

    # Set axis labels and title with a specific font size
    plt.xlabel('Alpha Value', fontsize=16)
    plt.ylabel('Spearman Correlation', fontsize=16)
    plt.xticks(alpha_ticks, fontsize=16)  # Set font size for x-axis ticks
    plt.yticks(fontsize=16)  # Set font size for y-axis ticks

    # Set the legend with a larger font size
    plt.legend(fontsize=16)

    plt.show()

def split_into_half(words, cue_words, seed):
    ground_truth = {}
    test_response = {}
    
    # Set the random seed (now it's a required parameter)
    random.seed(seed)
    
    for cue_word, response_dict in words.items():
        if cue_word.lower() not in [cw.lower() for cw in cue_words]:
            continue
        # Create a list with responses based on their frequency (converted to lowercase)
        response_list = []
        for response, freq in response_dict.items():
            response_list.extend([response.lower()] * freq)  # Convert each response to lowercase
        
        # Shuffle the list randomly
        random.shuffle(response_list)
        
        # Split the response list into two halves
        half = len(response_list) // 2
        ground_truth_responses = response_list[:half]
        test_response_responses = response_list[half:]
        
        # Count the frequency of each response in both halves
        ground_truth_counter = Counter(ground_truth_responses)
        test_response_counter = Counter(test_response_responses)
        
        # Store the result back into the dictionaries with the cue also in lowercase
        ground_truth[cue_word.lower()] = dict(ground_truth_counter)
        test_response[cue_word.lower()] = dict(test_response_counter)
    
    return ground_truth, test_response

def calculate_relative_frequency(human_ground_truth, human_test_response, llms_ground_truth,llms_test_response, human_full, llms_full):
    relative_freq = {
        'human_ground_truth': defaultdict(dict),
        'human_test_response': defaultdict(dict),
        'llms_ground_truth': defaultdict(dict),
        'llms_test_response': defaultdict(dict),
        'human_full': defaultdict(dict),
        'llms_full': defaultdict(dict)
    }
    
    # Helper function to calculate relative frequency
    def compute_relative_freq(cue_dict):
        relative_dict = defaultdict(dict)
        for cue, responses in cue_dict.items():
            total_responses = sum(responses.values())
            for response, count in responses.items():
                response = response.lower()
                relative_dict[cue][response] = count / total_responses if total_responses > 0 else 0
        return relative_dict

    # Calculate relative frequencies for each of the input dictionaries
    relative_freq['human_ground_truth'] = compute_relative_freq(human_ground_truth)
    relative_freq['human_test_response'] = compute_relative_freq(human_test_response)
    relative_freq['llms_ground_truth'] = compute_relative_freq(llms_ground_truth)
    relative_freq['llms_test_response'] = compute_relative_freq(llms_test_response)
    relative_freq['human_full'] = compute_relative_freq(human_full)
    relative_freq['llms_full'] = compute_relative_freq(llms_full)
    return relative_freq

def structure_checking():
    model_path = "Your W2V model here"
    word2vec_model = KeyedVectors.load(model_path, mmap='r')
    word2vec_model.vectors = word2vec_model.vectors.astype('float16')
    with open("./data/human_association.json", 'r', encoding='utf-8') as file:
        human_pairs = json.load(file)
    with open("./data/selected_words.json", 'r', encoding='utf-8') as file:
        selected_words = json.load(file)

    cue_words = []
    for key,value in selected_words.items():
        cue_words.extend(value)
    cue_words = list(set(cue_words))
    k_values = [1,5,10,15,20,25,30,35,40,45,50,55,60]
    precision_result_dict = {
        "human": {f"precision_at_{k}": [] for k in k_values},
        "llms": {f"precision_at_{k}": []  for k in k_values},
        "human_llms": {f"precision_at_{k}": []  for k in k_values},
        "word2vec": {f"precision_at_{k}": [] for k in k_values}
    }
    runs = 30

    correlation_result_dict = {
        "human": {w : [] for w in cue_words},
        "llms": {w: [] for w in cue_words},
        "human_llms": {w: [] for w in cue_words},
        "w2v":{}
    }
    done = False
    llms_file_name = f"./data/llama_2.1_association.json"
    with open(llms_file_name, 'r', encoding='utf-8') as file:
        llms_pairs = json.load(file) 
    for i in tqdm(range(runs), desc="Processing files"): 

        llms_ground_truth_dict, llms_test_response_dict = split_into_half(llms_pairs, cue_words,i)
        human_ground_truth_dict, human_test_response_dict = split_into_half(human_pairs, cue_words,i)
        word_strength_dict = calculate_relative_frequency(human_ground_truth_dict, human_test_response_dict, llms_ground_truth_dict,llms_test_response_dict, human_pairs, llms_pairs)

        for cue in tqdm(cue_words, desc="Processing cue words"):
            human_ground_truth = {key.lower(): value for key, value in human_ground_truth_dict[cue].items()}
            human_test_response = {key.lower(): value for key, value in human_test_response_dict[cue].items()}
            llms_ground_truth = {key.lower(): value for key, value in llms_ground_truth_dict[cue].items()}
            llms_test_response = {key.lower(): value for key, value in llms_test_response_dict[cue].items()}
            
            human_full = {key.lower(): value for key, value in human_pairs[cue].items()}
            llms_full = {key.lower(): value for key, value in llms_pairs[cue].items()}


            human_ground_truth_set = set(human_ground_truth.keys())
            llms_ground_truth_set = set(llms_ground_truth.keys())
            human_test_sorted = sorted(human_test_response, key=human_test_response.get, reverse=True)
            llms_test_sorted = sorted(llms_test_response, key=llms_test_response.get, reverse=True)

            human_full_set = set(human_full.keys())
            human_full_sorted = sorted(human_full, key=human_full.get, reverse=True)
            llms_full_sorted = sorted(llms_full, key=llms_full.get, reverse=True)

            if not done:
                nearest_neighbors = word2vec_model.similar_by_word(cue, topn=61, restrict_vocab=100000) 
                words_only = [w for w, _ in nearest_neighbors if w != cue]

            for k in k_values:
                if k <= len(human_test_sorted):
                    human_test_topk = human_test_sorted[:k]
                    precision_human = len(set(human_test_topk) & human_ground_truth_set) / k
                    precision_result_dict["human"][f"precision_at_{k}"].append(precision_human)

                if k <= len(llms_test_sorted):
                    llms_test_topk = llms_test_sorted[:k]
                    precision_llms = len(set(llms_test_topk) & llms_ground_truth_set) / k
                    precision_result_dict["llms"][f"precision_at_{k}"].append(precision_llms)

                if k <= len(human_full_sorted) and k <= len(llms_full_sorted):
                    llms_full_topk = llms_full_sorted[:k]
                    precision_human_llms = len(set(llms_full_topk) & human_full_set) / k
                    precision_result_dict["human_llms"][f"precision_at_{k}"].append(precision_human_llms)

                if not done:
                    wvs_response = words_only[:k]
                    precision_word2vec = len(set(wvs_response) & human_full_set) / k
                    precision_result_dict["word2vec"][f"precision_at_{k}"].append(precision_word2vec)

            pred_scores_human = []
            gold_scores_human = []
            for response in human_ground_truth:
                if response in human_test_response:
                    pred_scores_human.append(word_strength_dict['human_test_response'][cue][response])
                    gold_scores_human.append(word_strength_dict['human_ground_truth'][cue][response])
            if len(set(pred_scores_human)) > 1 and len(set(gold_scores_human)) > 1:
                correlation_human, _ = spearmanr(pred_scores_human, gold_scores_human)
                correlation_result_dict["human"][cue].append((2*correlation_human)/(1 + correlation_human))

            pred_scores_llms = []
            gold_scores_llms = []
            for response in llms_ground_truth:
                if response in llms_test_response:
                    pred_scores_llms.append(word_strength_dict['llms_test_response'][cue][response])
                    gold_scores_llms.append(word_strength_dict['llms_ground_truth'][cue][response])
            if len(set(pred_scores_llms)) > 1 and len(set(gold_scores_llms)) > 1:
                correlation_llms, _ = spearmanr(pred_scores_llms, gold_scores_llms)
                correlation_result_dict["llms"][cue].append((2* correlation_llms)/(1 + correlation_llms))
            else:
                correlation_result_dict["llms"][cue].append(0)

            pred_scores_full_llms = []
            gold_scores_full_human = []
            for response in llms_full:
                if response in human_full:
                    pred_scores_full_llms.append(word_strength_dict['llms_full'][cue][response])
                    gold_scores_full_human.append(word_strength_dict['human_full'][cue][response])
            if len(set(pred_scores_full_llms)) > 1 and len(set(gold_scores_full_human)) > 1:
                correlation_human_llms, _ = spearmanr(pred_scores_full_llms, gold_scores_full_human)
                correlation_result_dict["human_llms"][cue].append(correlation_human_llms)
            else:
                correlation_result_dict["human_llms"][cue].append(0)

            if not done:
                pred_scores_w2v = []
                gold_scores_full_human = []
                for response in human_full:
                    if response in words_only:
                        pred_scores_w2v.append(len(words_only) - words_only.index(response))
                        gold_scores_full_human.append(word_strength_dict['human_full'][cue][response])
                if len(set(pred_scores_w2v)) > 1 and len(set(gold_scores_full_human)) > 1:
                    correlation_w2v, _ = spearmanr(pred_scores_w2v, gold_scores_full_human)
                    correlation_result_dict["w2v"][cue]= correlation_w2v
                else:
                    correlation_result_dict["w2v"][cue]= 0
        done = True


    human_means_sd = []
    llms_means_sd = []
    human_llms_means_sd = []
    word2vec_means_sd = []

    for k in k_values:
        precisions = precision_result_dict["human"][f"precision_at_{k}"]
        mean_precision_human = np.mean(precisions)
        sd_precision_human = np.std(precisions)
        human_means_sd.append((mean_precision_human, sd_precision_human))

    for k in k_values:
        precisions = precision_result_dict["llms"][f"precision_at_{k}"]
        mean_precision_llms = np.mean(precisions)
        sd_precision_llms = np.std(precisions)
        llms_means_sd.append((mean_precision_llms, sd_precision_llms))

    for k in k_values:
        precisions = precision_result_dict["human_llms"][f"precision_at_{k}"]
        mean_precision_human_llms = np.mean(precisions)
        sd_precision_human_llms = np.std(precisions)
        human_llms_means_sd.append((mean_precision_human_llms, sd_precision_human_llms))
    
    word2vec_means = []
    for k in k_values:
        precisions = precision_result_dict["word2vec"][f"precision_at_{k}"]
        mean_precision_word2vec = np.mean(precisions)
        word2vec_means.append(mean_precision_word2vec)
        sd_precision_word2vec = np.std(precisions)
        word2vec_means_sd.append((mean_precision_word2vec, sd_precision_word2vec))

    print("Human Mean and SD:")
    for i, k in enumerate(k_values):
        print(f"At K={k}: Mean = {human_means_sd[i][0]:.4f}, SD = {human_means_sd[i][1]:.4f}")

    print("\nLLMs Mean and SD:")
    for i, k in enumerate(k_values):
        print(f"At K={k}: Mean = {llms_means_sd[i][0]:.4f}, SD = {llms_means_sd[i][1]:.4f}")
    
    print("\nHuman-LLMs Mean and SD:")
    for i, k in enumerate(k_values):
        print(f"At K={k}: Mean = {human_llms_means_sd[i][0]:.4f}, SD = {human_llms_means_sd[i][1]:.4}")

    print("\nWord2Vec Mean and SD:")
    for i, k in enumerate(k_values):
        print(f"At K={k}: Mean = {word2vec_means_sd[i][0]:.4f}, SD = {word2vec_means_sd[i][1]:.4f}")

    def calculate_mean_and_ci(correlations, confidence=0.95):
        mean_correlation = np.mean(correlations)
        n = len(correlations)
        sem = stats.sem(correlations)
        margin = sem * stats.t.ppf((1 + confidence) / 2., n-1)
        return mean_correlation, mean_correlation - margin, mean_correlation + margin
    
    w2vc = correlation_result_dict["w2v"]
    print(f"w2v correlation: {sum(w2vc.values())/len(w2vc)}")


    for key, word_dict in correlation_result_dict.items():
        if key == "w2v":
            continue
        run_averages = []
        num_runs = 30

        for run_idx in range(num_runs):
            total_correlation = 0
            word_count = 0
            
            # Calculate average correlation for this particular run
            for word, correlations in word_dict.items():
                if len(correlations) > run_idx:
                    total_correlation += correlations[run_idx]
                    word_count += 1

            if word_count > 0:
                run_average = total_correlation / word_count
                run_averages.append(run_average)

        # Calculate the overall average correlation and 95% confidence interval for this key
        if run_averages:
            if key != "w2v":
                avg_correlation, lower_ci, upper_ci = calculate_mean_and_ci(run_averages)
                print(f"Average correlation for {key}: {avg_correlation:.5f}")
                print(f"95% Confidence Interval for {key}: ({lower_ci:.3f}, {upper_ci:.3f})")

def mag_comparison():
    human_path = "./data/human_association.json"
    llms_path = "./data/llama_2.1_association.json"
    human_vocab_dict, human_information_matrix = stereotype_propagation(human_path, 0.75)
    llms_vocab_dict, llms_information_matrix = stereotype_propagation(llms_path, 0.9)
    MFD_dict = get_mfd(human_path)
    MFD_words = set(MFD_dict.keys())
    df = pd.read_csv('./data/emfd_scoring.csv')

    df['care'] = df['care_p'] * df['care_sent']
    df['fairness'] = df['fairness_p'] * df['fairness_sent']
    df['loyalty'] = df['loyalty_p'] * df['loyalty_sent']
    df['authority'] = df['authority_p'] * df['authority_sent']
    df['sanctity'] = df['sanctity_p'] * df['sanctity_sent']
    dimension = ['care', 'fairness', 'loyalty', 'authority', 'sanctity']
    human_scores = []
    llms_scores = []
    all_emfd_scores = []
    for i,dim in enumerate(dimension):
        with open('./data/mag_words.json', 'r', encoding='utf-8') as file:
            mag_words = json.load(file)
        common_words = mag_words[dim]
        common_words = [word.replace("overall", "overalls").replace("judgment", "judgement") for word in common_words]
        emfd_scores = []
        human_propagated_scores = []
        llms_propagated_scores = []
        for word in common_words:
            if word == "overalls":
                emfd_word = "overall"
            elif word == "judgement":
                emfd_word = "judgment"
            else:
                emfd_word = word
            human_word_id = human_vocab_dict[word]
            llms_word_id = llms_vocab_dict[word]

            human_propagated_score = human_information_matrix[human_word_id][i].cpu().numpy()
            llms_propagated_score = llms_information_matrix[llms_word_id][i].cpu().numpy()

            if word in MFD_words:
                human_propagated_score = human_propagated_score - MFD_dict[word][dim] 
                llms_propagated_score = llms_propagated_score - MFD_dict[word][dim]

            human_propagated_scores.append(human_propagated_score)
            llms_propagated_scores.append(llms_propagated_score)
            emfd_score = df.loc[df['word'] == emfd_word, dim].values[0]
            emfd_scores.append(emfd_score)

        human_corr, h_p = spearmanr(human_propagated_scores, emfd_scores)
        print(f"human correlation: {dim}, {human_corr}")
        print(f"human p-value: {dim}, {h_p}")
        llms_corr, l_p = spearmanr(llms_propagated_scores, emfd_scores)
        print(f"llm correlation: {dim}, {llms_corr}")
        print(f"llm p-value: {dim}, {l_p}")
        human_scores.extend(human_propagated_scores)
        llms_scores.extend(llms_propagated_scores)
        all_emfd_scores.extend(emfd_scores)
    print("Overall")
    human_corr, h_p = spearmanr(human_scores, all_emfd_scores)
    print(f"human all: {human_corr}")
    print(f"human p-value: {h_p}")
    llms_corr, l_p = spearmanr(llms_scores, all_emfd_scores)
    print(f"llm: {llms_corr}")
    print(f"llm p-value: {l_p}")


def rank():
    with open("./data/intersection.json", 'r', encoding='utf-8') as file:
        common_words = json.load(file)

    dimensions = ['care', 'fairness', 'loyalty', 'authority', 'sanctity']
    with open("./data/human_moral.json", 'r', encoding='utf-8') as file:
        human_vocab_dict = json.load(file)
    with open("./data/llama_2.1_moral.json", 'r', encoding='utf-8') as file:
        llms_vocab_dict = json.load(file)
    mfd_dict = get_mfd("./data/human_association.json")
    mfd_words = set(mfd_dict.keys())
    total_words = set()
    for dim in dimensions:
        words = common_words[dim]
        for word in words:
            total_words.add(word)
    
    scores_dict = {'human': {}, 'llms': {}, 'emfd': {}}
    
    for word in llms_vocab_dict:
        human_scores = human_vocab_dict[word]
        llms_scores = llms_vocab_dict[word]
        if word not in mfd_words:
            scores_dict['human'][word] = human_scores
            scores_dict['llms'][word] = llms_scores
        else:
            mfd_d= mfd_dict[word]
            mfd_score = []
            for d in mfd_d:
                mfd_score.append(mfd_d[d])
            scores_dict['human'][word] = [s - a for s,a in zip(human_scores, mfd_score)]
            scores_dict['llms'][word] = [s - a for s,a in zip(llms_scores, mfd_score)]

    # Function to find top 10 most and least negative/positive words
    def find_extremes(scores):
        sorted_words = sorted(scores.items(), key=lambda item: item[1])
        top_10_negative = sorted_words[:10]
        top_10_positive = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:10]
        return top_10_negative, top_10_positive
    
    def normalize(scores_dict):
        summed_scores = {}
        for word, scores in scores_dict.items():
            summed_scores[word] = sum(scores)

        # Get all the summed scores for normalization
        summed_values = np.array(list(summed_scores.values()))

        # Calculate the median and median absolute deviation (MAD)
        median_val = np.median(summed_values)
        mad_val = mad(summed_values)

        if mad_val == 0:
            # If all values are the same, set them all to 0 to avoid division by zero
            normalized_scores = {word: 0 for word in summed_scores}
        else:
            # Apply robust Z-score normalization using median and MAD
            normalized_scores = {
                word: (score - median_val) / mad_val for word, score in summed_scores.items()
            }
        
        return normalized_scores
    
    # Function to find top 10 neutral words (closest to 0)
    def find_neutral(scores):
        sorted_words = sorted(scores.items(), key=lambda item: abs(item[1]))  # Sort by absolute value
        top_10_neutral_candidates = sorted_words[:10]  # Get the first 10 closest to zero
        
        # Sort these candidates by their actual values (closest to zero)
        #top_10_neutral = sorted(top_10_neutral_candidates, key=lambda item: item[1], reverse=False)
        return top_10_neutral_candidates

    scores_dict['human'] = normalize(scores_dict['human'])
    scores_dict['llms'] = normalize(scores_dict['llms'])
    scores_dict['emfd'] = normalize(scores_dict['emfd'])
    # # Get top 10 most and least negative/positive for each dataset
    human_negative, human_positive = find_extremes(scores_dict['human'])
    llms_negative, llms_positive = find_extremes(scores_dict['llms'])

    human_positive_a = []
    human_negative_a = []
    llms_positive_a = []
    llms_negative_a = []
    for word, score in scores_dict['human'].items():
        if score > 0:
            human_positive_a.append((word, score))
        else:
            human_negative_a.append((word, score))
    for word, score in scores_dict['llms'].items():
        if score > 0:
            llms_positive_a.append((word, score))
        else:
            llms_negative_a.append((word, score))
    print(f"human mean positive: {np.mean([x[1] for x in human_positive_a])}")
    print(f"human mean negative: {np.mean([x[1] for x in human_negative_a])}")
    print(f"llms mean positive: {np.mean([x[1] for x in llms_positive_a])}")
    print(f"llms mean negative: {np.mean([x[1] for x in llms_negative_a])}")

    human_neutral = find_neutral(scores_dict['human'])
    llms_neutral = find_neutral(scores_dict['llms'])

    print("Human - Top 10 Most Negative:")
    for word, score in human_negative:
        print(f"{word}: {score:.2f}")
    
    print("\nHuman - Top 10 Most Positive:")
    for word, score in human_positive:
        print(f"{word}: {score:.2f}")
    
    print("\nHuman - Top 10 Most Neutral:")
    for word, score in human_neutral:
        print(f"{word}: {score:.5f}")
    
    print("\nLLMs - Top 10 Most Negative:")
    for word, score in llms_negative:
        print(f"{word}: {score:.2f}")
    
    print("\nLLMs - Top 10 Most Positive:")
    for word, score in llms_positive:
        print(f"{word}: {score:.2f}")
    
    print("\nLLMs - Top 10 Most Neutral:")
    for word, score in llms_neutral:
        print(f"{word}: {score:.5f}")


def main():
    parser = argparse.ArgumentParser(
    description=(
        "Experiments available:\n"
        "  varying_alpha: The experiment in Section 5.1 (Optimizing Alpha) finds the optimal alpha for human and LLM data in moral information propagation.\n"
        "  structure_checking: The experiment checks the precision@k for LLMs and W2V models, and produces data for Figure 4 in Section 4.2 and Figure 5 in the Appendix. You will need to input the path to your google-news-300 model on line 187. \n"
        "  mag_comparison: The experiment produces the correlation between the GMN-propagated moral values and the eMFD scores for the concepts used in MAG. The data produced is used to construct Table 1.\n"
        "  rank: The experiment produces the top 10 most negative and positive concepts for GMN-H and GMN-L. The results are used to construct Table 2."
    ),
    formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "action", 
        type=str, 
        choices=["varying_alpha", "structure_checking", "mag_comparison", "rank"], 
        help="Select the function to execute."
    )

    args = parser.parse_args()

    if args.action == "varying_alpha":
        varying_alpha()
    elif args.action == "structure_checking":
        structure_checking()
    elif args.action == "mag_comparison":
        mag_comparison()
    elif args.action == "rank":
        rank()

if __name__ == "__main__":
    main()