import io
from turtle import color
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import logger
import colorsys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import classification_report
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import io

from activation_gathering import activation_gathering_data_ordering
from llm_wrapper import LLMWrapper


def evaluation_data_ordering(params):
    return activation_gathering_data_ordering(params) + "/detection_evaluation/"


def evaluate_detection(df, params, detection_dictionary, logger=None):
    # Run different evaluation methods
    # Returns a dictionary of the overall evaluation metrics and corresponding plots
    
    # In case of multiple evaluation types, get the eval code name
    eval_code_name = params.get("eval_code_name", "")

    # Truncation if prarameter is set
    if params["detection_arguments"].get("number_prompts_truncation", None) is not None:
        number_prompts_truncation = params["detection_arguments"]["number_prompts_truncation"]
        df = df[df["input_text_id"] < number_prompts_truncation]

    token_place_acc = token_wise_accuracy(
        detection_dictionary["test_predictions"], 
        detection_dictionary["test_ground_truth"],
        detection_dictionary["test_token_ids"]
    ) 

    last_token_accuracy_value = last_token_accuracy(
        detection_dictionary["test_predictions"], 
        detection_dictionary["test_ground_truth"],
        detection_dictionary["test_token_ids"]
    )
    print("Last token accuracy:", last_token_accuracy_value)


    # Convert the sentence id into a boolean for test/train split
    split_labels = detection_dictionary["split_list"]
    df["split_label"] = df["input_text_id"].apply(lambda x: split_labels[x])
    df_test = df[df["split_label"] == 2]

    # Get the token wise sentence numbers
    # token_wise_sentence_ids = []
    # for i in range(len(df_test)):
    #     sentence_id = df_test["input_text_id"].iloc[i]
    #     act_dic = df_test["activations"].iloc[i]
    #     num_tokens = len(act_dic[list(act_dic.keys())[0]]) # - df_test["input_token_length"].iloc[i]
    #     token_wise_sentence_ids.extend([sentence_id] * num_tokens)

    detection_sentence_ids = detection_dictionary["test_sentence_ids"]

    # Get list of accuracy according to the step where to perform the accuracy measure from
    sentence_acc, sentence_proba_acc = sentence_wise_accuracy_all_steps(
        # token_wise_sentence_ids,
        detection_sentence_ids,
        detection_dictionary["test_ground_truth"], 
        detection_dictionary["test_predictions"],
        detection_dictionary["test_probabilities"],
        detection_dictionary["test_token_ids"],
        number_of_steps=1,
        verbose=True
    )

    train_fig = plot_train_metrics(
        detection_dictionary["train_accuracy"], 
        detection_dictionary["train_loss"], 
        detection_dictionary["validation_accuracy"], 
        detection_dictionary["validation_loss"], 
        params
    )

    accuracy_fig = plot_accuracy_metrics(
        detection_dictionary, 
        token_place_acc, 
        sentence_acc, 
        params
    )


    html = sentence_html_hitmap(
        df_test,
        params,
        detection_sentence_ids,
        detection_dictionary["test_ground_truth"],
        detection_dictionary["test_probabilities"]
    )
    print("Generated HTML hitmap.")


    # Show the figures
    # train_fig.show()
    # accuracy_fig.show()
    # Save the figures
    # evaluation_path = evaluation_data_ordering(params)
    # train_fig.write_image(f"{evaluation_path}/train_metrics.png")
    # accuracy_fig.write_image(f"{evaluation_path}/accuracy_metrics.png")

    evaluation_dict = {
        # Detection metrics
        "train_accuracy": detection_dictionary["train_accuracy"],
        "validation_accuracy": detection_dictionary["validation_accuracy"],
        "test_accuracy": detection_dictionary["test_accuracy"],
        "token_position_accuracy": token_place_acc,
        "last_token_accuracy": last_token_accuracy_value,
        "sentence_voting_acc": sentence_acc,

        # html hitmap
        "token_hitmap_html": html,
        
        # Quality Metrics
        "perplexity_steered": df[df["steering_type"] == "steered"]["perplexity"].mean(),
        "perplexity_vanilla": df[df["steering_type"] == "vanilla"]["perplexity"].mean(),
    }
    

    ##### REPORT TO CLEARML #####
    if logger is not None:
        # logger.report_scalar("accuracy", "train", 0, evaluation_dict["train_accuracy"])

        logger.report_single_value("Mean token test accuracy" + eval_code_name, value=np.mean(token_place_acc))
        logger.report_single_value("Last token test accuracy" + eval_code_name, value=last_token_accuracy_value)
        logger.report_single_value("Sentence voting test accuracy" + eval_code_name, value=np.max(sentence_acc))
        logger.report_single_value("Sentence probability voting test accuracy" + eval_code_name, value=np.max(sentence_proba_acc))

        # Plot training and validation metrics
        logger.report_plotly(
            title="Training Metrics" + eval_code_name,
            series="train_metrics",
            # iteration=0,     # Use 0 or None for one-off plots
            figure=train_fig
        )
        # Plot sentence, token and validation accuracy
        logger.report_plotly(
            title="Accuracy Metrics" + eval_code_name,
            series="accuracy_metrics",
            # iteration=0,     # Use 0 or None for one-off plots
            figure=accuracy_fig
        )   

        # Wrap the HTML it in a stream (StringIO) and send via report_media
        html_stream = io.StringIO(html)
        logger.report_media(
            title="Token Highlight HTML" + eval_code_name,
            series="visualization",
            iteration=0,
            stream=html_stream,
            file_extension=".html",
        )

        # df_vanilla = df[df["steering_noise"] == 0.0]
        # df_steered = df[df["steering_noise"] > 0.0]
        # df_steered = df[~(df["steering_type"].isin(["vanilla", "human", "steered_bis"]))]
        # df_compared = df[(df["steering_type"].isin(["vanilla", "human", "steered_bis"]))]

        df_separated_label_list = [df[df["classification_label"] == label] for label in df["classification_label"].unique()]
        
        for classification_label, df_part in enumerate(df_separated_label_list):

            # Plot the Key vetcor for all classificated parts. This will be a zero vector for non steered texts
            key_vector = df_part["key_vector"].iloc[0]
            logger.report_scatter2d(
                title="Key Vectors" + eval_code_name,
                series="steering_key_vector label:" + str(classification_label),
                scatter=[[i, key_vector[i]] for i in range(len(key_vector))],
                iteration=0,
                xaxis="Dimension",
                yaxis="Value",
                mode='lines+markers'
            )


            print(f"Reporting quality metrics for label {classification_label} texts...")
            # Predicted Quality of text
            classified_quality = df_part["quality"].mean()[0]
            print("Classified quality of text:", classified_quality)
            logger.report_single_value("Classified quality for text labeled" + eval_code_name + str(classification_label), value=classified_quality)
            logger.report_scatter2d(
                title="Quality vs Accuracy" + eval_code_name,
                series=str(classification_label),
                iteration=0,
                scatter=np.array([[classified_quality, np.max(sentence_acc)]]),
                xaxis="Text Quality",
                yaxis="Sentence Accuracy",
                mode='lines+markers'
            )

            # Average Perplexity of text
            ppl = df_part["perplexity"].mean()
            print("Average Perplexity of text:", ppl)
            logger.report_single_value("Average Perplexity for label" + eval_code_name + str(classification_label), value=ppl)
            logger.report_scatter2d(
                title="Accuracy vs Perplexity" + eval_code_name,
                series=str(classification_label),
                iteration=0,
                scatter=np.array([[ppl, np.max(sentence_acc)]]),
                xaxis="Perplexity",
                yaxis="Sentence Accuracy",
                mode='lines+markers'
            )

            # Average log diversity of text
            log_diversity = df_part["log_diversity"].mean()
            print("Average Log Diversity of text:", log_diversity)
            logger.report_single_value("Average Log Diversity for label" + eval_code_name + str(classification_label), value=log_diversity)
            logger.report_scatter2d(
                title="Accuracy vs Log Diversity" + eval_code_name,
                series=str(classification_label),
                iteration=0,
                scatter=np.array([[log_diversity, np.max(sentence_acc)]]),
                xaxis="Log Diversity",
                yaxis="Sentence Accuracy",
                mode='lines+markers'
            )

        # logger.upload_artifact(name="total_dataframe", artifact_object=csv_path)
    
    return evaluation_dict




#####################
#   Plot generator  #
#####################

def plot_train_metrics(train_acc, train_loss, val_acc, val_loss, param):
    # plot the loss and train accuracy for each key
    color_palette = px.colors.qualitative.Plotly
    curve_counter = 0
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Loss Memory", "Train Accuracy", "Validation Loss", "Validation Accuracy"))
    
    layers = param["steering_arguments"]["steering_layers"]
    noise_amount = param["steering_arguments"]["noise_type"]
    legend_name = f"Layer {min(layers)} to {max(layers)} - noise {noise_amount}"

    fig.add_trace(
        go.Scatter(y=train_loss, name=legend_name, 
                line=dict(color=color_palette[curve_counter % len(color_palette)]),
                showlegend=True,
                legendgroup=f"{legend_name}",
                text=f"layers: {layers}\n noise: {noise_amount}",
                hoverinfo="text"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=train_acc, name=legend_name,
                line=dict(color=color_palette[curve_counter % len(color_palette)]),
                showlegend=False,
                legendgroup=f"{legend_name}",
                text=f"layers: {layers}\n noise: {noise_amount}",
                hoverinfo="y"),
        row=1, col=2
    ) 
    fig.add_trace(
        go.Scatter(y=val_loss, name=legend_name,
                line=dict(color=color_palette[curve_counter % len(color_palette)]),
                showlegend=False,
                legendgroup=f"{legend_name}",
                text=f"layers: {layers}\n noise: {noise_amount}",
                hoverinfo="text"),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(y=val_acc, name=legend_name,
                line=dict(color=color_palette[curve_counter % len(color_palette)]),
                showlegend=False,
                legendgroup=f"{legend_name}",
                text=f"layers: {layers}\n noise: {noise_amount}",
                hoverinfo="y"),
        row=2, col=2
    )
    curve_counter += 1

    fig.update_layout(title_text="Loss and Train Accuracy for each Key")
    # fig.show()
    return fig


def plot_accuracy_metrics(detection_dictionary, token_place_acc, sentence_acc, param):
    color_palette = px.colors.qualitative.Plotly
    curve_counter = 0
    # 2sub figures, 1 for validation accuracy, 1 for sentence accuracy
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Validation Accuracy", "Token wise accuracy", "Sentence Accuracy"))
    
    layers = param["steering_arguments"]["steering_layers"]
    noise_amount = param["steering_arguments"]["noise_type"]
    legend_name = f"Layer {min(layers)} to {max(layers)} - noise {noise_amount}"
    fig.add_trace(
        go.Scatter(
            x=list(range(len(detection_dictionary["validation_accuracy"]))), 
            y=detection_dictionary["validation_accuracy"], 
            line=dict(color=color_palette[curve_counter % len(color_palette)]),
            mode='lines+markers', 
            name=f"Noise: {param['steering_arguments']['noise_type']}",
            showlegend=True,
            legendgroup=f"{legend_name}",
            text=f"layers: {layers}\n noise: {noise_amount}",
            hoverinfo="text",
        ), 
        row=1, 
        col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[0, len(token_place_acc) - 1],
            y=[np.max(sentence_acc), np.max(sentence_acc)],
            mode='lines',
            line=dict(color='red', dash='dot'),
            name='Sentence Voting Accuracy Max',
            showlegend=True
        ),
        row=1,
        col=2
    )


    bin_size = len(token_place_acc) // len(sentence_acc)
    fig.add_trace(
        go.Scatter(
            x=[i*bin_size for i in range(len(sentence_acc))],
            y=sentence_acc,
            mode='lines',
            line=dict(color='yellow', dash='dot'),
            name='Sentence Voting Accuracy stepwise',
            showlegend=True
        ),
        row=1,
        col=2
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(len(token_place_acc))), 
            y=token_place_acc, 
            line=dict(color=color_palette[curve_counter % len(color_palette)]),
            mode='lines+markers', 
            name=f"Noise: {param['steering_arguments']['noise_type']}",
            showlegend=False,
            legendgroup=f"{legend_name}",
            text=f"layers: {layers}\n noise: {noise_amount}",
            hoverinfo="y",
        ), 
        row=1, 
        col=2
    )
    curve_counter += 1
    # fig.show()
    return fig



#######################
#   HTML generators   #
#######################


# def get_color_from_probs(probsXd):
#     """
#     Returns an RGB color in CSS rgba() format.
#     -1 = full blue (negative)
#      0 = white (neutral)
#     +1 = full red (positive)
#     """
#     neg, pos = probsXd
#     val = pos - neg  # [-1, +1]
#     val = max(-1, min(1, val))  # clip just in case

#     if val < 0:
#         # blend between blue (0,0,255) and white (255,255,255)
#         ratio = (val + 1)  # goes 0→1 as val -1→0
#         r = int(255 * ratio)
#         g = int(255 * ratio)
#         b = 255
#     else:
#         # blend between white (255,255,255) and red (255,0,0)
#         ratio = 1 - val  # goes 1→0 as val 0→1
#         r = 255
#         g = int(255 * ratio)
#         b = int(255 * ratio)

#     return f"rgba({r},{g},{b},0.9)"  # 0.9 transparency for nicer effect

def get_color_from_probs(probs):
    """
    Given a probability vector for n classes, returns a blended RGBA color.

    Strategy:
    - Assign each class a distinct hue uniformly spaced in [0,1].
    - Convert hue to RGB.
    - Linearly blend all class-colors weighted by probability.
    """
    n = len(probs)
    if n == 0:
        return "rgba(255,255,255,0.9)"  # fallback white

    # Normalize (in case probabilities are unnormalized scores)
    s = sum(probs)
    if s == 0:
        return "rgba(255,255,255,0.9)"
    probs = [p / s for p in probs]

    # Precompute a color per class using evenly spaced hues
    class_rgbs = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)  # vivid color
        class_rgbs.append((r, g, b))

    # Weighted blend
    r = sum(p * c[0] for p, c in zip(probs, class_rgbs))
    g = sum(p * c[1] for p, c in zip(probs, class_rgbs))
    b = sum(p * c[2] for p, c in zip(probs, class_rgbs))

    # convert 0–1 floats to 0–255 ints
    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)

    return f"rgba({r},{g},{b},0.9)"



def sentence_html_hitmap(df_test, params, token_wise_sentence_ids, token_wise_ground_truth, probabilities):

    # Build the HTML
    html = ""
    for tokens, input_text_id, input_text, label_int, sentence_quality in zip(df_test["output_token_strings"], df_test["input_text_id"], df_test["input_text"], df_test["classification_label"], df_test["quality"]):

        # Get the probabilities for the tokens in this sentence
        this_sentence_ids = np.where(np.array(token_wise_sentence_ids) == input_text_id)
        this_label_ids = np.where(np.array(token_wise_ground_truth) == label_int)
        this_sentence_ids = np.intersect1d(this_sentence_ids, this_label_ids)
        token_probs = np.array(probabilities)[this_sentence_ids]


        html += "<br> Sentence ID: " + str(input_text_id) + " label: " + str(label_int) + " Predicted proba:" + str(np.mean(token_probs, axis=0)) + " sentence quality: " + str(sentence_quality) + "<br>"

        # html += f'<span style="background-color:rgba(255,255,0,0.9); padding:2px 0px; margin:0px; border-radius:0px;">{"User query:" + input_text}</span>'
        html += (
            '<span style="white-space: pre-line; background-color:rgba(255,255,0,0.9);">'
            f'User query:{input_text}\n'
            '</span>'
        )

        for token, token_prob in zip(tokens, token_probs):
            if token == "<s>":
                token = "[START TOKEN]"
            color_str = get_color_from_probs(token_prob)
            html += f'<span style="background-color:{color_str}; padding:2px 0px; margin:0px; border-radius:0px;">{token}</span>'
        html += "<br><br>"
    
    return html



##########################
#   Accuracy calculators
##########################

def token_place_evaluation(model, df_test, do_plot=True, verbose=True):
    X_test, Y_test = df_test["fwd_data"].values, df_test["classification_label"].values.astype(np.int64)
    X_test = np.array([np.array(x, dtype=np.float32) for x in X_test])

     # Get predictions
    predictions = model.predict(X_test)

    # Calculate accuracy
    if verbose:
        accuracy = accuracy_score(Y_test, predictions)
        print("Overall token-wise accuracy:", accuracy)
    
    token_wise_accuracy = token_wise_accuracy(predictions, Y_test, df_test["token_id"].values)

    if do_plot:
        # Plot the token wise accuracy 
        plt.figure(figsize=(10, 5))
        plt.plot(token_wise_accuracy, marker='o')
        plt.title("Token Wise Accuracy")
        plt.xlabel("Token Position")
        plt.ylabel("Accuracy")
        plt.grid()
        plt.show()

    return token_wise_accuracy


def token_wise_accuracy(predictions, true_labels, token_ids):
    token_wise_accuracy = np.zeros(np.max(token_ids) + 1)
    token_counter = np.zeros(np.max(token_ids) + 1)

    for i in range(len(token_ids)):
        token_id = token_ids[i]
        if predictions[i] == true_labels[i]:
            token_wise_accuracy[token_id] += 1
        token_counter[token_id] += 1

    token_wise_accuracy = token_wise_accuracy / token_counter
    return token_wise_accuracy

def last_token_accuracy(predictions, true_labels, token_ids):
    last_token_indices = []
    sentence_length = []
    for i in range(1, len(token_ids)):
        if token_ids[i] < token_ids[i - 1]:
            # print("indice of the last token of a sentence:", i - 1)
            # print(token_ids[i - 1], "is last token of a sentence.")
            # print("Position of the first token of the new sentence:", i, "with token id:", token_ids[i])
            # print("Prediction:", predictions[i - 1], "True label:", true_labels[i - 1])
            # print("")
            last_token_indices.append(i - 1)
            sentence_length.append(token_ids[i - 1] + 1)
    last_token_indices.append(len(token_ids) - 1)  # last token of the last sentence

    last_token_predictions = [predictions[i] for i in last_token_indices]
    last_token_true_labels = [true_labels[i] for i in last_token_indices]

    print("Sentence lengths for last token accuracy calculation:", np.mean(sentence_length))

    accuracy = accuracy_score(last_token_true_labels, last_token_predictions)
    return accuracy


def sentence_evaluation(model, df_test, min_id_vote=5, verbose=True):
    # Prepare the data for sentence level accuracy
    # Deprecated, use sentence_wise_accuracy_all_steps or sentence_wise_accuracy instead
    X_test, Y_test = df_test["fwd_data"].values, df_test["classification_label"].values.astype(np.int64)
    X_test = np.array([np.array(x, dtype=np.float32) for x in X_test])

    sentence_ids = [df_test["input"].iloc[i]["input_text_id"] for i in range(len(df_test["input"]))]
    labels = df_test["classification_label"].values.astype(np.int64)

    # Get predictions
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(Y_test, predictions)

    final_accuracy = sentence_wise_accuracy(np.array(sentence_ids), labels, predictions, min_id_vote=min_id_vote, verbose=verbose)

    return final_accuracy




def sentence_wise_accuracy_all_steps(sentence_ids, labels, predictions, probabilities=None, test_token_ids=None, number_of_steps=1, verbose=True):
    # Calculate the sentence wise accuracy for different steps of minimum token id to start the vote
    vote_accuracies = []
    proba_accuracies = []
    bin_size = len(set(test_token_ids)) // number_of_steps

    print("Calculating sentence wise accuracy for", number_of_steps, "steps with bin size", bin_size)

    for step in range(number_of_steps):
        acc, proba_acc = sentence_wise_accuracy(sentence_ids, labels, predictions, probabilities, min_id_vote=step*bin_size, verbose=verbose)
        vote_accuracies.append(acc)
        proba_accuracies.append(proba_acc)
    return vote_accuracies, proba_accuracies


def sentence_wise_accuracy(sentence_ids, labels, predictions, probabilities=None, min_id_vote=5, verbose=True):
    # voting predictions based on sentence id
    unique_sentence_ids = np.unique(sentence_ids)
    sentence_predictions = np.zeros(len(labels), dtype=np.int64) - 1
    sentence_proba_vote = np.zeros(len(labels), dtype=np.int64) - 1

    sentence_level_predictions = {}

    # For each sentence id 
    for sentence_id in unique_sentence_ids:
        # And each of the labels
        sentence_level_predictions[sentence_id] = {}
        for label in np.unique(labels):
            # Mask out a single sentence of token predictions
            sentence_mask = (sentence_ids == sentence_id)
            label_mask = (labels == label)

            # print("----- Sentence wise Accuracy -----")
            # print("Sentence ID:", sentence_id, "Label:", label)
            # print("Sentence Mask:", len(sentence_mask), "Label Mask:", len(label_mask))

            # print("Input length: sentence_ids", len(sentence_ids), "labels", len(labels), "predictions", len(predictions))

            # exit()

            # Get the predictions for this sentence and label
            sentence_mask = sentence_mask & label_mask
            sentence_preds = np.array(predictions)[sentence_mask]
            sentence_probs = None
            if probabilities is not None:
                sentence_probs = np.array(probabilities)[sentence_mask]

            # Remove the first elements for the vote 
            if len(sentence_preds) <= min_id_vote:
                # raise ValueError(f"Not enough predictions for sentence ID {sentence_id} and label {label}. Minimum required: {min_id_vote}, found: {len(sentence_preds)}")
                # if verbose:
                #     print(f"Not enough predictions for sentence ID {sentence_id} and label {label}. Minimum required: {min_id_vote}, found: {len(sentence_preds)}")
                continue
            
            sentence_preds = sentence_preds[min_id_vote:]
            if sentence_probs is not None:
                sentence_probs = sentence_probs[min_id_vote:]

            # print("Sentence preds unique counts:")
            # print(np.unique(sentence_preds, return_counts=True))


            # Majority vote
            majority_vote = np.bincount(sentence_preds).argmax()
            if sentence_probs is not None:
                # Weighted vote based on probabilities
                prob_sum = np.sum(sentence_probs, axis=0)
                # print("Probability sum:", prob_sum)
                proba_vote = np.argmax(prob_sum)

            # print("Majority vote:", majority_vote)

            sentence_level_predictions[sentence_id][label] = majority_vote

            sentence_predictions[sentence_mask] = majority_vote
            sentence_proba_vote[sentence_mask] = proba_vote if sentence_probs is not None else majority_vote
            # print("This sentence ID:", sentence_id, "with label", labels[sentence_mask][0])
            # print("Predictions:", sentence_preds)
            # print("Probabilities:", sentence_probs)
            # print("Majority vote:", majority_vote)
            # print("Probability vote:", proba_vote)
            # # print("All predictions:", sentence_predictions[sentence_mask])
            # # print("True labels:", labels[sentence_mask])
            # print("-----")

    # final_accuracy = accuracy_score(labels, sentence_predictions)
    # final_proba_accuracy = accuracy_score(labels, sentence_proba_vote)

    print("Classification Report for sentence level voting accuracy:")
    sentence_level_pred = []
    sentence_level_label = []
    for sentence_id in unique_sentence_ids:
        this_labels = sentence_level_predictions[sentence_id].keys()
        for label in this_labels:
            sentence_level_pred.append(sentence_level_predictions[sentence_id][label])
            sentence_level_label.append(label)

    print(classification_report(sentence_level_label, sentence_level_pred))

    final_accuracy = accuracy_score(sentence_level_label, sentence_level_pred)

    print("Confusion Matrix for sentence voting:")
    print(confusion_matrix(sentence_level_label, sentence_level_pred))

    if verbose:
        print(f"For a min id vote of {min_id_vote}:")
        print("TODO: add the real probability voted sentence-wise accuracy calculation")
        print("Final sentence-wise accuracy:", final_accuracy)
        print("Final Probability voted sentence-wise accuracy:", final_accuracy)
    return final_accuracy, final_accuracy