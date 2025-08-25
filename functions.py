import json
from collections import Counter
import pandas as pd
from sklearn.metrics import f1_score

def load_jsonl(file_path):
    """
    Load a JSONL file and return a list of JSON objects.
    :param file_path: str, path to the JSONL file
    :return: list of dicts, each representing a JSON object
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def assign_bin_maj(item, is_test=False):
    """
    takes a tweet and its annotations (if available) and computes 1 if a majority of annotators assigned a label other than 0-Kein, predicts 0 if a majority assigned 0-Kein. If there was no majority, either label is considered correct for evaluation.
    :param item: dictionary of the form {'id': , 'text': , 'annotators': }
    :param is_test: Boolean. If False annotations are available. If True not
    :return: label
    """
    if not is_test:
        labels = [ann['label'] for ann in item['annotations']]
        label_counts = Counter(labels)
        majority_label, majority_count = label_counts.most_common(1)[0]
        bin_maj_label = 1 if majority_label != '0-Kein' else 0
    else:
        bin_maj_label = None
    return bin_maj_label


def assign_bin_one(item, is_test=False):
    """
    takes a tweet and its annotations (if available) and computes 1 if at least one annotator assigned a label other than 0-Kein, 0 otherwise.
    :param item: dictionary of the form {'id': , 'text': , 'annotators': }
    :param is_test: Boolean. If False annotations are available. If True not
    :return: label
    """
    if not is_test:
        bin_one_label = 1 if any(ann['label'] != '0-Kein' for ann in item['annotations']) else 0
    else:
        bin_one_label = None
    return bin_one_label


def assign_bin_all(item, is_test=False):
    """
    takes a tweet and its annotations (if available) and computes 1 if all annotators assigned labels other than 0-Kein, 0 otherwise.
    :param item: dictionary of the form {'id': , 'text': , 'annotators': }
    :param is_test: Boolean. If annotations are available. If True not
    :return: label
    """
    if not is_test:
        bin_all_label = 1 if all(ann['label'] != '0-Kein' for ann in item['annotations']) else 0
    else:
        bin_all_label = None
    return bin_all_label


def assign_multi_maj(item, is_test=False):
    """
    takes a tweet and its annotations (if available) and predicts the majority label if there is one, if there is no majority label, any of the labels assigned is counted as a correct prediction for evaluation.
    :param item: dictionary of the form {'id': , 'text': , 'annotators': }
    :param is_test: Boolean. If False annotations are available. If True not
    :return: label
    """
    if not is_test:
        labels = [ann['label'] for ann in item['annotations']]
        label_counts = Counter(labels)
        majority_label, majority_count = label_counts.most_common(1)[0]
        multi_maj_label = majority_label if majority_count > len(labels) / 2 else labels[0]
        multi_maj_label = int(multi_maj_label.split('-')[0])
    else:
        multi_maj_label = None
    return multi_maj_label


def assign_disagree_bin(item, is_test=False):
    """
    takes a tweet and its annotations (if available) and predicts 1 if there is a disagreement between annotators on 0-Kein versus all other labels and 0 otherwise.
    :param item: dictionary of the form {'id': , 'text': , 'annotators': }
    :param is_test: Boolean. If False annotations are available. If True not
    :return: label
    """
    if not is_test:
        labels = [ann['label'] for ann in item['annotations']]
        unique_labels = set(labels)
        disagree_bin_label = 1 if '0-Kein' in unique_labels and len(unique_labels) > 1 else 0
    else:
        disagree_bin_label = None
    return disagree_bin_label


def total_data(item, is_test=False):
    """
    collects all labels described above for one tweet.
    :param item: dictionary of the form {'id': , 'text': , 'annotators': }
    :param is_test: Boolean. If False annotations are available. If True not
    :return: dictionary of the form {'id': , 'text': , 'bin_maj_label':, 'bin_one_label': , ... }
    """
    text = item['text']
    text = text.replace('\n', ' ')
    return {'id': item['id'], 'text': text, 'bin_maj_label': assign_bin_maj(item), 'bin_one_label': assign_bin_one(item),
            'bin_all_label': assign_bin_all(item), 'multi_maj_label': assign_multi_maj(item),
            'disagree_bin_label': assign_disagree_bin(item)}


def combine_data(data, dataframe=False):
    """
    iterates over a list of tweets and annotations
    :param data: list of dictionaries
    :return: list of dictionaries or dataframe
    """
    data_with_labels = [total_data(item) for item in data]
    if dataframe:
        header = data_with_labels[0].keys()
        data_with_labels = pd.DataFrame(data_with_labels, columns=header)
    return data_with_labels


def compute_metrics(lbls, preds):
    """
    Computes F1 score
    :param lbls: list of ground truth labels
    :param preds: list of predicted labels
    :return: dictionary of F1 score
    """
    f1 = f1_score(lbls, preds, average='weighted')
    return {'f1': f1}


def compute_f1(real_data, prediction):
    """
    Prints F1 scores for our data
    :param real_data: dataframe containing our ground truth data
    :param prediction: dataframe containing our predicted data
    :return: no return, just print statement
    """
    print(f"Dev set F1 score Bin Maj: {compute_metrics(real_data['bin_maj_label'], prediction['bin_maj_label'])['f1']}")
    print(f"Dev set F1 score Bin One: {compute_metrics(real_data['bin_one_label'], prediction['bin_one_label'])['f1']}")
    print(f"Dev set F1 score Bin All: {compute_metrics(real_data['bin_all_label'], prediction['bin_all_label'])['f1']}")
    print(f"Dev set F1 score Multi Maj: {compute_metrics(real_data['multi_maj_label'], prediction['multi_maj_label'])['f1']}")
    print(
        f"Dev set F1 score Disagree Bin: {compute_metrics(real_data['disagree_bin_label'], prediction['disagree_bin_label'])['f1']}")