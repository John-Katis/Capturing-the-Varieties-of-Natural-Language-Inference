"""
This script is written after publication and is meant to provide a way
for people interested in this work, to quickly re-calculate our metrics
and see them in detail.
"""

import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

mapping_2304 = {
    "neutral": "neutral", # this is here for completeness
    "entailment": "entails",
    "contradiction": "contradicts"
}



def load_data(path):
    return pd.read_csv(
        path,
        encoding='UTF-8',
        sep=';'
    )



def print_metrics(print_title, results_dict):
    print('\n')
    print('##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####')
    print(print_title)
    print('--- --- --- --- --- --- --- --- ---')
    print(f"{'accuracy:':<12} {results_dict['accuracy']}")
    print(f"{'precision:':<12} {results_dict['precision']}")
    print(f"{'recall:':<12} {results_dict['recall']}")
    print(f"{'fscore:':<12} {results_dict['fscore']}")
    print(f"{'support:':<12} {results_dict['support']}")
    print('##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####')



def get_metrics(true, pred):
    accuracy = accuracy_score(true, pred)
    precision, recall, fscore, support = precision_recall_fscore_support(true, pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'fscore': fscore,
        'support': support
    }



def chat_gpt_results():
    runs = ['2304_chatgpteval_1k', 'AAE_1k']
    print_titles = {
        '2304_chatgpteval_1k' : "Results for ChatGPT - Feedback Prize dataset reformed like Argument Annotated Essays - Random 1k sample",
        'AAE_1k'              : "Results for ChatGPT - Argument Annotated Essays - Random 1k sample"
    }

    for run in runs:
        data = load_data(fr'Output\chatGPT\(clean){run}_sample_with_predictions.csv').copy()

        predicted_relations = data['predicted_relation'].tolist()
        true_relations      = data['label'].tolist()

        if run == '2304_chatgpteval_1k':
            predicted_relations = [mapping_2304[x] for x in predicted_relations]
            true_relations      = [mapping_2304[x] for x in true_relations]

        results_dict = get_metrics(true_relations, predicted_relations)

        print_metrics(print_titles[run], results_dict)
        print(
            f"True entailments:{true_relations.count('entails'):>6} || True contradictions:{true_relations.count('contradicts'):>6} || True neutral:{true_relations.count('neutral'):>6}"
        )
        print(
            f"Pred entailments:{predicted_relations.count('entails'):>6} || Pred contradictions:{predicted_relations.count('contradicts'):>6} || Pred neutral:{predicted_relations.count('neutral'):>6}")
        print('##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####')



def bart_results():
    runs = ['2304_1k', 'AAE_1k', 'all_annotated', 'all_sentences']
    print_titles = {
        '2304_1k'       : "Results for Bart - Feedback Prize dataset reformed like Argument Annotated Essays - Random 1k sample",
        'AAE_1k'        : "Results for Bart - Argument Annotated Essays - Random 1k sample",
        'all_annotated' : "Results for Bart - Argument Annotated Essays - All annotated sentence pairs",
        'all_sentences' : "Results for Bart - Argument Annotated Essays - All sentence pairs (+ not annotated ones)"
    }

    for run in runs:
        data = load_data(fr'Output\bart\true_and_pred_relations({run}).csv').copy()

        predicted_relations = data['pred_label'].tolist()
        true_relations      = data['true_label'].tolist()

        if run == '2304_1k':
            true_relations = [mapping_2304[x] for x in true_relations]

        results_dict = get_metrics(true_relations, predicted_relations)

        print_metrics(print_titles[run], results_dict)
        print(
            f"True entailments:{true_relations.count('entails'):>6} || True contradictions:{true_relations.count('contradicts'):>6} || True neutral:{true_relations.count('neutral'):>6}"
        )
        print(
            f"Pred entailments:{predicted_relations.count('entails'):>6} || Pred contradictions:{predicted_relations.count('contradicts'):>6} || Pred neutral:{predicted_relations.count('neutral'):>6}")
        print('##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####')



def deberta_results():
    runs = ['2304_1k', 'AAE_1k', 'all_sentences']
    print_titles = {
        '2304_1k'       : "Results for Deberta - Feedback Prize dataset reformed like Argument Annotated Essays - Random 1k sample",
        'AAE_1k'        : "Results for Deberta - Argument Annotated Essays - Random 1k sample",
        'all_sentences' : "Results for Deberta - Argument Annotated Essays - All sentence pairs (+ not annotated ones)"
    }

    for run in runs:
        data = load_data(fr'Output\deberta\true_and_pred_relations({run}).csv').copy()

        predicted_relations = data['pred_label'].tolist()
        true_relations      = data['true_label'].tolist()

        if run == '2304_1k':
            true_relations = [mapping_2304[x] for x in true_relations]

        results_dict = get_metrics(true_relations, predicted_relations)

        print_metrics(print_titles[run], results_dict)
        print(
            f"True entailments:{true_relations.count('entails'):>6} || True contradictions:{true_relations.count('contradicts'):>6} || True neutral:{true_relations.count('neutral'):>6}"
        )
        print(
            f"Pred entailments:{predicted_relations.count('entails'):>6} || Pred contradictions:{predicted_relations.count('contradicts'):>6} || Pred neutral:{predicted_relations.count('neutral'):>6}")
        print('##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####')



def ft_deberta_results():
    runs = ['2304_1k_128p_1e3lr_8bs_3ep', '2304_1k_128p_5e5lr_4bs_3ep']
    print_titles = {
        '2304_1k_128p_1e3lr_8bs_3ep': """Results of fine tuned Deberta (on argument annotated essays) - Evaluation for Feedback Prize dataset reformed like Argument Annotated Essays - Random 1k sample
    Model Params:
        input tokens  : 128
        learning rate : 1e3
        batch size    : 8
        ft epochs     : 3""" ,
        '2304_1k_128p_5e5lr_4bs_3ep': """Results of fine tuned Deberta (on argument annotated essays) - Evaluation for Feedback Prize dataset reformed like Argument Annotated Essays - Random 1k sample
    Model Params:
        input tokens  : 128
        learning rate : 5e5
        batch size    : 4
        ft epochs     : 3""" ,
    }

    for run in runs:
        data = load_data(fr'Output\ft_deberta\true_and_pred_relations({run}).csv').copy()

        predicted_relations = data['pred_label'].tolist()
        true_relations      = data['true_label'].tolist()
        true_relations = [mapping_2304[x] for x in true_relations]

        results_dict = get_metrics(true_relations, predicted_relations)

        print_metrics(print_titles[run], results_dict)
        print(
            f"True entailments:{true_relations.count('entails'):>6} || True contradictions:{true_relations.count('contradicts'):>6} || True neutral:{true_relations.count('neutral'):>6}"
        )
        print(
            f"Pred entailments:{predicted_relations.count('entails'):>6} || Pred contradictions:{predicted_relations.count('contradicts'):>6} || Pred neutral:{predicted_relations.count('neutral'):>6}")
        print('##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####')



def miniLM2_results():
    runs = ['2304_1k', 'AAE_1k', 'all_sentences']
    print_titles = {
        '2304_1k'       : "Results for MiniLM2 - Feedback Prize dataset reformed like Argument Annotated Essays - Random 1k sample",
        'AAE_1k'        : "Results for MiniLM2 - Argument Annotated Essays - Random 1k sample",
        'all_sentences' : "Results for MiniLM2 - Argument Annotated Essays - All sentence pairs (+ not annotated ones)"
    }

    for run in runs:
        data = load_data(fr'Output\miniLM2\true_and_pred_relations({run}).csv').copy()

        predicted_relations = data['pred_label'].tolist()
        true_relations      = data['true_label'].tolist()

        if run == '2304_1k':
            true_relations = [mapping_2304[x] for x in true_relations]

        results_dict = get_metrics(true_relations, predicted_relations)

        print_metrics(print_titles[run], results_dict)
        print(
            f"True entailments:{true_relations.count('entails'):>6} || True contradictions:{true_relations.count('contradicts'):>6} || True neutral:{true_relations.count('neutral'):>6}"
        )
        print(
            f"Pred entailments:{predicted_relations.count('entails'):>6} || Pred contradictions:{predicted_relations.count('contradicts'):>6} || Pred neutral:{predicted_relations.count('neutral'):>6}")
        print('##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####')



def ft_miniLM2_results():
    runs = [
        '2304_1k_512p_1e3lr_8bs_3e',
        '2304_1k_512p_1e3lr_16bs_3e',
        '2304_1k_512p_1e4lr_32bs_1e',
        '2304_1k_512p_1e5lr_32bs_1e',
        '2304_1k_512p_1e6lr_32bs_1e',
        '2304_1k_512p_1e7lr_4bs_1e',
        '2304_1k_512p_1e7lr_4bs_3e',
        '2304_1k_512p_1e7lr_32bs_3e',
        '2304_1k_512p_1e9lr_4bs_1e',
        '2304_1k_512p_1e9lr_16bs_5e',
        '2304_1k_512p_1e9lr_32bs_1e',
        '2304_1k_512p_5e5lr_4bs_3e'
    ]

    print_titles = {
        '2304_1k_512p_1e3lr_8bs_3e': """Results of fine tuned MiniLM2 (on argument annotated essays) - Evaluation for Feedback Prize dataset reformed like Argument Annotated Essays - Random 1k sample
    Model Params:
        input tokens  : 512
        learning rate : 1e3
        batch size    : 8
        ft epochs     : 3""" ,
        '2304_1k_512p_1e3lr_16bs_3e': """Results of fine tuned MiniLM2 (on argument annotated essays) - Evaluation for Feedback Prize dataset reformed like Argument Annotated Essays - Random 1k sample
    Model Params:
        input tokens  : 512
        learning rate : 1e3
        batch size    : 16
        ft epochs     : 3""" ,
        '2304_1k_512p_1e4lr_32bs_1e': """Results of fine tuned MiniLM2 (on argument annotated essays) - Evaluation for Feedback Prize dataset reformed like Argument Annotated Essays - Random 1k sample
    Model Params:
        input tokens  : 512
        learning rate : 1e4
        batch size    : 32
        ft epochs     : 1""" ,
        '2304_1k_512p_1e5lr_32bs_1e': """Results of fine tuned MiniLM2 (on argument annotated essays) - Evaluation for Feedback Prize dataset reformed like Argument Annotated Essays - Random 1k sample
    Model Params:
        input tokens  : 512
        learning rate : 1e5
        batch size    : 32
        ft epochs     : 1""" ,
        '2304_1k_512p_1e6lr_32bs_1e': """Results of fine tuned MiniLM2 (on argument annotated essays) - Evaluation for Feedback Prize dataset reformed like Argument Annotated Essays - Random 1k sample
    Model Params:
        input tokens  : 512
        learning rate : 1e6
        batch size    : 32
        ft epochs     : 1""" ,
        '2304_1k_512p_1e7lr_4bs_1e': """Results of fine tuned MiniLM2 (on argument annotated essays) - Evaluation for Feedback Prize dataset reformed like Argument Annotated Essays - Random 1k sample
    Model Params:
        input tokens  : 512
        learning rate : 1e7
        batch size    : 4
        ft epochs     : 1""" ,
        '2304_1k_512p_1e7lr_4bs_3e': """Results of fine tuned MiniLM2 (on argument annotated essays) - Evaluation for Feedback Prize dataset reformed like Argument Annotated Essays - Random 1k sample
    Model Params:
        input tokens  : 512
        learning rate : 1e7
        batch size    : 4
        ft epochs     : 3""" ,
        '2304_1k_512p_1e7lr_32bs_3e': """Results of fine tuned MiniLM2 (on argument annotated essays) - Evaluation for Feedback Prize dataset reformed like Argument Annotated Essays - Random 1k sample
    Model Params:
        input tokens  : 512
        learning rate : 1e7
        batch size    : 32
        ft epochs     : 3""" ,
        '2304_1k_512p_1e9lr_4bs_1e': """Results of fine tuned MiniLM2 (on argument annotated essays) - Evaluation for Feedback Prize dataset reformed like Argument Annotated Essays - Random 1k sample
    Model Params:
        input tokens  : 512
        learning rate : 1e9
        batch size    : 4
        ft epochs     : 1""" ,
        '2304_1k_512p_1e9lr_16bs_5e': """Results of fine tuned MiniLM2 (on argument annotated essays) - Evaluation for Feedback Prize dataset reformed like Argument Annotated Essays - Random 1k sample
    Model Params:
        input tokens  : 512
        learning rate : 1e9
        batch size    : 16
        ft epochs     : 5""" ,
        '2304_1k_512p_1e9lr_32bs_1e': """Results of fine tuned MiniLM2 (on argument annotated essays) - Evaluation for Feedback Prize dataset reformed like Argument Annotated Essays - Random 1k sample
    Model Params:
        input tokens  : 512
        learning rate : 1e9
        batch size    : 32
        ft epochs     : 1""" ,
        '2304_1k_512p_5e5lr_4bs_3e': """Results of fine tuned MiniLM2 (on argument annotated essays) - Evaluation for Feedback Prize dataset reformed like Argument Annotated Essays - Random 1k sample
    Model Params:
        input tokens  : 512
        learning rate : 5e5
        batch size    : 4
        ft epochs     : 3""" ,
    }

    for run in runs:
        data = load_data(fr'Output\ft_miniLM2\true_and_pred_relations({run}).csv').copy()

        predicted_relations = data['pred_label'].tolist()
        true_relations      = data['true_label'].tolist()
        predicted_relations = [mapping_2304[x] for x in predicted_relations]
        true_relations      = [mapping_2304[x] for x in true_relations]

        results_dict = get_metrics(true_relations, predicted_relations)

        print_metrics(print_titles[run], results_dict)
        print(
            f"True entailments:{true_relations.count('entails'):>6} || True contradictions:{true_relations.count('contradicts'):>6} || True neutral:{true_relations.count('neutral'):>6}"
        )
        print(
            f"Pred entailments:{predicted_relations.count('entails'):>6} || Pred contradictions:{predicted_relations.count('contradicts'):>6} || Pred neutral:{predicted_relations.count('neutral'):>6}")
        print('##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####')



def all_results():
    print("================================")
    print("===== RESULTS FOR CHAT GPT =====")
    print("================================")
    print()
    chat_gpt_results()
    print()
    print("=============================")
    print("===== RESULTS FOR BART =====")
    print("============================")
    print()
    bart_results()
    print()
    print("===============================")
    print("===== RESULTS FOR DEBERTA =====")
    print("===============================")
    print()
    deberta_results()
    print()
    print("===============================")
    print("===== RESULTS FOR MINILM2 =====")
    print("===============================")
    print()
    miniLM2_results()
    print()
    print("====================================")
    print("===== RESULTS FOR DEBERTA (ft) =====")
    print("====================================")
    print()
    ft_deberta_results()
    print()
    print("====================================")
    print("===== RESULTS FOR MINILM2 (ft) =====")
    print("====================================")
    print()
    ft_miniLM2_results()



# chat_gpt_results()
# bart_results()
# deberta_results()
# miniLM2_results()
# ft_deberta_results()
# ft_miniLM2_results()
all_results()
