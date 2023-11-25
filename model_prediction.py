import pickle
import pandas as pd
#from bart import bart_predict_relations
#from deberta import deberta_predict_relations
#from ft_deberta import predict_ft_deberta
from ft_minilm2 import predict_ft_minilm2
#from minilm2 import miniLM2_predict_relations

all_data_with_predictions = []

#o_data = pd.read_csv(
#    r"C:\Users\john1\PycharmProjects\chatGPT_test\Output\bart\true_and_pred_relations(all_sentences).csv",
#    encoding='UTF-8',
#    sep=';',
#    index_col=0
#)

#o_data = pd.read_csv(
#    r"C:\Users\john1\PycharmProjects\chatGPT_test\Data\AAE_1k_sample.csv",
#    encoding='UTF-8',
#    sep=';'
#)

#o_data = pd.read_csv(
#    r"C:\Users\john1\PycharmProjects\chatGPT_test\Data\2304_chatgpteval_1ksample.csv",
#    encoding='UTF-8',
#    sep=';',
#    names=["id", "sentence1", "sentence2", "label"]
#)

o_data = pd.read_csv(
    r"C:\Users\john1\PycharmProjects\chatGPT_test\Data\RTE_all_rows(2767_rows).csv",
    encoding='UTF-8',
    sep=';'
)

RTE_mapping = {
    "entailment": "entailment",
    "contradiction": "not_entailment",
    "neutral": "not_entailment"
}

data_copy = o_data.copy()

for index, row in data_copy.iterrows():
    print(index)
    pred_label = predict_ft_minilm2(row["sentence1"], row["sentence2"])
    all_data_with_predictions.append([row["sentence1"], row["sentence2"], row["label"], RTE_mapping[pred_label]])

new_df = pd.DataFrame(all_data_with_predictions, columns=["sentence1", "sentence2", "true_label", "pred_label"])
new_df.to_csv(
    r"C:\Users\john1\PycharmProjects\chatGPT_test\Output\ft_miniLM2\interim_evaluation\true_and_pred_relations(RTE_512p_1e7lr_32bs_1e).csv",
    encoding='UTF-8',
    sep=';'
)

with open(r"C:\Users\john1\PycharmProjects\chatGPT_test\Output\ft_miniLM2\interim_evaluation\true_and_pred_relations(RTE_512p_1e7lr_32bs_1e).pkl", 'wb') as f_out:
    pickle.dump(all_data_with_predictions, f_out)
