"""

Creating the dataset to use for fine tuning
- All contradictions and entailments = 3414 sentence pairs
- 10k neutral pairs

"""



import pickle
import random
import pandas as pd

with open(r"C:\Users\john1\PycharmProjects\chatGPT_test\Data\AAE_all_sentences.pkl", "rb") as handle:
    relations = pickle.load(handle)

contradictions_list = []
entailments_list = []
neutral_list = []

for rel in relations:
    if rel[2] == "entails":
        entailments_list.append(rel)
    if rel[2] == "contradicts":
        contradictions_list.append(rel)
    if rel[2] == "neutral":
        if len(neutral_list) < 10000:
            neutral_list.append(rel)

print(len(contradictions_list))
print(len(entailments_list))
print(len(neutral_list))

ft_df = contradictions_list + entailments_list + neutral_list
random.shuffle(ft_df)
print(type(ft_df))

ltn_dict = {'contradicts': 0, 'entails': 1, 'neutral': 2}

for line in ft_df:
    line[2] = ltn_dict[line[2]]

to_save = pd.DataFrame(ft_df, columns=['sentence1', 'sentence2', 'label'])
to_save.to_csv(r"C:\Users\john1\desktop\ft_ds.csv", encoding='UTF-8', sep=';')
