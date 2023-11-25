import os
import openai
import pandas as pd
import pickle

openai.api_key = os.getenv("OPENAI_API_KEY")


def request_label(prompt):
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )

    # Extract the message content from the response
    return response["choices"][0]["message"]["content"]


def process_df(relations_df, index):

    result = []

    for i in range(index, relations_df.shape[0]):
        prompt = f'In one word, is the relation between the sentences "{relations_df.iloc[i]["sentence1"]}" and "{relations_df.iloc[i]["sentence2"]}" an entailment, contradiction or neutral?'

        pred_label = request_label(prompt)
        result.append(pred_label)
        print(pred_label)

        with open(r"C:\Users\john1\PycharmProjects\chatGPT_test\Output\counter.txt", 'w') as f_c:
            f_c.write(str(i))

    return result


o_data = pd.read_csv(
    r"C:\Users\john1\PycharmProjects\chatGPT_test\Data\2304_chatgpteval_1ksample.csv",
    encoding='UTF-8', sep=';',
    names=["id", "sentence1", "sentence2", "label"]
)
data = o_data.copy()

with open(r"C:\Users\john1\PycharmProjects\chatGPT_test\Output\counter.txt", 'r') as f_cin:
    index = int(f_cin.read())

predictions = process_df(data, index)

with open(r'C:\Users\john1\PycharmProjects\chatGPT_test\Output\2304_chatgpteval_raw_predictions.pkl', 'wb') as handle:
    pickle.dump(predictions, handle)

data["predicted_relation"] = predictions

data.to_csv(
    r"C:\Users\john1\PycharmProjects\chatGPT_test\Output\2304_chatgpteval_1k_sample_with_predictions.csv",
    encoding='UTF-8',
    sep=';'
)



#print(data.head())

#prompt = f'In one word, is the relation between the sentences "{data.iloc[0]["sentence1"]}" and "{data.iloc[0]["sentence2"]}" an entailment, contradiction or neutral?'

#print(request_label(prompt))
