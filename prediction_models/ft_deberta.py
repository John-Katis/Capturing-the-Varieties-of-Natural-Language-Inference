from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
# TODO fix this path somehow
model = AutoModelForSequenceClassification.from_pretrained(
    r"C:\Users\john1\Desktop\Work\DS Chair\Artist\Fine tuning transformers\deberta_ft\deberta_512padding_1e7lr_8batchsize_1epochs"
)
tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-large')


def predict_ft_deberta(hypothesis, premise):

    features = tokenizer(hypothesis, premise,  padding=True, truncation=True, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        scores = model(**features).logits
        label_mapping = ['contradiction', 'entailment', 'neutral']
        label = label_mapping[scores.argmax(dim=1)]
        return label
