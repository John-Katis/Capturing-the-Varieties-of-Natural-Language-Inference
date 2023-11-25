from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained(
    r"C:\Users\john1\Desktop\Work\DS Chair\Artist\Fine tuning transformers\minilm2_ft\minilm2_512padding_1e7lr_32batchsize_1epochs"
)
tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-MiniLM2-L6-H768')


def predict_ft_minilm2(hypothesis, premise):

    features = tokenizer(hypothesis, premise,  padding=True, truncation=True, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        scores = model(**features).logits
        label_mapping = ['contradiction', 'entailment', 'neutral']
        label = label_mapping[scores.argmax(dim=1)]
        return label