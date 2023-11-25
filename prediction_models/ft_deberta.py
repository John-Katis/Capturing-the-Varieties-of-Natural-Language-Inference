from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained(
    r"ft_model_weights\deberta_512padding_1e7lr_8batchsize_1epochs"
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
