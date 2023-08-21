from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn.functional as nnf
import colorama
from colorama import Back, Fore, Style
import json

def predict(tokenizer, model, masked_token, masked_sentence):
    inputs = tokenizer(masked_sentence, return_tensors="pt")
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    logits = model(**inputs).logits
    probas = nnf.softmax(logits, dim=2)
    mask_token_logits = logits[0, mask_token_index, :]
    mask_token_probas = probas[0, mask_token_index, :]
    k = 10
    top_k_tokens = torch.topk(mask_token_probas, k, dim=1)

    print(top_k_tokens)

    top_k_tpl = list(zip(top_k_tokens.indices.tolist()[0], top_k_tokens.values.tolist()[0]))

    print(masked_token)
    for token_idx, token_prob in top_k_tpl:
        print(f"prob: {token_prob:.2f} ", masked_sentence.replace(tokenizer.mask_token, Back.GREEN + tokenizer.decode([token_idx]) +Style.RESET_ALL))

models_name = {
        "bert-wm"   : "bert-large-uncased-whole-word-masking",
        "bert-base" : "bert-base-uncased",
        "xml-roberta" : "xlm-roberta-large",
        "bart-large" : "bart-large",
        "br": "bert-base-uncased-finetuned-c4200m-nationality-br",
        "distilbert-base-uncased" : "distilbert-base-uncased",
        }

model_name = models_name["bert-base"]
model_folder = f"/app/pipelines/models/{model_name}" 
model = AutoModelForMaskedLM.from_pretrained(model_folder)
tokenizer = AutoTokenizer.from_pretrained(model_folder)

dataset = "fce"
if dataset == "fce": 
    input_filename = "/app/pipelines/data/fce_dataset/fce_error_annotations.json"
    with open(input_filename) as inpf:
        fce_sentences_dict = json.loads(inpf.read())
        for sentence_dict in fce_sentences_dict.values():
            sent = sentence_dict["deannotated_sentence"]
            for annotation in sentence_dict["annotations"]:
                s, e  = annotation["span_in_DeannotatedSentence"]
                masked_token = sent[s:e]
                if masked_token == ""
                    continue
                corrected_token = annotation["correct_token"]
                masked_sentence = sent[:s] + tokenizer.mask_token + sent[e:]
                predict(tokenizer, model, masked_token, masked_sentence)
                input()
else:
        pass
