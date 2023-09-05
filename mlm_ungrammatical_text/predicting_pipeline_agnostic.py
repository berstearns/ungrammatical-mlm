from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.metrics import top_k_accuracy_score
import torch
import torch.nn.functional as nnf
import colorama
from colorama import Back, Fore, Style
import json
from datetime import datetime


debugging_examples = [
        f''
    ]

def top_k(true_token_idx, top_k_predictions, max_k=10):
    top_k_token_idxs = [idx for (idx, _) in top_k_predictions]
    top_k_flags = []
    for k in range(1, max_k+1):  
        if true_token_idx in top_k_token_idxs[:k]:
            top_k_flags.append(1)
        else:
            top_k_flags.append(0)
    return top_k_flags

def predict(tokenizer, model, masked_token, masked_sentence, printFlag=False):
    inputs = tokenizer(masked_sentence, return_tensors="pt")
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    logits = model(**inputs).logits
    probas = nnf.softmax(logits, dim=2)
    mask_token_logits = logits[0, mask_token_index, :]
    mask_token_probas = probas[0, mask_token_index, :]
    k = 10
    top_k_tokens = torch.topk(mask_token_probas, k, dim=1)

    top_k_tpls_per_masked_token = [list(zip(*tpl)) for tpl in zip(top_k_tokens.indices.tolist(), top_k_tokens.values.tolist())]
    if printFlag:
        for predictions_tpls in top_k_tpls_per_masked_token:
            for prediction_tpl in predictions_tpls:
                (token_idx, token_prob) = prediction_tpl
                print(f"prob: {token_prob:.2f} ",masked_sentence.replace(tokenizer.mask_token, Back.GREEN + tokenizer.decode([token_idx]) +Style.RESET_ALL,1))
    return top_k_tpls_per_masked_token 

def load_words(fo):
    words = set()
    for word in fo:
        word = word.replace("\n","")
        words.add(word)
    return words

if __name__ == "__main__":
    models_name = {
            "xlm-ende": "xlm-mlm-ende-1024",
            "xlm-17": "xlm-mlm-17-1280",
            "xlm-100": "xlm-mlm-100-1280",
            "bert-wm"   : "bert-large-uncased-whole-word-masking",
            "bert-base" : "bert-base-uncased",
            "bert-multilingual": "bert-base-multilingual-cased",
            "xlm-roberta" : "xlm-roberta-large",
            "bart-large" : "bart-large",
            "br": "bert-base-uncased-finetuned-c4200m-nationality-br",
            "fr": "batch-24-bert-base-uncased-2023-09-01_13-59-45",
            "cn": "bert-base-uncased-finetuned-c4200m-nationality-cn",
            "mx": "bert-base-uncased-finetuned-c4200m-nationality-mx",
            "distilbert-base-uncased" : "distilbert-base-uncased",
            "full": "batch-414-bert-base-uncased-2023-08-24_11-35-00",
            }

    model_name = models_name["xlm-ende"]
    model_folder = f"/app/pipelines/models/{model_name}" 
    tokenizer_folder = model_folder
    # tokenizer_folder = f"/app/pipelines/models/bert-base-uncased" 
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_metrics_filepath = f"/app/pipelines/metrics/{model_name}_run_{time}.json"
    try:
        model = AutoModelForMaskedLM.from_pretrained(model_folder)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_folder)
    except:
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)



    dataset = "fce"
    if dataset == "fce": 
        max_k = 30502
        instances_count = 0
        global_top_k = [0 for _ in range(max_k)]
        #"/app/pipelines/data/fce_dataset/fce_error_annotations.json"
        input_filename = "/app/pipelines/preprocessing_fce/fce_identical_error_annotations.json"
        with open(input_filename) as inpf, open("/app/pipelines/mlm_ungrammatical_text/mlm_ungrammatical_text/words.txt") as wordsf:
            words = load_words(wordsf)
            fce_sentences_dict = json.loads(inpf.read())
            for sentence_dict in fce_sentences_dict.values():
                sent = sentence_dict["deannotated_sentence"]
                for annotation in sentence_dict["annotations"]:
                    s, e  = annotation["span_in_DeannotatedSentence"]
                    masked_token = sent[s:e]
                    masked_token_idx = tokenizer.encode(masked_token)[1]
                    if not (masked_token in words): 
                        continue
                        print(s, e)
                        print(sent)
                    if annotation["match_type"] != "replacement_correction": #masked_token == "":
                        continue
                    if masked_token != annotation["incorrect_token"]:
                        continue
                    if annotation["number_of_tokens"]  >= 2:
                        continue
                    corrected_token = annotation["correct_token"]
                    masked_sentence = sent[:s] +\
                                    " ".join([tokenizer.mask_token
                                             for _ in range(annotation["number_of_tokens"])])+\
                                    sent[e:]
                    print(sent)
                    print(sent[s:e])
                    print(sentence_dict["error_annotated_sentence"])
                    print(annotation["incorrect_token"])
                    print(masked_token )
                    print(annotation["correct_token"])
                    top_k_tpls_per_masked_token = predict(tokenizer, model, masked_token, masked_sentence)
                    for top_k_tpls in top_k_tpls_per_masked_token:
                        instance_top_k_flags = top_k(masked_token_idx,top_k_tpls,max_k=max_k)
                        global_top_k = [a+b for (a,b) in zip(global_top_k, instance_top_k_flags)]
                    instances_count+=1
            top_k_acc = [v/instances_count for v in global_top_k]

            with open(model_metrics_filepath,"w") as metrics_outf:
                top_ks = {
                }
                top_ks_by_language = {
                }
                for lastIdx in [10,100,1000,10000,max_k]:
                    top_ks[lastIdx] = top_k_acc[:lastIdx]
                metrics_outf.write(json.dumps(top_ks, indent=4))
    else:
            pass
