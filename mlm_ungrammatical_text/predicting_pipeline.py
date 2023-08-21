from transformers import pipeline
import json
from sklearn.metrics import top_k_accuracy_score
from collections import defaultdict
import random
import colorama
from colorama import Back, Fore, Style

def top_k_metric(original_token, arr_topk_tokens):
    if original_token in arr_topk_tokens: return 1
    else: return 0

def top_1_100_metrics(original_token, arr_top5):
    top_100_acc = []
    for i in range(100):
        top_100_acc.append(top_k_metric(original_token, arr_top5[:i+1]))
    return top_100_acc

def sample_sentences(fce_sentences_dict):
    return random.sample(list(fce_sentences_dict.values()),10)

Style.RESET_ALL 
input_filename = "../data/fce_dataset/fce_error_annotations.json"
# input_filename = "input.json"
model_name ="bert-large-uncased-whole-word-masking"
# "bert-base-uncased"
fill_masker = pipeline(model=model_name)
with open(input_filename) as inpf:
    fce_sentences_dict = json.loads(inpf.read())
    incorrectToken_top_k_metrics = defaultdict(int)
    correctToken_top_k_metrics = defaultdict(int)
    counts = {
            "total_tokens": 0,
            "masked_tokens": 0,
            "empty_tokens": 0,
            "spaced_tokens": 0,
            "len1_tokens": 0
            }
    for sentence_dict in fce_sentences_dict.values():
        sent = sentence_dict["deannotated_sentence"]
        for annotation in sentence_dict["annotations"]:
            counts["total_tokens"] += 1
            s, e  = annotation["span_in_DeannotatedSentence"]
            #print(s, e)
            masked_token = sent[s:e]
            corrected_token = annotation["correct_token"]
            masked_sentence = sent[:s] + "[MASK]" + sent[e:]
            colored_masked_sentence = Fore.WHITE + sent[:s] + Style.RESET_ALL + Back.GREEN + "[MASK]" + Style.RESET_ALL + sent[e:]  
            colored_masked_token = Back.GREEN + masked_token + Style.RESET_ALL
            colored_correct_token = Back.BLUE + corrected_token + Style.RESET_ALL
            if masked_token == "": 
               counts["empty_tokens"] += 1
               continue 
            if len(masked_token) == 1: 
               counts["len1_tokens"] += 1
               continue 
            if (" " in masked_token):
               print(annotation)
               counts["spaced_tokens"] += 1
               print(sent)
               print("MASKED SENTENCE: ",colored_masked_sentence)
               print("MASKED Token:\t",colored_masked_token,colored_correct_token )
               input()
               continue 
            counts["masked_tokens"] += 1
            #print("MASKED SENTENCE: ",colored_masked_sentence)
            #print("MASKED Token:\t",colored_masked_token,colored_correct_token )
            #print("ORIGINAL SENTENCE:",sent)
            #print("MASKED TOKEN:\t", masked_token)
            incorrectToken_scores = fill_masker(masked_sentence,top_k=100)
            #for d in scores[:5]:
            #   print(f'{Fore.RED} {d["token_str"]} {d["score"]} ')
            #print([{"s":d["score"],"t":d["token_str"]} for d in scores[:10]])
            #print(f'{Fore.RED} {masked_token} {Style.RESET_ALL}')
            for i, v in enumerate(top_1_100_metrics(masked_token, [d["token_str"] for d in incorrectToken_scores[:5]])):
               incorrectToken_top_k_metrics[i] += v
            for i, v in enumerate(top_1_100_metrics(corrected_token, [d["token_str"] for d in incorrectToken_scores[:5]])):
               correctToken_top_k_metrics[i] += v

print(counts)
print({k:v/counts["masked_tokens"] for k,v in incorrectToken_top_k_metrics.items()})
print({k:v/counts["masked_tokens"] for k,v in correctToken_top_k_metrics.items()})
