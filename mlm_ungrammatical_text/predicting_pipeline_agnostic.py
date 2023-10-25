from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.metrics import top_k_accuracy_score
import torch
import torch.nn.functional as nnf
import colorama
from colorama import Back, Fore, Style
import json
from datetime import datetime
from nltk.tag import pos_tag
import collections
import time



debugging_examples = [
        f''
    ]

def top_k(true_token_idx, top_k_predictions, max_k=10):
    #top_k_token_idxs = [idx for (_,idx, *_) in top_k_predictions ]
    true_token_rank = [rank_idx for rank_idx, (_,idx, *_) in enumerate(top_k_predictions) if idx == true_token_idx]
    true_token_rank = true_token_rank[0] if len(true_token_rank) > 0 else max_k
    top_k_flags = [ 1 if idx >= true_token_rank else 0 for idx in range(0, max_k) ]
    '''
    for k in range(1, max_k+1):  
        if true_token_idx in top_k_token_idxs[:k]:
            top_k_flags.append(1)
        else:
            top_k_flags.append(0)
    '''
    return top_k_flags, true_token_rank

def prediction_postag(tokenized_masked_sentence_, prediction_token_str, tokenizer):
    mask_token = tokenizer.mask_token
    mask_token_idx = tokenized_masked_sentence_.index(mask_token)  
    tokenized_masked_sentence_[mask_token_idx] = prediction_token_str 
    pos_tags = pos_tag(tokenized_masked_sentence_)
    prediction_pos = pos_tags[mask_token_idx][1]
    tokenized_masked_sentence_[mask_token_idx] = mask_token 
    return prediction_pos 

def predict(tokenizer, model, model_name, tokenized_masked_sentence, annotation_,k, printFlag=False):
    masked_sentence_str = " ".join(tokenized_masked_sentence)
    inputs = tokenizer(masked_sentence_str, return_tensors="pt")
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    logits = model(**inputs).logits
    probas = nnf.softmax(logits, dim=2)
    mask_token_logits = logits[0, mask_token_index, :]
    mask_token_probas = probas[0, mask_token_index, :]
    top_k_tokens = torch.topk(mask_token_probas, k, dim=1)
    top_k_tpls_per_masked_token = [list(zip(*tpl)) for tpl in zip(top_k_tokens.indices.tolist(), top_k_tokens.values.tolist())]
    top_k_tpls_per_masked_token = [[(tokenizer.decode(tpl[0]), *tpl, prediction_postag(tokenized_masked_sentence,tokenizer.decode(tpl[0]),tokenizer))
                                   for tpl in prediction_list] for prediction_list in top_k_tpls_per_masked_token] 

    #top_k_tpls_per_masked_token = [[("", *tpl, "")
    #                               for tpl in prediction_list] for prediction_list in top_k_tpls_per_masked_token] 
    annotation_["aligned_tokens"] = [(*tpl[0],(model_name,*tpl[1])) for tpl in zip(annotation_["aligned_tokens"],top_k_tpls_per_masked_token)]
    if printFlag:
        for predictions_tpls in top_k_tpls_per_masked_token:
            for prediction_tpl in predictions_tpls:
                (token_idx, token_prob) = prediction_tpl
                print(f"prob: {token_prob:.2f} ",masked_sentence_str.replace(tokenizer.mask_token, Back.GREEN + tokenizer.decode([token_idx]) +Style.RESET_ALL,1))
    return top_k_tpls_per_masked_token, annotation_ 

def load_words(fo):
    words = set()
    for word in fo:
        word = word.replace("\n","")
        words.add(word)
    return words


def main():
    models_name = {
            "xlm-ende": "xlm-mlm-ende-1024",
            "xlm-17": "xlm-mlm-17-1280",
            "xlm-100": "xlm-mlm-100-1280",
            "bert-wm"   : "bert-large-uncased-whole-word-masking",
            "bert-base" : "bert-base-uncased",
            "bert-multilingual": "bert-base-multilingual-cased",
            "bert-multilingual-uncased": "bert-base-multilingual-uncased",
            "xlm-roberta" : "xlm-roberta-large",
            "bart-large" : "bart-large",
            "br": "bert-base-uncased-finetuned-c4200m-nationality-br",
            "fr": "batch-24-bert-base-uncased-2023-09-01_13-59-45",
            "cn": "bert-base-uncased-finetuned-c4200m-nationality-cn",
            "mx": "bert-base-uncased-finetuned-c4200m-nationality-mx",
            "distilbert-base-uncased" : "distilbert-base-uncased",
            "full": "batch-414-bert-base-uncased-2023-08-24_11-35-00",
            "mbert-full": "batch-345-bert-base-multilingual-cased-2023-09-07_12-06-45",
            }

    #model_name = models_name["bert-base"]
    for model_name in ["bert-base-uncased"]: 
    #["batch-414-bert-base-uncased-2023-08-24_11-35-00","batch-345-bert-base-multilingual-cased-2023-09-07_12-06-45"] + ["bert-large-uncased-whole-word-masking","distilbert-base-uncased"]:
    #["bart-large","xlm-roberta-large","bert-base-uncased","bert-base-multilingual-cased"]:
    #models_name.values():
        model_folder = f"/app/pipelines/models/{model_name}" 
        tokenizer_folder = model_folder
        # tokenizer_folder = f"/app/pipelines/models/bert-base-uncased" 
        time_ = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_metrics_filepath = f"/app/pipelines/metrics/{model_name}_run_{time_}.json"
        instances_count_filepath = f"/app/pipelines/metrics/fce_instances_count_run_{time_}.json" 
        try:
            model = AutoModelForMaskedLM.from_pretrained(model_folder)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_folder)
        except:
            try:
                model = AutoModelForMaskedLM.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            except:
                model = AutoModelForMaskedLM.from_pretrained(model_folder)
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


        dataset = "fce"
        if dataset == "fce": 
            columns_written = False
            #print(dir(tokenizer))
            #print(len(tokenizer.vocab))
            #print(tokenizer.unk_token_id)
            max_k = 10 #len(tokenizer.vocab) # 30522
            instances_count = collections.defaultdict(int)
            global_top_k = [0 for _ in range(max_k)]
            languages_top_k = collections.defaultdict(lambda: [0 for _  in range(max_k)]) 
            #"/app/pipelines/data/fce_dataset/fce_error_annotations.json"
            input_filepath = "/app/pipelines/preprocessing_fce/fce_identical_error_annotations.json"
            fce_predictions_dataset_filepath = f"/app/pipelines/preprocessing_fce/fce_predictions_dataset_{model_name}_run_{time_}.tsv"
            with open(input_filepath) as inpf, open(fce_predictions_dataset_filepath,"w") as fce_dataset_outf:
                fce_sentences_dict = json.loads(inpf.read())
                for sentence_dict in fce_sentences_dict.values():
                    '''if instances_count["annotations"] > 50:
                        break
                    '''
                    sent = sentence_dict["deannotated_sentence"]
                    learnerl1 = sentence_dict["learnerl1"] 
                    for annotation in sentence_dict["annotations"]:
                        if annotation["match_type"] != "replacement_correction": #masked_token == "":
                            continue
                        if annotation["number_of_tokens"]  >= 2:
                            continue
                        start_time = time.time()
                        s, e  = annotation["span_in_DeannotatedSentence"]
                        tokens_idx_to_be_masked = [token[2] for token in annotation["aligned_tokens"]]
                        span_masked_token = sent[s:e]
                        masked_tokens = [t[0] for t in annotation["aligned_tokens"]]
                        masked_tokenized_sentence = [ token_tpl[0] 
                                                    if token_tpl[2] not in tokens_idx_to_be_masked 
                                                    else tokenizer.mask_token
                                                     for token_tpl in sentence_dict["tokenized_deannotated_sentence"] ] 
                        #print(masked_tokens)
                        #print(masked_tokenized_sentence)
                        #print(tokenizer.tokenize(masked_tokens[0]))
                        masked_tokens_vocab_idxs = [tokenizer.encode(masked_token)[1] for masked_token in masked_tokens]
                        #print(masked_tokens_vocab_idxs)
                        #print(tokenizer.decode(masked_tokens_vocab_idxs))
                        corrected_token = annotation["correct_token"]
                        span_masked_sentence = sent[:s] +\
                                        " ".join([tokenizer.mask_token
                                                 for _ in range(annotation["number_of_tokens"])])+\
                                        sent[e:]
                        top_k_tpls_per_masked_token, annotation = predict(tokenizer, model, model_name, masked_tokenized_sentence, annotation, k=max_k)
                        annotation_csv_line = f""
                        annotation_csv_columns = f""
                        # l1 
                        annotation_csv_columns += f"learnerl1\t"
                        annotation_csv_line  += f"{sentence_dict['learnerl1']}\t"
                        # raw_score None sentence_dict["learner_CEFR"]
                        annotation_csv_columns += f"learnerRawScore\t"
                        annotation_csv_line  += f"{sentence_dict['learnerscore']}\t"
                        # CEFR None sentence_dict["learner_CEFR"]
                        annotation_csv_columns += f"learnerCEFRproxy\t"
                        annotation_csv_line  += f"{None}\t"
                        #  annotation error type
                        annotation_csv_columns += f"annotationErrorType\t"
                        annotation_csv_line  += f"{annotation['error_type_symbol']}\t"
                        # incorrect_token 
                        annotation_csv_columns += f"masked_sentence\t"
                        annotation_csv_line  += f"{' '.join(masked_tokenized_sentence)}\t"
                        # incorrect_token 
                        annotation_csv_columns += f"incorrect_token\t"
                        annotation_csv_line  += f"{annotation['aligned_tokens'][0][0]}\t"
                        # incorrect_tokenIdx annotation["aligned_tokens"][0][2]
                        annotation_csv_columns += f"incorrect_token_idx\t"
                        annotation_csv_line  += f"{annotation['aligned_tokens'][0][2]}\t"
                        # incorrect_token_length annotation["aligned_tokens"][0][4]
                        annotation_csv_columns += f"incorrect_token_length\t"
                        annotation_csv_line  += f"{annotation['aligned_tokens'][0][4]}\t"
                        # incorrect_token_surprisal  None
                        annotation_csv_columns += f"incorrect_token_surprisal\t"
                        annotation_csv_line  += f"{None}\t"
                        # incorrect_token_position None
                        annotation_csv_columns += f"incorrect_token_position\t"
                        annotation_csv_line  += f"{None}\t"
                        # incorrect_token_pos annotation["aligned_tokens"][0][3]
                        annotation_csv_columns += f"incorrect_token_pos\t"
                        annotation_csv_line  += f"{annotation['aligned_tokens'][0][3]}\t"
                        # model_name annotation["aligned_tokens"][0][5][0])
                        annotation_csv_columns += f"model_name\t"
                        annotation_csv_line  += f"{annotation['aligned_tokens'][0][5][0]}\t"
                        top_k_predictions_to_report = 10
                        for i in range(1, top_k_predictions_to_report+1):
                            annotation_csv_columns += f"top_{i}_prediction\ttop_{i}_probability\ttop_{i}_pos\t"
                            annotation_csv_line  += f"{annotation['aligned_tokens'][0][5][i][0]}\t"
                            annotation_csv_line  += f"{annotation['aligned_tokens'][0][5][i][2]}\t"
                            annotation_csv_line  += f"{annotation['aligned_tokens'][0][5][i][3]}\t"

                        for (top_k_tpls, masked_token_idx) in zip(top_k_tpls_per_masked_token, masked_tokens_vocab_idxs):
                            instances_count["tokens"] +=1
                            instances_count[f"tokens_{learnerl1}"] +=1

                            instance_top_k_flags, true_token_rank = top_k(masked_token_idx,top_k_tpls,max_k=max_k)
                            top_k_metrics_to_report = 10 
                            annotation_csv_columns += f"true_token_rank\t"
                            annotation_csv_line  += f"{true_token_rank}\t"
                            for k_idx in range(0,top_k_metrics_to_report): 
                                k = k_idx + 1
                                annotation_csv_columns += f"top_{k}_metric\t"
                                annotation_csv_line  += f"{instance_top_k_flags[k_idx]}\t"
                            global_top_k = [a+b for (a,b) in zip(global_top_k, instance_top_k_flags)]
                            languages_top_k[learnerl1] = [a+b for (a,b) in zip(languages_top_k[learnerl1] , instance_top_k_flags)]
                        instances_count["annotations"] +=1
                        instances_count[f"annotations_{learnerl1}"] +=1
                        if not columns_written:
                            fce_dataset_outf.write(annotation_csv_columns+"\n")
                            columns_written = True
                        fce_dataset_outf.write(annotation_csv_line+"\n")
                        print(time.time()-start_time)
                        print(instances_count["annotations"])
                    instances_count["sentences"] +=1
                    instances_count[f"sentences_{learnerl1}"] +=1
                top_k_acc = [v/instances_count["annotations"] for v in global_top_k]
                languages_top_k_acc = {
                        language: [v/instances_count[f"annotations_{language}"] for v in language_top_k_lst]
                            for language, language_top_k_lst in languages_top_k.items()
                        }

                with open(model_metrics_filepath,"w") as metrics_outf:
                    top_ks = {
                    }
                    for lastIdx in [max_k]:
                        top_ks[lastIdx] = top_k_acc[:lastIdx]
                        for language in languages_top_k_acc.keys():
                            top_ks[f"{language}_{lastIdx}"] = languages_top_k_acc[language][:lastIdx]
                    metrics_outf.write(json.dumps(top_ks, indent=4))
                '''
                with open(instances_count_filepath,"w") as instances_count_outf:
                    instances_count_outf.write(json.dumps(instances_count, indent=4))
                '''
    else:
            pass

if __name__ == "__main__":
    main()
