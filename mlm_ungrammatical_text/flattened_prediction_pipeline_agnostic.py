'''
    extract linguistic metrics for a given
    language model for a given set of Masked Senteces
    todos:
        - test loading configuration from an existing config file is sucessful
        - test loading configuration from an unexisting config file fails 
        - test output file has expected columns
            - for FCE corrected tokens
            - for FCE incorrect tokens
            - for EFCAMDAT dataset
            - for error annotated EFCAMDAT
            - for NUCLE
            - for general masked sentences
        - create config file for the fce for the bert native model
        - 
'''
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
import sys

debugging_examples = [
        f''
    ]

def str_based_top_k(true_token_str, top_k_predictions, MAX_K):
    true_token_str = "shocked"
    #top_k_token_idxs = [idx for (_,idx, *_) in top_k_predictions ]
    true_token_rank = [rank_idx+1 for rank_idx, (token_str, idx, *_) in enumerate(top_k_predictions) if token_str == true_token_str]
    true_token_rank = true_token_rank[0] if len(true_token_rank) > 0 else MAX_K
    top_k_flags = [ 1 if idx >= true_token_rank else 0 for idx in range(0, MAX_K) ]
    '''
    for k in range(1, MAX_K+1):  
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

def predict(tokenizer, model, MODEL_NAME, masked_tokenized_sentence_str_batch, k, printFlag=False):
    '''
        returns the top k predictions for a given masked sentence
    '''
    start_time = time.time()
    inputs = tokenizer.batch_encode_plus(masked_tokenized_sentence_str_batch, return_tensors="pt",padding=True)
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    logits = model(**inputs).logits
    probas = nnf.softmax(logits, dim=2)
    mask_token_logits = logits[0, mask_token_index, :]
    mask_token_probas = probas[0, mask_token_index, :]
    top_k_tokens = torch.topk(mask_token_probas, k, dim=1)
    top_k_tpls_per_masked_token = [list(zip(*tpl)) for tpl in zip(top_k_tokens.indices.tolist(), top_k_tokens.values.tolist())]
    top_k_tpls_per_masked_token = [[(tokenizer.decode(tpl[0]), *tpl, '')
                                   for tpl in prediction_list] for prediction_list in top_k_tpls_per_masked_token]
    if printFlag:
        for predictions_tpls in top_k_tpls_per_masked_token:
            for prediction_tpl in predictions_tpls:
                (token_idx, token_prob) = prediction_tpl
                print(f"prob: {token_prob:.2f} ",masked_sentence_str.replace(tokenizer.mask_token, Back.GREEN + tokenizer.decode([token_idx]) +Style.RESET_ALL,1))
    return top_k_tpls_per_masked_token 

def load_words(fo):
    words = set()
    for word in fo:
        word = word.replace("\n","")
        words.add(word)
    return words


def generate_tsv_line_from_masked_sentence_dict(masked_sentence_dict):
    annotation_csv_line = f""
    annotation_csv_columns = f""
    # l1 
    annotation_csv_columns += f"learnerl1\t"
    annotation_csv_line  += f"{learnerl1_}\t"
    # raw_score None sentence_dict["learner_CEFR"]
    annotation_csv_columns += f"learnerRawScore\t"
    annotation_csv_line  += f"{sentence_dict_['learnerscore']}\t"
    # CEFR None sentence_dict["learner_CEFR"]
    annotation_csv_columns += f"learnerCEFRproxy\t"
    annotation_csv_line  += f"{None}\t"
    #  annotation error type
    annotation_csv_columns += f"annotationErrorType\t"
    annotation_csv_line  += f"{annotation_['error_type_symbol']}\t"
    # masked_sentence 
    annotation_csv_columns += f"masked_sentence\t"
    annotation_csv_line  += f"{' '.join(masked_tokenized_sentence)}\t"
    # incorrect_token 
    annotation_csv_columns += f"incorrect_token\t"
    annotation_csv_line  += f"{token_data[0]}\t"
    # incorrect_tokenIdx annotation["aligned_incorrect_tokens"][0][2]
    annotation_csv_columns += f"incorrect_token_idx\t"
    annotation_csv_line  += f"{token_data[2]}\t"
    # incorrect_token_length annotation["aligned_incorrect_tokens"][0][4]
    annotation_csv_columns += f"incorrect_token_length\t"
    annotation_csv_line  += f"{token_data[4]}\t"
    # incorrect_token_surprisal  None
    annotation_csv_columns += f"incorrect_token_surprisal\t"
    annotation_csv_line  += f"{None}\t"
    # incorrect_token_position None
    annotation_csv_columns += f"incorrect_token_position\t"
    annotation_csv_line  += f"{None}\t"
    # incorrect_token_pos annotation["aligned_incorrect_tokens"][0][3]
    annotation_csv_columns += f"incorrect_token_pos\t"
    annotation_csv_line  += f"{token_data[3]}\t"
    # MODEL_NAME annotation["aligned_incorrect_tokens"][0][5][0])
    annotation_csv_columns += f"MODEL_NAME\t"
    annotation_csv_line  += f"{annotation_['aligned_incorrect_tokens'][0][5][0]}\t"
    top_k_predictions_to_report = 10
    for i in range(1, top_k_predictions_to_report+1):
        annotation_csv_columns += f"top_{i}_prediction\ttop_{i}_probability\ttop_{i}_pos\t"
        annotation_csv_line  += f"{annotation_['aligned_incorrect_tokens'][aligned_token_idx][5][i][0]}\t"
        annotation_csv_line  += f"{annotation_['aligned_incorrect_tokens'][aligned_token_idx][5][i][2]}\t"
        annotation_csv_line  += f"{annotation_['aligned_incorrect_tokens'][aligned_token_idx][5][i][3]}\t"

def process_annotation_incorrect_token(instances_count_, global_top_k_, languages_top_k_, columns_written_,  tokenizer_, masked_sentences_batch_, model_, MODEL_NAME, MAX_K, FCE_DATASET_OUTF):
    """
        process an annotation of an incorrect token
        params:
            instances_count_: dict
            global_top_k_: list
            languages_top_k_: dict
            columns_written_: bool
            tokenizer_: tokenizer
            masked_sentence_dict_: dict
            learnerl1_: str
            model_: model
            MODEL_NAME: str
            MAX_K: int
            FCE_DATASET_OUTF: file
        returns:
            instances_count_: dict
            global_top_k_: list
            languages_top_k_: dict
            columns_written_: bool
        examples:
            >>> process_annotation_incorrect_token(instances_count_, global_top_k_, languages_top_k_, columns_written_,  annotation_, tokenizer_, masked_sentence_dict_, learnerl1_, model_, MODEL_NAME, MAX_K, FCE_DATASET_OUTF)
            (instances_count_, global_top_k_, languages_top_k_, columns_written_)
    """
    tokens_idx_to_be_masked = [masked_sentence_dict_["incorrect_token_idx"] for masked_sentence_dict_ in masked_sentences_batch_]
    masked_tokens = [masked_sentence_dict_["incorrect_token"] for masked_sentence_dict_ in masked_sentences_batch_]
    masked_tokenized_sentences_str_batch =  [masked_sentence_dict_["masked_sentence"].replace("[MASK]", tokenizer_.mask_token) for masked_sentence_dict_ in masked_sentences_batch_]   
    start_time = time.time()
    top_k_tpls_per_masked_token  = predict(tokenizer_, model_, MODEL_NAME, masked_tokenized_sentences_str_batch, k=MAX_K)

    for MS_idx, (masked_sentence_dict_, masked_token_top_k_tpl) in enumerate(zip(masked_sentences_batch_, top_k_tpls_per_masked_token)): 
        print(masked_token_top_k_tpl[:10])
        instance_top_k_flags, true_token_rank = str_based_top_k(masked_sentence_dict_["incorrect_token"], masked_token_top_k_tpl,MAX_K=MAX_K)
        masked_sentences_batch_[MS_idx]["true_token_rank"] = true_token_rank  
        for k_idx, top_k_prediction_tpl in enumerate(masked_token_top_k_tpl):
            masked_sentences_batch_[MS_idx][f"top_{k_idx+1}_token"] = top_k_prediction_tpl[0]
            masked_sentences_batch_[MS_idx][f"top_{k_idx+1}_probability"] = top_k_prediction_tpl[2] 

        for k_idx in range(0,top_k_metrics_to_report): 
            k = k_idx + 1
            annotation_csv_columns += f"top_{k}_metric\t"
            annotation_csv_line  += f"{instance_top_k_flags[k_idx]}\t"
            # generate_tsv_line_from_masked_sentence_dict(masked_sentence_dict_)
            global_top_k_ = [a+b for (a,b) in zip(global_top_k_, instance_top_k_flags)]
            languages_top_k_[learnerl1_] = [a+b for (a,b) in zip(languages_top_k_[learnerl1_] , instance_top_k_flags)]
            if not columns_written_:
                FCE_DATASET_OUTF.write(annotation_csv_columns+"\n")
                columns_written_ = True
            FCE_DATASET_OUTF.write(annotation_csv_line+"\n")
    print(time.time()-start_time)
    print(instances_count_)
    return instances_count_, global_top_k_, languages_top_k_, columns_written_


def predict_correct_token(tokenizer, model, MODEL_NAME, tokenized_masked_sentence, annotation_, mask_idx, k, printFlag=False):
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
    annotation_["tokenized_partially_corrected_sentence"][mask_idx].append((MODEL_NAME, *top_k_tpls_per_masked_token[0])) 
    if printFlag:
        for predictions_tpls in top_k_tpls_per_masked_token:
            for prediction_tpl in predictions_tpls:
                (token_idx, token_prob) = prediction_tpl
                print(f"prob: {token_prob:.2f} ",masked_sentence_str.replace(tokenizer.mask_token, Back.GREEN + tokenizer.decode([token_idx]) +Style.RESET_ALL,1))
    return top_k_tpls_per_masked_token, annotation_ 

def process_annotation_correct_token(instances_count_, global_top_k_, languages_top_k_, columns_written_,  annotation_, tokenizer_, sentence_dict_, learnerl1_, model_, MODEL_NAME, MAX_K, FCE_DATASET_OUTF):
    mapping_type = f"mapping_{len(annotation_['aligned_incorrect_tokens'])}-{len(annotation_['correct_token'].split(' '))}"
    instances_count_[f"replacement_annotations_n_tokens{annotation_['number_of_incorrect_tokens']}"] +=1
    instances_count_[mapping_type]+=1
    instances_count_["annotations"] +=1
    instances_count_[f"annotations_{learnerl1_}"] +=1
    start_time = time.time()
    corrected_token_idx = annotation_['correction_idxs_in_partially_corrected_sentence'][0]
    tokens_idx_to_be_masked = [corrected_token_idx]
    masked_tokens = [annotation_['tokenized_partially_corrected_sentence'][corrected_token_idx][0]]
    token_data = annotation_['tokenized_partially_corrected_sentence'][corrected_token_idx] 
    masked_tokenized_sentence = [ token_tpl[0] 
                                if token_tpl[2] not in tokens_idx_to_be_masked 
                                else tokenizer_.mask_token
                                 for token_tpl in sentence_dict_["tokenized_deannotated_sentence"] ] 
    #print(masked_tokens)
    #print(masked_tokenized_sentence)
    #print(tokenizer_.tokenize(masked_tokens[0]))
    masked_tokens_vocab_idxs = [tokenizer_.encode(masked_token)[1] for masked_token in masked_tokens]
    corrected_token = annotation_["correct_token"]
    top_k_tpls_per_masked_token, annotation_ = predict_correct_token(tokenizer_, model_, MODEL_NAME, masked_tokenized_sentence, annotation_, corrected_token_idx, k=MAX_K)
    annotation_csv_line = f""
    annotation_csv_columns = f""
    # mapping type
    annotation_csv_columns += f"mapping_type\t"
    annotation_csv_line  += f"{mapping_type}\t"
    # l1 
    annotation_csv_columns += f"learnerl1\t"
    annotation_csv_line  += f"{learnerl1_}\t"
    # raw_score None sentence_dict["learner_CEFR"]
    annotation_csv_columns += f"learnerRawScore\t"
    annotation_csv_line  += f"{sentence_dict_['learnerscore']}\t"
    # CEFR None sentence_dict["learner_CEFR"]
    annotation_csv_columns += f"learnerCEFRproxy\t"
    annotation_csv_line  += f"{None}\t"
    #  annotation error type
    annotation_csv_columns += f"annotationErrorType\t"
    annotation_csv_line  += f"{annotation_['error_type_symbol']}\t"
    # masked_sentence 
    annotation_csv_columns += f"masked_sentence\t"
    annotation_csv_line  += f"{' '.join(masked_tokenized_sentence)}\t"
    # correct_token 
    annotation_csv_columns += f"correct_token\t"
    annotation_csv_line  += f"{token_data[0]}\t"
    # correct_tokenIdx annotation["aligned_incorrect_tokens"][0][2]
    annotation_csv_columns += f"correct_token_idx\t"
    annotation_csv_line  += f"{token_data[2]}\t"
    # correct_token_length annotation["aligned_incorrect_tokens"][0][4]
    annotation_csv_columns += f"correct_token_length\t"
    annotation_csv_line  += f"{token_data[4]}\t"
    # correct_token_surprisal  None
    annotation_csv_columns += f"correct_token_surprisal\t"
    annotation_csv_line  += f"{None}\t"
    # correct_token_position None
    annotation_csv_columns += f"correct_token_position\t"
    annotation_csv_line  += f"{None}\t"
    # correct_token_pos annotation["aligned_incorrect_tokens"][0][3]
    annotation_csv_columns += f"correct_token_pos\t"
    annotation_csv_line  += f"{token_data[3]}\t"
    # MODEL_NAME annotation["aligned_incorrect_tokens"][0][5][0])
    annotation_csv_columns += f"model_name\t"
    annotation_csv_line  += f"{annotation_['tokenized_partially_corrected_sentence'][corrected_token_idx][5][0]}\t"
    top_k_predictions_to_report = 10
    for i in range(1, top_k_predictions_to_report+1):
        annotation_csv_columns += f"top_{i}_prediction\ttop_{i}_probability\ttop_{i}_pos\t"
        annotation_csv_line  += f"{annotation_['tokenized_partially_corrected_sentence'][corrected_token_idx][5][i][0]}\t"
        annotation_csv_line  += f"{annotation_['tokenized_partially_corrected_sentence'][corrected_token_idx][5][i][2]}\t"
        annotation_csv_line  += f"{annotation_['tokenized_partially_corrected_sentence'][corrected_token_idx][5][i][3]}\t"

    for (top_k_tpls, masked_token_idx) in zip(top_k_tpls_per_masked_token, masked_tokens_vocab_idxs):
        instances_count_["tokens"] +=1
        instances_count_[f"tokens_{learnerl1_}"] +=1

        instance_top_k_flags, true_token_rank = top_k(masked_token_idx,top_k_tpls,MAX_K=MAX_K)
        top_k_metrics_to_report = 10 
        annotation_csv_columns += f"true_token_rank\t"
        annotation_csv_line  += f"{true_token_rank}\t"
        for k_idx in range(0,top_k_metrics_to_report): 
            k = k_idx + 1
            annotation_csv_columns += f"top_{k}_metric\t"
            annotation_csv_line  += f"{instance_top_k_flags[k_idx]}\t"
        global_top_k_ = [a+b for (a,b) in zip(global_top_k_, instance_top_k_flags)]
        languages_top_k_[learnerl1_] = [a+b for (a,b) in zip(languages_top_k_[learnerl1_] , instance_top_k_flags)]
    if not columns_written_:
        FCE_DATASET_OUTF.write(annotation_csv_columns+"\n")
        columns_written_ = True
    FCE_DATASET_OUTF.write(annotation_csv_line+"\n")
    print(time.time()-start_time)
    print(instances_count_)
    return instances_count_, global_top_k_, languages_top_k_, columns_written_

def load_fce(file_obj,
             input_type):
        '''
            expects a .tsv input type,
            return a list of dicts FCE_ROWs
        '''
        rows = []
        columns = next(file_obj).rstrip().split("\t") 
        for row in file_obj:
            row = row.rstrip().split("\t")
            row_data = {k:v for k,v in zip(columns, row)}
            rows.append(row_data) 
        return rows  
        # json.loads(inpf.read())
def main(_INPUT_FILEPATH,
         _INPUT_TYPE,
         _DATASET_NAME,
         _MODEL_NAME, 
         _MODEL_FOLDER,
         _TOKENIZER_FOLDER,
         _MODEL_METRICS_FILEPATH,
         _INSTANCES_COUNT_FILEPATH,
         _FCE_PREDICTIONS_DATASET_FILEPATH,
         _PREDICTION_TARGET,
         _MAX_K,
         _BATCH_SIZE
        ):

    ##### model loading strategy
    try:
        model = AutoModelForMaskedLM.from_pretrained(_MODEL_FOLDER)
        tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_FOLDER)
    except:
        try:
            model = AutoModelForMaskedLM.from_pretrained(_MODEL_NAME)
            tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        except:
            model = AutoModelForMaskedLM.from_pretrained(_MODEL_FOLDER)
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if _MAX_K == "vocabulary":
        _MAX_K = len(tokenizer.vocab)

    if _DATASET_NAME.upper()  == "FCE": 
        columns_written = False
        instances_count = collections.defaultdict(int)
        global_top_k = [0 for _ in range(_MAX_K)]
        languages_top_k = collections.defaultdict(lambda: [0 for _  in range(_MAX_K)]) 
        with open(_INPUT_FILEPATH) as inpf, open(_FCE_PREDICTIONS_DATASET_FILEPATH,"w") as FCE_DATASET_OUTF:
            fce_sentences_dict = load_fce(inpf, _INPUT_TYPE)
            for batch_start_idx in range(0, len(fce_sentences_dict), _BATCH_SIZE):
                batch_end_idx =  batch_start_idx+_BATCH_SIZE
                masked_sentences_batch = fce_sentences_dict[batch_start_idx:batch_end_idx]
                if _PREDICTION_TARGET == PredictionTargets["i"]:
                    instances_count, global_top_k, languages_top_k, columns_written =\
                            process_annotation_incorrect_token(instances_count, global_top_k, languages_top_k, columns_written, tokenizer, masked_sentences_batch, model, _MODEL_NAME, _MAX_K, FCE_DATASET_OUTF)
                elif _PREDICTION_TARGET == PredictionTargets["c"]:
                    instances_count, global_top_k, languages_top_k, columns_written =\
                            process_annotation_correct_token(instances_count, global_top_k, languages_top_k, columns_written, annotation, tokenizer, sentence_dict, learnerl1, model, _MODEL_NAME, _MAX_K, FCE_DATASET_OUTF)
                else: 
                    raise Exception("invalid option")
            top_k_acc = [v/instances_count["tokens"] for v in global_top_k]
            languages_top_k_acc = {
                    language: [v/instances_count[f"tokens_{language}"] for v in language_top_k_lst]
                        for language, language_top_k_lst in languages_top_k.items()
                    }
            maps_counts = sorted([tpl for tpl in instances_count.items() if 'mapping_' in tpl[0]],key=lambda x:x[1])
            total = sum([v for k,v in maps_counts])
            print([(k, v/total) for k,v in maps_counts]) 

            with open(model_metrics_filepath,"w") as metrics_outf:
                top_ks = {
                }
                for lastIdx in [_MAX_K]:
                    top_ks[lastIdx] = top_k_acc[:lastIdx]
                    for language in languages_top_k_acc.keys():
                        top_ks[f"{language}_{lastIdx}"] = languages_top_k_acc[language][:lastIdx]
                metrics_outf.write(json.dumps(top_ks, indent=4))
            with open(instances_count_filepath,"w") as instances_count_outf:
                instances_count_outf.write(json.dumps(instances_count, indent=4))
    else:
            pass

if __name__ == "__main__":
    #### those are not being used
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

    models_to_evaluate = [
            #"bert-base-uncased",
            #"batch-414-bert-base-uncased-fine-tuned",
            #"xlm-roberta-large",
            #"bert-base-multilingual-uncased",
            #"bert-base-uncased-finetuned",
            #"bert-large-uncased-whole-word-masking",
            ]
    #### those are not being used
    #################
    InputTypes = {
            "masked_sentences",
            "annotations_json"
            }
    PredictionTargets = {
            "c": "corrected_tokens",
            "i": "incorrect_tokens"
            }

    config_filepath = sys.argv[1]
    with open(config_filepath) as inpf:
        config = json.load(inpf)
        config = {k.upper(): v  for k,v in config.items()}
        locals().update(**config)
    MAX_K = MAX_K if MAX_K != "vocabulary" else "vocabulary"
    time_ = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    MODEL_FOLDER = f"/app/pipelines/models/{MODEL_NAME}" 
    TOKENIZER_FOLDER = MODEL_FOLDER
    PREDICTION_TARGET = PredictionTargets[PREDICTION_TARGET]
    MODEL_METRICS_FILEPATH = f"/app/pipelines/metrics/corrected_tokens_{MODEL_NAME}_run_{time_}.json" if PREDICTION_TARGET == PredictionTargets["c"]\
            else f"/app/pipelines/metrics/incorrect_tokens_{MODEL_NAME}_run_{time_}.json"  
    INSTANCES_COUNT_FILEPATH = f"/app/pipelines/metrics/fce_corrected_tokens_instances_count_run_{time_}.json" if PREDICTION_TARGET == PredictionTargets["c"]\
            else f"/app/pipelines/metrics/fce_incorrect_tokens_instances_count_run_{time_}.json"  
    FCE_PREDICTIONS_DATASET_FILEPATH = f"/app/pipelines/predictions/fce_corrected_tokens_predictions_{MODEL_NAME}_run_{time_}.tsv" if PREDICTION_TARGET == PredictionTargets["c"]\
            else f"/app/pipelines/predictions/fce_incorrect_tokens_predictions_{MODEL_NAME}_run_{time_}.tsv" 
    main(
            INPUT_FILEPATH,
            INPUT_TYPE,
            DATASET_NAME,
            MODEL_NAME,
            MODEL_FOLDER,
            TOKENIZER_FOLDER,
            MODEL_METRICS_FILEPATH,
            INSTANCES_COUNT_FILEPATH,
            FCE_PREDICTIONS_DATASET_FILEPATH,
            PREDICTION_TARGET,
            MAX_K,
            BATCH_SIZE
        )
