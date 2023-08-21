#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8


# In[37]:

# In[15]:

import math
from transformers import BertTokenizerFast, BertModel
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import Trainer
import collections
import numpy as np
from transformers import default_data_collator
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch import nn
from transformers import BertTokenizer, BertForMaskedLM
from transformers import AdamW
import torch
from tqdm import tqdm
from cleantext import clean
from datasets import Dataset

c4200m_filepath ="./datasets/writings_sentences_cn.tsv" #"./datasets/br_efcamdat_sentences.tsv"#"./datasets/c4200m.tsv"
model_checkpoint = "bert-base-uncased"

def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

def group_texts(examples):
    chunk_size = 128
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
            k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

def whole_word_masking_data_collator(features):
    wwm_probability = 0.15
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
    return default_data_collator(features)                                

def partial_dataset_train(texts):
	c4_huggingface_dataset = Dataset.from_list(texts)
	tokenized_datasets = c4_huggingface_dataset.map(
	    tokenize_function, batched=True, remove_columns=["text"]#, "label"]
	)

	device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
	data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
	lm_datasets = tokenized_datasets.map(group_texts, batched=True)
	train_size = int(len(lm_datasets)*0.85)
	test_size = len(lm_datasets) - train_size 
	print(f"train size: {train_size} , test size: {test_size}")

	downsampled_dataset = lm_datasets.train_test_split(
	    train_size=train_size, test_size=test_size, seed=42
	)
	batch_size = 32 
	logging_steps = len(downsampled_dataset["train"]) // batch_size

	training_args = TrainingArguments(
	    output_dir=f"{model_checkpoint}-finetuned-c4200m-nationality-cn",
	    overwrite_output_dir=True,
	    evaluation_strategy="epoch",
	    learning_rate=2e-5,
	    weight_decay=0.01,
	    per_device_train_batch_size=batch_size,
	    per_device_eval_batch_size=batch_size,
	    push_to_hub=False,
	    # fp16=True,
	    logging_steps=logging_steps,
	)

	trainer = Trainer(
	    model=model,
	    args=training_args,
	    train_dataset=downsampled_dataset["train"],
	    eval_dataset=downsampled_dataset["test"],
	    data_collator=data_collator,
	)




if __name__ == "__main__":
    model_folder = "./datasets/bert-base-uncased-c4200m-unchaged-vocab-1111102022"
    model = AutoModelForMaskedLM.from_pretrained(model_folder)
    tokenizer = AutoTokenizer.from_pretrained(model_folder)
    starting_line = 0
    iter_count = starting_line // 10000 
    count = 0
    line_count = 0
    texts=[]
    with open(c4200m_filepath, "r") as inpf:
        print("starting")
        for line in inpf:
            while line_count < starting_line:
                line_count+=1 
                continue
            count+=1
            if count > 10_000:
                iter_count += 1
                print(f"{'*'*200}")
                print(f"iter count : {iter_count} , {iter_count*10000} lines processed")
                print(f"{'*'*200}")
                partial_dataset_train(texts)
                texts=[]
                count=0
            texts.append({"text":line.split("\t")[0]})
    print(f"texts were loadead")
    '''
	# In[55]:

	eval_results = trainer.evaluate()
	print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

	trainer.train()
	eval_results = trainer.evaluate()
	print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
	trainer.save_model()
    '''
