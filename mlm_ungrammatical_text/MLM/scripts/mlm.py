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


# In[2]:
def clean_input(text):
    return clean(text,
                fix_unicode=True,               # fix various unicode errors
                    to_ascii=True,                  # transliterate to closest ASCII representation
                    lower=True,                     # lowercase text
                    no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
                    no_urls=False,                  # replace all URLs with a special token
                    no_emails=False,                # replace all email addresses with a special token
                    no_phone_numbers=False,         # replace all phone numbers with a special token
                    no_numbers=False,               # replace all numbers with a special token
                    no_digits=False,                # replace all digits with a special token
                    no_currency_symbols=False,      # replace all currency symbols with a special token
                    no_punct=False,                 # remove punctuations
                    replace_with_punct="",          # instead of removing punctuations you may replace them
                    replace_with_url="<URL>",
                    replace_with_email="<EMAIL>",
                    replace_with_phone_number="<PHONE>",
                    replace_with_number="<NUMBER>",
                    replace_with_digit="0",
                    replace_with_currency_symbol="<CUR>",
                    lang="en"                       # set to 'de' for German special handling
          )

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
    wwm_probability = 0.2
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


	# In[55]:

	eval_results = trainer.evaluate()
	print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

	trainer.train()
	eval_results = trainer.evaluate()
	print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
	trainer.save_model()

# In[2]:


"""
# straight steps to add new token to wordembbeging and tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
tokenizer.add_tokens(['[newtoken]'])
model = BertModel.from_pretrained('bert-base-uncased')
model.embeddings
weights = model.embeddings.word_embeddings.weight.data
new_weights = torch.cat((weights, weights[101:102]), 0)
new_emb = nn.Embedding.from_pretrained(new_weights, padding_idx=0, freeze=False)
model.embeddings.word_embeddings = new_emb
# out = model(**tokenized)
# out.last_hidden_state
"""

# In[ ]:


'''
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
tokenizer.tokenize("[CLS] Hello world, how are you?")
tokenizer.tokenize("[newtoken] Hello world, how are you?")
tokenizer.add_tokens(['[newtoken]'])
tokenizer.tokenize("[newtoken] Hello world, how are you?")
tokenized = tokenizer("[newtoken] Hello world, how are you?", add_special_tokens=False, return_tensors="pt")
print(tokenized['input_ids'])
tkn = tokenized['input_ids'][0, 0]
print("First token:", tkn)
print("Decoded:", tokenizer.decode(tkn))
'''


# ## Adding a new token to bert


# In[ ]:


'''
model = BertModel.from_pretrained('bert-base-uncased')
model.embeddings
try:
    out = model(**tokenized)
    out.last_hidden_state
except Exception as e:
    print(e)
weights = model.embeddings.word_embeddings.weight.data
print(weights.shape)
new_weights = torch.cat((weights, weights[101:102]), 0)
new_emb = nn.Embedding.from_pretrained(new_weights, padding_idx=0, freeze=False)
new_emb
model.embeddings.word_embeddings = new_emb
model.embeddings
out = model(**tokenized)
out.last_hidden_state
model = BertModel.from_pretrained('bert-base-uncased')
out2 = model(
    **tokenizer("[CLS] Hello world, how are you?", add_special_tokens=False, return_tensors="pt")
)
torch.all(out.last_hidden_state == out2.last_hidden_state)
'''


# ## Picking a pretrained model for masked language modeling<br>
# pick a model here https://huggingface.co/models?pipeline_tag=fill-mask&sort=downloads with fill-mask filter<br>
# 

# In[17]:

# In[4]:


# In[5]:


'''
text = "This is a great [MASK]."
inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits
# Find the location of [MASK] and extract its logits
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]
# Pick the [MASK] candidates with the highest logits
top_5_tokens = torch.topk(mask_token_logits, 10, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")
'''


# In[16]:
model = AutoModelForMaskedLM.from_pretrained("./datasets/bert-base-uncased-c4200m-unchaged-vocab-1111102022")
# ./bert-base-uncased-c4200m-unchaged-vocab
# bert-base-uncased-c4200m-unchaged-vocab-1111102022
# model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')




# inputs = tokenizer(texts, return_tensors="pt",
#                         max_length=512, truncation=True, padding="max_length")
# 
# inputs['labels'] = inputs.input_ids.detach().clone()
# 
# random_val_per_token = torch.rand(inputs.input_ids.shape)
# random_selected_mask = random_val_per_token < 0.15
# 
# not_special_tokens = (inputs.input_ids != 101) *\
#                         (inputs.input_ids != 102) *\
#                             (inputs.input_ids != 0)
# 
# masked_tokens = random_selected_mask * not_special_tokens
# masked_sentences = torch.where(masked_tokens > 0, 103, inputs.input_ids)
# inputs.input_ids = masked_sentences
# 
# class C4SentencesDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings):
#         self.encodings = encodings
# 
#     def __getitem__(self, idx):
#         return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
# 
#     def __len__(self):
#         return len(self.encodings.input_ids)
# c4_dataset = C4SentencesDataset(inputs)
# c4_dataloader = torch.utils.data.DataLoader(c4_dataset, batch_size=32, shuffle=True)


# In[31]:


'''
concatenated_examples = {
    k: sum(tokenized_samples[k], []) for k in tokenized_samples.keys()
}
total_length = len(concatenated_examples["input_ids"])
print(f"'>>> Concatenated reviews length: {total_length}'")
chunks = {
    k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
    for k, t in concatenated_examples.items()
}


for chunk in chunks["input_ids"]:
    print(f"'>>> Chunk length: {len(chunk)}'")
'''



# In[31]:

#samples = [lm_datasets[i] for i in range(10)]
'''
for sample in samples:
    _ = sample.pop("word_ids") 
'''

# for chunk in data_collator(samples)["input_ids"]:
#   print(f"\n'>>> {tokenizer.decode(chunk)}'")

# 61_010_000
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
		# clean_input(
print(f"texts were loadead")
