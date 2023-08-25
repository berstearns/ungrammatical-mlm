"""
This file runs Masked Language Model. You provide a training file. Each line is interpreted as a sentence / paragraph.
Optionally, you can also provide a dev file.

The fine-tuned model is stored in the output/model_name folder.

Usage:
python train_mlm.py model_name data/train_sentences.txt [data/dev_sentences.txt]
"""

from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers import Trainer, TrainingArguments
import sys
import gzip
from datetime import datetime
import os
from dataloader import TokenizedSentencesDataset


def decide_trainedModelsDir():
    lastEpochDir_files = os.listdir("/content/drive/MyDrive/phd/code/data/run_20230821/output_1epoch/")
    currEpochDir_files = os.listdir("/content/drive/MyDrive/phd/code/data/run_20230821/output_2epoch/") 
    if len(currEpochDir_files) == 0:
        return lastEpochDir_files  
    else:
        return currEpochDir_files

epoch_num = 2
per_device_train_batch_size = 64
trained_models_dir = decide_trainedModelsDir 
last_batch_idx, model_foldername = max([(int(folder.split("-")[1]),folder) for folder in trained_models_dir]) if len(trained_models_dir) > 0 else (0, None)
curr_batch_idx = last_batch_idx + 1
print(f"last_batch_idx : {last_batch_idx}")
print(f"curr_batch_idx : {curr_batch_idx}")
model_name = "bert-base-uncased"
log_folder  = f"/content/drive/MyDrive/phd/code/data/run_20230821/logs/"
checkpointModel_folder = f"{trained_models_dir}/{model_foldername}" if model_foldername else None
batches_folder = "/content/drive/MyDrive/phd/code/data/run_20230821/batches_gt5_20230822/"
train_filepath = os.path.join(batches_folder, f"batch_{curr_batch_idx}.txt") 
save_steps = 1000               #Save model every 1k steps
num_train_epochs = 1            #Number of epochs
use_fp16 = False                #Set to True, if your GPU supports FP16 operations
max_length = 100                #Max length for a text input
do_whole_word_mask = True       #If set to true, whole words are masked
mlm_prob = 0.15                 #Probability that a word is replaced by a [MASK] token
output_dir = "/content/drive/MyDrive/phd/code/data/run_20230821/output_{}epoch/batch-{}-{}-{}".format(epoch_num, curr_batch_idx, model_name,  datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
print("Save checkpoints to:", output_dir)

# Load the model
model = AutoModelForMaskedLM.from_pretrained(checkpointModel_folder if checkpointModel_folder != None else "bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained(checkpointModel_folder if checkpointModel_folder != None else "bert-base-uncased")




##### Load our training datasets

train_sentences = []
with gzip.open(train_filepath, 'rt', encoding='utf8') if train_filepath.endswith('.gz') else  open(train_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        line = line.strip()
        if len(line) >= 10:
            train_sentences.append(line)

print("Train sentences:", len(train_sentences))

dev_sentences = []
'''
if len(sys.argv) >= 4:
    dev_path = sys.argv[3]
    with gzip.open(dev_path, 'rt', encoding='utf8') if dev_path.endswith('.gz') else open(dev_path, 'r', encoding='utf8') as fIn:
        for line in fIn:
            line = line.strip()
            if len(line) >= 10:
                dev_sentences.append(line)

print("Dev sentences:", len(dev_sentences))
'''


train_dataset = TokenizedSentencesDataset(train_sentences, tokenizer, max_length)
#dev_dataset = TokenizedSentencesDataset(dev_sentences, tokenizer, max_length, cache_tokenization=True) if len(dev_sentences) > 0 else None
dev_dataset = None


##### Training arguments

if do_whole_word_mask:
    data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)
else:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,
    evaluation_strategy="steps" if dev_dataset is not None else "no",
    per_device_train_batch_size=per_device_train_batch_size,
    eval_steps=save_steps,
    save_steps=save_steps,
    logging_steps=save_steps,
    save_total_limit=1,
    prediction_loss_only=True,
    fp16=use_fp16
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset
)

print("Save tokenizer to:", output_dir)
tokenizer.save_pretrained(output_dir)

trainer.train()

print("Save model to:", output_dir)
model.save_pretrained(output_dir)

print("Training done")
