import pickle as pickle
import os
import pandas as pd
import numpy as np
import random
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, SchedulerType
from load_data import *

import wandb

import transformers
from transformers import ElectraModel, ElectraForSequenceClassification, ElectraConfig, ElectraTokenizer
from tokenization_kocharelectra import KoCharElectraTokenizer

from transformers import get_cosine_with_hard_restarts_schedule_with_warmup, EarlyStoppingCallback
from torch.optim import AdamW

from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, XLMRobertaConfig
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit , StratifiedKFold

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
    print(f'이 실험은 seed {seed}로 고정되었습니다.')


# 평가를 위한 metrics function.
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

def train():

  seed_everything(90) # random_seed
  # load model and tokenizer
  MODEL_NAME = "xlm-roberta-large"
  # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # KoCharElectra-Base
  # model = ElectraModel.from_pretrained("monologg/kocharelectra-base-discriminator")
  # tokenizer = KoCharElectraTokenizer.from_pretrained("monologg/kocharelectra-base-discriminator")
  
  # koelectra-base-v3
  # tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
  
  tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

  dataset = pd.read_csv("/opt/ml/input/data/train/my_train.tsv", delimiter='\t')
  label  = dataset['label'].values
  print(tokenizer.tokenize("나는 걸어가고 있는 중입니다."))
  
  
  cv = StratifiedShuffleSplit(n_splits=5, test_size = 0.2, train_size= 1-0.2 ,random_state=90)
  for idx , (train_idx , val_idx) in enumerate(cv.split(dataset, label)):

    train = dataset.iloc[train_idx]
    val = dataset.iloc[val_idx]

    # train.to_csv('/opt/ml/input/data/train/train_train.tsv', sep='\t', header=None, index=False)
    # val.to_csv('/opt/ml/input/data/train/train_dev.tsv', sep='\t', header=None, index=False)
    # train_dataset = load_data("/opt/ml/input/data/train/train_train.tsv")
    # val_dataset = load_data("/opt/ml/input/data/train/train_dev.tsv")

    tokenized_train = tokenized_dataset(train, tokenizer)
    tokenized_val = tokenized_dataset(val, tokenizer)

    train_y = label[train_idx]
    val_y = label[val_idx]

        # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_y)
    RE_valid_dataset = RE_Dataset(tokenized_val, val_y)
  

    output_dir = './result/xlm-roberta-large-3' + str(idx)
    #for ner dataset
    
    
    # train, dev = train_test_split(dataset, test_size=0.2, random_state=90)


    # train_dataset = pd.read_csv("/opt/ml/input/data/train/ner_train_ver2.tsv", delimiter='\t')
    # train_label = train_dataset['label'].values
    # dev_label = dev_dataset['label'].values
    
    # tokenizing dataset
    # tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    # tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    # RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    # RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)



    # setting model hyperparameter
    # bert_config = BertConfig.from_pretrained(MODEL_NAME)
    # bert_config.num_labels = 42
    # model = BertForSequenceClassification.from_pretrained(MODEL_NAME, config=bert_config) 
    roberta_config = XLMRobertaConfig.from_pretrained(MODEL_NAME)
    roberta_config.num_labels = 42
    #electra_config = ElectraConfig.from_pretrained("monologg/koelectra-base-v3-discriminator")
    #electra_config.num_labels = 42
    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=roberta_config)
    #model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator", config=electra_config) 
    model.parameters
    model.to(device)
    print(torch.cuda.is_available())
    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.


    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=300, num_training_steps=2250, num_cycles=3)
    #early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)


    training_args = TrainingArguments(
      seed=90,
      output_dir=output_dir,          # output directory
      fp16=True,
      dataloader_num_workers=4,
      label_smoothing_factor=0.5,
      #lr_scheduler_type='cosine_with_restarts',
      save_total_limit=1,              # number of total save model.
      save_steps=200,                 # model saving step.
      num_train_epochs=10,              # total number of training epochs
      #learning_rate=5e-5,               # learning_rate
      per_device_train_batch_size=32,  # batch size per device during training
      per_device_eval_batch_size=32,   # batch size for evaluation
      #warmup_steps=300,                # number of warmup steps for learning rate scheduler
      #weight_decay=0.01,               # strength of weight decay
      logging_dir='./logs',            # directory for storing logs
      logging_steps=100,              # log saving step.
      load_best_model_at_end=True,
      metric_for_best_model="accuracy",
      greater_is_better = True,
      evaluation_strategy='steps', # evaluation strategy to adopt during training
                                  # `no`: No evaluation during training.
                                  # `steps`: Evaluate every `eval_steps`.
                                  # `epoch`: Evaluate every end of epoch.
      eval_steps = 200,            # evaluation step.
      #report_to='wandb'
    )
    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=RE_train_dataset,         # training dataset
      eval_dataset=RE_valid_dataset,             # evaluation dataset
      compute_metrics=compute_metrics,         # define metrics function
      optimizers=[optimizer, scheduler]
      #callbacks=[early_stopping]
    )

    # train model
    trainer.train()

    model.cpu()
    del model
    torch.cuda.empty_cache()


def main():
  train()

if __name__ == '__main__':
  main()
  