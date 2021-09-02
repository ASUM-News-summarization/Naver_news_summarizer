import numpy as np
import datasets
from transformers import (
    PreTrainedTokenizerFast,
    BartForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from tabulate import tabulate
import nltk

model = BartForConditionalGeneration.from_pretrained("hyunwoongko/kobart")
model.config.__dict__['max_length'] = 64
print('modified model config max_len 20 -> 64...')
tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart")

## train 코드 
