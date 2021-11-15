
# -*- coding: utf-8 -*-
import json
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, random_split
from datasets import load_dataset, load_metric
import transformers
from transformers import AutoModelForSeq2SeqLM, Trainer, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer, AutoConfig
from transformers.models.bart import BartForConditionalGeneration, BartTokenizer, BartConfig
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers import HoulsbyConfig, PfeifferConfig
from transformers.adapters.configuration import AdapterConfig
from transformers.models.auto import AutoModelWithHeads, AutoModelForSeq2SeqLM
from transformers import set_seed
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
import os
from rouge_score import rouge_scorer, scoring
import argparse
from fetaqa import FeTaQaDataset, FeTaQaProcessor
from tablesum import TableSumDataset, TableSumProcessor
from narrativeqa import NarrativeQADataset, NarrativeQAProcessor


os.environ["WANDB_DISABLED"] = "true"
arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    # '': get_constant_schedule,             # not supported for now
    # '': get_constant_schedule_with_warmup, # not supported for now
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, help="name of dataset to adapter-tune on")
parser.add_argument("--decoder_max_length", type=int, help="encoder sequence max length")
parser.add_argument("--pretrained_model_name", type=str, default=None, help="prtrained model name")
parser.add_argument("--adapter_tune", type=str, default=None, help='adapter name for adapter-tuning')
parser.add_argument("--adapter_config", type=str, default="houlsby", help="Adapter configs: [houlsby, Pfeiffer]")
parser.add_argument("--leaveout", type=int, nargs="*", help="Adapter layers to leave out")
parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--lr_scheduler", default="linear", choices=arg_to_scheduler_choices,  metavar=arg_to_scheduler_metavar, type=str, help="Learning rate scheduler",)
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--num_workers", default=4, type=int, help="kwarg passed to DataLoader")
parser.add_argument("--num_train_epochs", default=30, type=int)
parser.add_argument("--train_batch_size", default=32, type=int)
parser.add_argument("--eval_batch_size", default=32, type=int)
parser.add_argument("--adafactor", action="store_true")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--seed", type=int, default=24)
parser.add_argument("--cpu", action="store_true", help="train using cpu")

args = parser.parse_args()
print(args)
print('leaveout', args.leaveout)
print('adapter_tune', args.adapter_tune)
print('output_dir', args.output_dir)
print('num_train_epochs', args.num_train_epochs)
print('learning_rate', args.learning_rate)
print('seed', args.seed)
print('')
use_cuda = False if args.cpu else True
device = torch.device("cuda" if use_cuda else "cpu")


experiments2_seed = 123
experiments3_seed = 42
experiments4_seed = 64
experiments5_seed = 5
experiments6_seed = 6

seed = args.seed
def model_init():
    set_seed(args.seed)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model_name)
    if args.adapter_tune:
        if args.leaveout:
            if args.adapter_config.lower() == "houlsby":
                config = HoulsbyConfig(leave_out=list(args.leaveout))
            elif args.adapter_config.lower() == "pfeiffer":
                config = PfeifferConfig(leave_out=list(args.leaveout))
            else:
                config = AdapterConfig(original_ln_after=True,
                                       residual_before_ln=True,
                                       adapter_residual_before_ln=True,
                                       ln_before=True,
                                       ln_after=True,
                                       mh_adapter=True,
                                       output_adapter=True,
                                       non_linearity="relu",
                                       reduction_factor=64,
                                       inv_adapter=None,#: str | None = None,
                                       inv_adapter_reduction_factor=64,
                                       cross_adapter=True,
                                       )
        else:
            if args.adapter_config.lower() == "houlsby":
                config = HoulsbyConfig()
            elif args.adapter_config.lower() == "pfeiffer":
                config = PfeifferConfig()
            else:
                config = AdapterConfig(original_ln_after=True,
                                       residual_before_ln=True,
                                       adapter_residual_before_ln=True,
                                       ln_before=True,
                                       ln_after=True,
                                       mh_adapter=True,
                                       output_adapter=True,
                                       non_linearity="relu",
                                       reduction_factor=64,
                                       inv_adapter=None,
                                       inv_adapter_reduction_factor=64,
                                       cross_adapter=True,
                                       )
        model.add_adapter(args.adapter_tune, config=config)
        model.train_adapter(adapter_setup=args.adapter_tune)
    model.config.max_length=args.decoder_max_length
    model = model.to(device)
    print(model)
    return model


tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
config = AutoConfig.from_pretrained(args.pretrained_model_name)


def get_dataset(dataset_name):
    if dataset_name in "tablesum":
        print("Training with Tablesum Dataset....")
        tablesum_data = TableSumDataset(data_directory="data/tablesum/data/", use_multiprocessing=False)
        train_set_size = int(len(tablesum_data) * 0.8)
        valid_set_size = len(tablesum_data) - train_set_size
        train_set, valid_set = random_split(tablesum_data, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(42))
        test_set = valid_set
    elif dataset_name in "fetaqa":
        print("Training with FeTaQA Dataset...")
        valid_set = FeTaQaDataset(data_directory="data/FeTaQA/data/", split="validation", use_multiprocessing=False)
        test_set = FeTaQaDataset(data_directory="data/FeTaQA/data/", split="test", use_multiprocessing=False)
        train_set = FeTaQaDataset(data_directory="data/FeTaQA/data/", split="train", use_multiprocessing=False)
    elif dataset_name in "narrativeqa":
        print("Training with NarrativeQA Dataset...")
        valid_set =  NarrativeQADataset(split="validation", use_multiprocessing=False)
        test_set =  NarrativeQADataset(split="test", use_multiprocessing=False)
        train_set = NarrativeQADataset(split="train", use_multiprocessing=False)
    return train_set, valid_set, test_set

train_dataset, valid_dataset, test_dataset = get_dataset(args.dataset_name)

def rouge_metric_builder(tokenizer):
    def compute_rouge_metrics(pred):
        """utility to compute ROUGE during training."""
        # All special tokens are removed.
        pred_ids, labels_ids = pred.predictions, pred.label_ids
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()

        for ref, pred in zip(label_str, pred_str):
            print("target:", ref)
            print("pred:", pred)
            score = scorer.score(ref, pred)

            aggregator.add_scores(score)


        result = aggregator.aggregate()
        return {
            "rouge1": round(result['rouge1'].mid.fmeasure, 4),
            "rouge2": round(result['rouge2'].mid.fmeasure, 4),
            "rougeL": round(result['rougeL'].mid.fmeasure, 4),
        }
    return compute_rouge_metrics

rouge_metric_fn = rouge_metric_builder(tokenizer)

def collate(batch):
        """
        Generates tokenized batches
        """
        source = [samp["question"] + " " + samp["document"] for samp in batch]
        target = [samp["target"] for samp in batch]
        tokenized_input = tokenizer.prepare_seq2seq_batch(source, target, max_length=512,
                                                               max_target_length=args.decoder_max_length,
                                                               truncation=True, padding='max_length',
                                                               return_tensors="pt")

        if isinstance(config, BartConfig) or isinstance(config, T5Config):
            decoder_input_ids = shift_tokens_right(tokenized_input['labels'], tokenizer.pad_token_id, config.decoder_start_token_id)
        else:
            decoder_input_ids = tokenized_input['labels']

        return {"input_ids": tokenized_input["input_ids"],
                "attention_mask": tokenized_input["attention_mask"],
                "labels": tokenized_input["labels"],
                "decoder_input_ids": decoder_input_ids}  # tokenized_input["labels"]}#decoder_input_ids["input_ids"]}


train_args = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    no_cuda=args.cpu,
    fp16=True if use_cuda else False,
    save_strategy="epoch",
    save_total_limit = 1,
    logging_steps=100,
    eval_accumulation_steps=8,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=args.learning_rate,
    adam_epsilon=args.adam_epsilon,
    num_train_epochs=args.num_train_epochs,
    warmup_steps=args.warmup_steps,
    seed=seed,
    disable_tqdm=False,
    predict_with_generate=True,
    generation_max_length = 200,
    generation_num_beams = 4,
    load_best_model_at_end=True,
    )

transformers.logging.set_verbosity_info()
trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=collate,
    compute_metrics=rouge_metric_fn,
)

trainer.train()
trainer.save_state()

