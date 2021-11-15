# -*- coding: utf-8 -*-

import json
from json import decoder
import random
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, random_split
from transformers import AutoTokenizer
from transformers.models.bart.modeling_bart import shift_tokens_right

import os

os.environ["WANDB_DISABLED"] = "true"


class TableSumDataset(Dataset):
    def __init__(self,
                 data_directory,
                 path: str = None,
                 use_multiprocessing: bool = True,
                 multiprocessing_chunksize: int = -1,
                 process_count: int = 2):
        """
        TableSumDataset: Creates a Pytorch Dataset for the Tablesum (tabular summarzation) data
        :param split: train, validation, test split of data
        :param use_multiprocessing: whether to use multiprocessing or not
        :param multiprocessing_chunksize: size of a chunk to be passed to a multiprocessor
        :param process count: number of multiprocessor
        """
        self.path = path
        self.candidate_summaries = pd.read_csv(os.path.join(data_directory, "04-candidate_summaries.csv"))
        self.raw_tables = json.load(open(os.path.join(data_directory, "02-simplified_tables.json")))
        self.use_multiprocessing = use_multiprocessing
        self.multiprocessing_chunksize = multiprocessing_chunksize
        self.process_count = process_count
        print("self.use_multiprocessing", self.use_multiprocessing)
        self.make_samples()

    def flatten_table(self, table_dictionary):
        """
        Flattens a rectangluar table (list of list) to in a row-major sequence where each column in the row is represented as key:value pair
        :param table: a list of list
        """
        table = table_dictionary["simplified_table"] if "simplified_table" in table_dictionary.keys() else \
        table_dictionary["rows"]
        headers = table[0]
        flattened_table = []
        for row in table[1:]:
            flattened_table.extend([f"{k}: {v} <,> " for k, v in zip(headers, row)])
        return " ".join(flattened_table)

    def preprocess_sample(self, sample):
        """
        Each sample of the preprocessed dataset comprises of a dictionary of format
                                            {"source": str,
                                             "target": str,
                                             }
        Each sample of Tablesum data comprises of a question and its associated list of answers. We split each answer into multiple samples by repeating the question with its unique answer.
        :param sample: A TableSum sample
        :return: A Tablesum sample
        """
        question = "<question> " + sample["Input_context"].strip()
        # question = "question: " + sample["Input_context"].strip()
        table = self.flatten_table(self.raw_tables[sample["Input_url"]])
        document = "<title> " + sample["Input_title"].strip().replace("\n", " ").replace("_",
                                                                                         " ") + " <context> " + table
        # document = "context: " + sample["Input_title"].strip().replace("\n", " ").replace("_", " ") + " " + table + "</s>"
        preprocessed = {"source": question + " " + document,
                        "document": document,
                        "question": question,
                        "target": sample["Answer_description"].strip().replace("\n", " ")}

        return preprocessed

    def make_samples(self):
        """Creates the TableSumDataset list of samples"""
        if self.use_multiprocessing:
            print("make_samples use_multiprocessing")
            if self.multiprocessing_chunksize == -1:
                chunksize = max(len(self.candidate_summaries) // (self.process_count * 2), 500)
            else:
                chunksize = self.multiprocessing_chunksize

            with Pool(self.process_count) as p:
                samples = list(
                    tqdm(p.imap(self.preprocess_sample, self.candidate_summaries, chunksize=chunksize),
                         total=len(self.candidate_summaries),
                         disable=False,
                         )
                )
                self.samples = []
                for samp in samples:
                    self.samples += samp
        else:
            print("make_samples without use_multiprocessing")
            self.samples = []
            for indx in tqdm(self.candidate_summaries.index, disable=False):
                sample = self.candidate_summaries.loc[indx]
                self.samples.append(self.preprocess_sample(sample))
            print('Processing Samples complete')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class TableSumProcessor():
    def __init__(self,
                 training_dataset: Dataset = None,
                 eval_dataset: Dataset = None,
                 tokenizer: AutoTokenizer = None,
                 decoder_start_token_id=0,
                 decoder_max_length: int = 100,
                 test_dataset: Dataset = None,
                 is_test: bool = False,
                 **params):
        """
        Generated tokenized batches for training, evaluation and testing of seq2seq task
        :param training_dataset: A TableSumDataset of training samples
        :param eval_dataset: A TableSumDataset of evaluation samples
        :param docder_max_length:
        :param tokenizer: Tokenizer for tokenizing the samples of BioASQ9Dataset dataset
        :param test_dataset: Optional TableSum of test samples

        """

        self.decoder_max_length = 512
        self.decoder_start_token_id = decoder_start_token_id
        # tokenizer_class = params["tokenizer"] if "tokenizer" in params.keys() else "bert-base-uncased"
        self.tokenizer = tokenizer

        if not is_test:
            self.training_dataset = training_dataset
            self.eval_dataset = eval_dataset
            self.sampler = RandomSampler(data_source=self.training_dataset)
            self.training_generator = DataLoader(self.training_dataset,
                                                 sampler=self.sampler,
                                                 collate_fn=self.collate,
                                                 **params)
            self.eval_generator = DataLoader(self.eval_dataset,
                                             sampler=SequentialSampler(data_source=self.eval_dataset),
                                             collate_fn=self.collate,
                                             **params)
        if is_test:
            self.test_dataset = test_dataset
            self.test_generator = DataLoader(self.test_dataset,
                                             sampler=SequentialSampler(data_source=self.test_dataset),
                                             collate_fn=self.collate_generate,
                                             **params)

    def shift_tokens_right(self, input_ids, pad_token_id):
        """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
        prev_output_tokens = input_ids.clone()
        index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
        prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
        prev_output_tokens[:, 1:] = input_ids[:, :-1]
        return prev_output_tokens

    def collate(self, batch):
        """
        Generates tokenized batches
        """
        tokenized_input = self.tokenizer.prepare_seq2seq_batch(batch["source"], batch["target"], max_length=512,
                                                               max_target_length=self.decoder_max_length,
                                                               truncation=True, padding='max_length',
                                                               return_tensors="pt")

        decoder_input_ids = shift_tokens_right(tokenized_input['labels'], self.tokenizer.pad_token_id)

        return {"input_ids": tokenized_input["input_ids"],
                "attention_mask": tokenized_input["attention_mask"],
                "labels": tokenized_input["labels"],
                "decoder_input_ids": decoder_input_ids}  # tokenized_input["labels"]}#decoder_input_ids["input_ids"]}

    def collate_generate(self, batch):
        """
        Generates tokenized batches
        """
        source = [samp["source"] for samp in batch]
        target = [samp["target"] for samp in batch]
        tokenized_input = self.tokenizer.prepare_seq2seq_batch(source, target, max_length=512,
                                                               max_target_length=100,
                                                               truncation=True, padding='max_length',
                                                               return_tensors="pt")

        decoder_input_ids = shift_tokens_right(tokenized_input['labels'], self.tokenizer.pad_token_id,
                                               self.decoder_start_token_id)

        return {"input_ids": tokenized_input["input_ids"],
                "attention_mask": tokenized_input["attention_mask"],
                "labels": tokenized_input["labels"],
                "decoder_input_ids": decoder_input_ids}