from transformers import AutoTokenizer, AutoConfig
from transformers.models.auto import AutoModelForSeq2SeqLM
from fetaqa import FeTaQaDataset, FeTaQaProcessor
from tablesum import TableSumDataset, TableSumProcessor
from narrativeqa import NarrativeQADataset, NarrativeQAProcessor
from collections import defaultdict
import argparse
import torch
from torch.utils.data import random_split

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=1, type=int, help="inference batch size")
parser.add_argument("--cpu", type=bool, help="Load the model in cpu")
parser.add_argument("--pretrained_model_name", default=None, type=str,
                    help="huggingface pretrained language model name or local path to language model")
parser.add_argument("--adapter_model_name", default=None, type=str, help="modelname or local path to model")
parser.add_argument("--dataset", type="str", help="Dataset from [narrativeqa, tablesum, fetaqa]")
parser.add_argument("--adapter_name", type=str, default=None, help='adapter name for adapter-tuning')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.adapter_model_name)  # args.pretrained_model_name)#"facebook/bart-large")
config = AutoConfig.from_pretrained(args.pretrained_model_name)  # "facebook/bart-large")
model = AutoModelForSeq2SeqLM.from_pretrained(
    args.adapter_model_name)
if args.adapter_name:
    print(f"Activating adapter {args.adapter_name}")
    model.set_active_adapters(args.adapter_name)
use_cuda = False if args.cpu else True
device = torch.device("cuda" if use_cuda else "cpu")
model.to(device).eval()
outputs = defaultdict(list)

print('Creating test dataset')
if args.dataset.lower() in "narrativeqa":
    test_dataset = NarrativeQADataset(split="test", use_multiprocessing=False, process_count=1)
    print('Creating test processor')
    test_processor = NarrativeQAProcessor(test_dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          decoder_max_length=100,
                                          tokenizer=tokenizer,
                                          decoder_start_token_id=config.decoder_start_token_id,
                                          is_test=True)
elif args.dataset.lower() in "fetaqa":
    test_dataset = FeTaQaDataset(data_directory="data/FeTaQA/data/", split="test", use_multiprocessing=False)
    test_processor = FeTaQaProcessor(test_dataset=test_dataset,
                                     batch_size=args.batch_size,
                                     decoder_max_length=512,
                                     tokenizer=tokenizer,
                                     decoder_start_token_id=config.decoder_start_token_id,
                                     is_test=True)
elif args.dataset.lower() in "tablesum":
    tablesum_data = TableSumDataset(data_directory="data/sigir2020-tablesum/data/", use_multiprocessing=False)
    train_set_size = int(len(tablesum_data) * 0.8)
    valid_set_size = len(tablesum_data) - train_set_size
    train_set, valid_set = random_split(tablesum_data, [train_set_size, valid_set_size],
                                        generator=torch.Generator().manual_seed(42))
    test_dataset = valid_set
    print('Creating test processor')
    test_processor = TableSumProcessor(test_dataset=test_dataset,
                                       batch_size=args.batch_size,
                                       decoder_max_length=300,
                                       tokenizer=tokenizer,
                                       decoder_start_token_id=config.decoder_start_token_id,
                                       is_test=True)
print('Starting Inference')
for i, batch in enumerate(test_processor.test_generator):
    batch_sz = len(batch["input_ids"])
    question = [tokenizer.decode(samp.to("cuda"), skip_special_tokens=True, clean_up_tokenizatimodon_spaces=False) for
                samp in batch["input_ids"]]
    prediction = model.generate(batch["input_ids"].to("cuda"), num_beams=5, return_dict_in_generate=True,
                                output_scores=True, max_length=200)
    seq_len = prediction["sequences"].shape[1]
    answer = [tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True) for pred in
              prediction['sequences']]
    target = [tokenizer.decode(samp, skip_special_tokens=True, clean_up_tokenization_spaces=False) for samp in
              batch["labels"]]
    for ques, tgt, ans in zip(question, target, answer):
        print("question:", ques)
        print("target:", tgt)
        print("prediction:", ans)
        print()

