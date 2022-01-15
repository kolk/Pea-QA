# Pea-QA
Parameter-Efficient Abstractive Question Answering over Tables and over Text

Datasets present in data directory:
1. Cleaned version of Tablesum: https://github.com/kolk/Pea-QA/tree/main/data/tablesum/data/02-simplified_tables.json
+ 'pageTitle': webpage title
+ 'caption': table title
+ 'headings': table headers
+ 'rows' : original table provided by Tablesum dataset
+ **'simplified_table'**: optionally present.  Linearized Regular representation of hierarchical table
3. FeTaQA 

Arguments for datasets to adapter-tune:
+ **FeTaQA**: fetaqa
+ **Tablesum**: tablesum
+ **NarrativeQA**: narrativeqa

To adapter-tune:
```
python train.py  --adapter_tune "fetaqa" \
                 --adapter_config "houlsby" \
                 --num_train_epochs 15 \
                 --learning_rate 6e-4 \
                 --lr_scheduler linear \
                 --seed 6 \
                 --output_dir "saved_models/fetaqa_adaptertune" \
                 --pretrained_model_name "facebook/bart-large" \
                 --decoder_max_length 100 \
                 --dataset_name "fetaqa"
```

To fine-tune:
```
python train.py  --num_train_epochs 15 \
                 --learning_rate 6e-4 \
                 --lr_scheduler linear  \
                 --seed 6 \
                 --output_dir "saved_models/fetaqa_finetune" \
                 --pretrained_model_name "facebook/bart-large" \
                 --decoder_max_length 100 \
                 --dataset_name "fetaqa"
```

To evaluate:
```
python evaluate.py --batch_size 2 \
                   --pretrained_model_name "facebook/bart-large" \
                   --adapter_model_name "saved_models/fetaqa_adaptertune/checkpoint-x"
```
