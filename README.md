# Pea-QA: Parameter-Efficient Abstractive Question Answering over Tables or Text

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
                   --adapter_name "fetaqa"
                   --dataset "fetaqa"
```

To cite:
```
@inproceedings{pal-etal-2022-parameter,
    title = "Parameter-Efficient Abstractive Question Answering over Tables or Text",
    author = "Pal, Vaishali  and
      Kanoulas, Evangelos  and
      Rijke, Maarten",
    booktitle = "Proceedings of the Second DialDoc Workshop on Document-grounded Dialogue and Conversational Question Answering",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.dialdoc-1.5",
    pages = "41--53",
    abstract = "A long-term ambition of information seeking QA systems is to reason over multi-modal contexts and generate natural answers to user queries. Today, memory intensive pre-trained language models are adapted to downstream tasks such as QA by fine-tuning the model on QA data in a specific modality like unstructured text or structured tables. To avoid training such memory-hungry models while utilizing a uniform architecture for each modality, parameter-efficient adapters add and train small task-specific bottle-neck layers between transformer layers. In this work, we study parameter-efficient abstractive QA in encoder-decoder models over structured tabular data and unstructured textual data using only 1.5{\%} additional parameters for each modality. We also ablate over adapter layers in both encoder and decoder modules to study the efficiency-performance trade-off and demonstrate that reducing additional trainable parameters down to 0.7{\%}-1.0{\%} leads to comparable results. Our models out-perform current state-of-the-art models on tabular QA datasets such as Tablesum and FeTaQA, and achieve comparable performance on a textual QA dataset such as NarrativeQA using significantly less trainable parameters than fine-tuning.",
}
```
