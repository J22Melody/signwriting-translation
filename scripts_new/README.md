# SignBank+ 

Scripts for running experiments of factored MT models on the new datasets introduced in the follow-up work [SignBank+](https://github.com/sign-language-processing/signbank-plus/).


## Preprare and preprocess data

Load three datasets:

(plus one for pretraining on *expanded* then finetuning on *cleaned*)

```
sh ./scripts_new/prepare_data.sh original data_new_original
sh ./scripts_new/prepare_data.sh cleaned data_new_cleaned
sh ./scripts_new/prepare_data.sh expanded data_new_expanded
sh ./scripts_new/prepare_data.sh cleaned data_new_expanded_cleaned data_new_expanded
```

Filter an English subset for testing:

```
bash ./scripts_new/find_lines.sh ../../../data/parallel/test/test.source.unique "\$en" > ../../../data/parallel/test/en_ids.txt
bash ./scripts_new/filter_test_set.sh
```

## Sockeye prepare data

From scratch (cleaned):

```
sh ./scripts_new/sockeye_prepare_factor.sh data_new_cleaned data_sockeye_new_cleaned
```

Finetune - use pretrained vocabularay (expanded -> cleaned):

```
sh ./scripts_new/sockeye_prepare_factor.sh data_new_expanded_cleaned data_sockeye_new_expanded_cleaned data_sockeye_new_expanded
```

## Train

From scratch (cleaned):

```
sh scripts_new/sockeye_train_factor.sh data_new_cleaned data_sockeye_new_cleaned cleaned
```

Finetune - use pretrained vocabularay and pretrained model (expanded -> cleaned):

```
sh scripts_new/sockeye_train_factor.sh data_new_expanded_cleaned data_sockeye_new_expanded_cleaned expanded_cleaned expanded
```

## Evaluation

From scratch (cleaned):

```
sh ./scripts_new/sockeye_translate_factor.sh data_new_cleaned cleaned
```

Finetune - use pretrained vocabularay and finedtuned model (expanded -> cleaned):

```
sh ./scripts_new/sockeye_translate_factor.sh data_new_expanded_cleaned expanded_cleaned
```

## Results

| Model | BLEU | chrF | BLEU-en | chrF-en |
|-----------|-----------|-----------|-----------|-----------|
| Original | 6.44 | 22.01 | 12.08 | 26.91 |
| Cleaned | 24.65 | 31.22 | 26.4 | 38.22 |
| Expanded | 19.73 | 25.12 | 21.69 | 32.08 |
| Expanded->Cleaned | 23.88 | 30.06 | 26.68 | 38.22 |

