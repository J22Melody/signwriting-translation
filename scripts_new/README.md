# Preprare and preprocess data

Load three datasets:

```
sh ./scripts_new/prepare_data.sh original data_new_original
sh ./scripts_new/prepare_data.sh cleaned data_new_cleaned
sh ./scripts_new/prepare_data.sh expanded data_new_expanded
```

# Sockeye prepare data

From scratch (cleaned):

```
sh ./scripts_new/sockeye_prepare_factor.sh data_new_cleaned data_sockeye_new_cleaned
```

Finetune (expanded -> cleaned):

```
sh ./scripts_new/sockeye_prepare_factor.sh  data_new_cleaned data_sockeye_new_expanded_cleaned data_sockeye_new_expanded
```

# Train

From scratch (cleaned):

```
sh scripts_new/sockeye_train_factor.sh data_new_cleaned data_sockeye_new_cleaned cleaned
```

Finetune (expanded -> cleaned):

```
sh scripts_new/sockeye_train_factor.sh data_new_cleaned data_sockeye_new_expanded_cleaned expanded_cleaned expanded
```

# Evaluation

```
sh ./scripts_new/sockeye_translate_factor.sh data_new_cleaned cleaned
```