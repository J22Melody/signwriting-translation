# signwriting-translation

This repo contains code and documentation for training bilingual and multilingual translation models between spoken languages and signed languages in [SignWriting](https://www.signwriting.org/). We also provide an API server for inferring based on the best trained models.

## Environment

We have the following main dependencies:

- We use Python 3.8.11 (a virtual Python environment is recommended).
- We use [sign-language-datasets](https://github.com/sign-language-processing/datasets) as our training data source.
- We use [SentencePiece](https://github.com/google/sentencepiece) for tokenization.
- We use [joeynmt](https://github.com/joeynmt/joeynmt) for training models translating from signed languages to spoken languages (SIGN to SPOKEN).
- We use [Sockeye](https://github.com/awslabs/sockeye) for training models translating from spoken languages to signed languages (SPOKEN to SIGN), which in turn depends on [MXNet](mxnet.apache.org).


Install common dependencies:

`pip install -r requirements.txt`

`sudo apt-get install sentencepiece`

For SIGN to SPOKEN, install `joeynmt` (a customized version supporting source factors):

`pip install git+ssh://git@github.com/J22Melody/joeynmt.git@factors_complete`

or 

`pip install <path_to_local_joeynmt>`

For SPOKEN to SIGN, install `MXNet` and `Sockeye`:

`pip install --pre -f https://dist.mxnet.io/python 'mxnet>=2.0.0b2021'`

`pip install sockeye==3.0.13`

Note `joeynmt` and `Sockeye` require different versions of the package [sacreBLEU](https://github.com/mjpost/sacrebleu), thus results in version conflict if both installed. 

For training, we recommend two separate environments for the two different settings. For inferring (using the API server), this conflict can be safely ignored because `sacreBLEU` is not used there.

## Data Analysis

`python analyze_dataset.py`

which results in [dataset_stats.txt](https://github.com/J22Melody/signwriting-translation/blob/main/dataset_stats.txt).

For a more visual overview in a [Colab Notebook](https://colab.research.google.com/drive/12_MTjQ-1YD4TCyhnvOlcMCnyA3_BmBCP?usp=sharing).

## Model Training

### 40K SIGN to SPOKEN (Bilingual)

Translate from [Formal SignWriting (FSW)](https://tools.ietf.org/id/draft-slevinski-formal-signwriting-09.html) of American Sign Language to American English.

Prepare data in `./data_bilingual/` directory:

`python ./scripts/fetch_data_bilingual.py`

`sh ./scripts/preprocess_bilingual.sh`

Train the start-of-art model `baseline_multilingual` from scratch:

`python -m joeynmt train ./configs/baseline_transformer_spm_factor_sign+`

Test, postprocess and evaluate it:

`sh ./scripts/evaluate.sh baseline_transformer_spm_factor_sign+`

See full results of all experiments [here](https://github.com/J22Melody/signwriting-translation/blob/main/results_sign2en.csv).

### 100K SIGN to SPOKEN (Multilingual)

Extend the first from a bilingual setting to a multilingual setting, translate from FSW of 4 signed languages to 4 corresponding spoken languages.

Prepare data in `./data/` directory:

`python ./scripts/fetch_data.py`

`sh ./scripts/preprocess.sh`

Train the start-of-art model `baseline_multilingual` from scratch:

`python -m joeynmt train ./configs/baseline_multilingual.yaml`

Test, postprocess and evaluate it:

`sh ./scripts/evaluate_multilingual.sh baseline_multilingual`

We also have an additional multilingual model that contains 21 language pairs, see branch [multilingual_plus](https://github.com/J22Melody/signwriting-translation/tree/multilingual_plus) (not well documented).

See full results of both experiments [here](https://github.com/J22Melody/signwriting-translation/blob/main/results_multilingual.csv).

### 100K SPOKEN to SIGN (Multilingual)

Translate the reverse direction, from 4 spoken languages to FSW of 4 corresponding signed languages.

Prepare data in `./data_reverse/` directory:

`python ./scripts/fetch_data.py`

`sh ./scripts/preprocess_reverse.sh`

`sh ./scripts/sockeye_prepare_factor.sh`

Train the start-of-art model `sockeye_spoken2symbol_factor_0.1` from scratch:

`sh ./scripts/sockeye_train_factor.sh`

Test, postprocess and evaluate it:

`sh ./scripts/sockeye_translate_factor.sh sockeye_spoken2symbol_factor_0.1`

`sh ./scripts/sockeye_evaluate_factor.sh sockeye_spoken2symbol_factor_0.1`

`sh ./scripts/sockeye_evaluate_factor_multilingual.sh sockeye_spoken2symbol_factor_0.1`

See full results of all experiments [here](https://github.com/J22Melody/signwriting-translation/blob/main/results_reverse.csv).

## API Server

(make sure you have the model checkpoint files in place, which are not included in this repo)

Run [Flask](https://flask.palletsprojects.com/) locally for debugging:

`python app.py`

Run with [Gunicorn](https://gunicorn.org/) for deployment:

`gunicorn -w 4 -b 0.0.0.0:3030 app:app`

Example [Supervisor](http://supervisord.org/) config file (`/etc/supervisor/conf.d/gunicorn.conf`):

```
[program:gunicorn]
user=xxx
directory=/home/xxx/signwriting-translation
command=gunicorn -w 4 -b 0.0.0.0:3030 app:app

autostart=true
autorestart=true
stdout_logfile=/home/xxx/log/gunicorn.log
stderr_logfile=/home/xxx/log/gunicorn.err.log
```

See [API_spec.md](https://github.com/J22Melody/signwriting-translation/blob/main/API_spec.md) for API specifications.